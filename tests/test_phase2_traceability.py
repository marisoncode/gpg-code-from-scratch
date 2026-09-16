"""Integration and compliance tests for CPG AI Phase 2: Multi-Entity Investigation & Traceability.

Verifies:
1. Support starting investigation from ANY supported entity type (batch, product, material lot,
   component, operator, equipment, location, deviation, finished drug).
2. Forward traceability chain: material/chemical lot -> batches -> finished drugs.
3. Packaging component traceability: component lot -> batches -> finished drugs.
4. Reverse traceability chain: finished drug -> batch -> input lots (materials, components).
5. Operator history: batches executed, equipment handled, date-aware qualifications, deviations.
6. Equipment history: batches processed, PM status, and linked deviations.
7. Deviation forward impact: directly affected batches vs co-manufactured/shared equipment batches.
8. SRS Section 19 compliance: distinct labeling of [CONFIRMED IMPACT] vs [POTENTIAL IMPACT],
   ensuring potential impact is NEVER presented as confirmed impact.
9. Audit trail captures entity_type and traceability_chain in every audit entry.
10. Pre-query permission gating strictly enforced across all Phase 2 entity types.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.agent.prompts import SYSTEM_PROMPT
from app.core.config import settings
from app.main import app
from app.services import audit_service, facility_api_service
from app.services.ai_service import generate_response
from app.services.audit_service import get_recent_audit_entries, log_audit_entry
from app.services.capabilities import ChatPermissions, ModulePermission
from app.services.facility_api_service import (
    execute_predefined_tool,
    get_component_lot_genealogy,
    get_deviation_impact,
    get_equipment_batch_history,
    get_finished_drug_genealogy,
    get_material_lot_genealogy,
    get_operator_batch_history,
    get_related_batches,
    investigate_entity,
)
from app.services.permissions_service import PermissionDeniedError


@pytest.fixture
def qa_all_permissions() -> ChatPermissions:
    """Full QA access to both Production and Compliance modules."""
    return ChatPermissions(
        Dashboard_assign="All",
        modules=[
            ModulePermission(Module_name="Batch Record", View=True),
            ModulePermission(Module_name="Compliance", View=True),
            ModulePermission(Module_name="Equipment", View=True),
            ModulePermission(Module_name="Inventory", View=True),
        ],
    )


@pytest.fixture
def prod_only_permissions() -> ChatPermissions:
    """Production only access."""
    return ChatPermissions(
        Dashboard_assign="Production",
        modules=[
            ModulePermission(Module_name="Batch Record", View=True),
            ModulePermission(Module_name="Inventory", View=True),
        ],
    )


@pytest.fixture
def comp_only_permissions() -> ChatPermissions:
    """Compliance only access."""
    return ChatPermissions(
        Dashboard_assign="Compliance",
        modules=[
            ModulePermission(Module_name="Compliance", View=True),
        ],
    )


@pytest.fixture
def sales_permissions() -> ChatPermissions:
    """Sales lens (blocked from production and compliance)."""
    return ChatPermissions(
        Dashboard_assign="Sales",
        modules=[],
    )


# ── TEST 1: FORWARD MATERIAL LOT GENEALOGY & SRS SEC 19 IMPACT LABELS ─────────


def test_material_lot_genealogy_distinguishes_confirmed_and_potential_impact(qa_all_permissions: ChatPermissions) -> None:
    """
    SRS Section 5 & 19:
    material_lot -> batches -> finished_drugs.
    Direct verified consumption of defective lot or linked OOS -> [CONFIRMED IMPACT].
    Used same lot without confirmed defect -> [POTENTIAL IMPACT].
    """
    mock_mat_container = MagicMock()
    mock_batch_container = MagicMock()

    # Defective raw material lot
    mock_mat_container.query_items.return_value = [
        {"id": "RM-88321", "lot_number": "RM-88321", "quality_status": "REJECTED", "supplier": "SupplierAlpha"}
    ]

    # Two batches: B-101 and B-102 used this material lot
    mock_batch_container.query_items.return_value = [
        {
            "id": "B-101",
            "batch_number": "B-101",
            "product": "ProductX",
            "status": "REJECTED",
            "materials": ["RM-88321"],
            "finished_drugs": ["FD-901"],
            "deviations": ["DEV-11"],
        },
        {
            "id": "B-102",
            "batch_number": "B-102",
            "product": "ProductY",
            "status": "IN_PROGRESS",
            "materials": ["RM-88321"],
            "finished_drugs": ["FD-902"],
            "deviations": [],
        },
    ]

    def container_dispatch(name: str):
        if name == "materials":
            return mock_mat_container
        if name == "batches":
            return mock_batch_container
        return MagicMock()

    with patch("app.services.facility_api_service._get_business_container", side_effect=container_dispatch):
        res = get_material_lot_genealogy("RM-88321", permissions=qa_all_permissions)

        assert res["status"] == "found"
        assert res["lot_id"] == "RM-88321"
        assert res["traceability_chain"] == "material_lot -> batches -> finished_drugs"
        assert "B-101" in res["batches"]
        assert "B-102" in res["batches"]
        assert "FD-901" in res["finished_drugs"]
        assert "FD-902" in res["finished_drugs"]

        # Check impact labels
        confirmed = res["confirmed_impact"]
        potential = res["potential_impact"]

        assert len(confirmed) == 2  # defective lot makes both consumed batches confirmed impact
        assert all(item["impact_type"] == "CONFIRMED" for item in confirmed)

        # Now test when lot was NOT defective, but B-101 had a deviation and B-102 did not
        mock_mat_container.query_items.return_value = [
            {"id": "RM-88321", "lot_number": "RM-88321", "quality_status": "RELEASED"}
        ]
        res2 = get_material_lot_genealogy("RM-88321", permissions=qa_all_permissions)
        conf2 = res2["confirmed_impact"]
        pot2 = res2["potential_impact"]

        assert len(conf2) == 1
        assert conf2[0]["batch_id"] == "B-101"
        assert conf2[0]["impact_type"] == "CONFIRMED"

        assert len(pot2) == 1
        assert pot2[0]["batch_id"] == "B-102"
        assert pot2[0]["impact_type"] == "POTENTIAL"
        # NEVER report potential impact as confirmed impact
        assert pot2[0]["impact_type"] != "CONFIRMED"


# ── TEST 2: COMPONENT LOT GENEALOGY ───────────────────────────────────────────


def test_component_lot_genealogy(qa_all_permissions: ChatPermissions) -> None:
    """Trace packaging component genealogy: component_lot -> batches -> finished_drugs."""
    mock_batch_container = MagicMock()
    mock_batch_container.query_items.return_value = [
        {
            "id": "B-201",
            "batch_number": "B-201",
            "components": ["COMP-501"],
            "finished_drugs": ["FD-555"],
            "status": "RELEASED",
        }
    ]

    with patch("app.services.facility_api_service._get_business_container", return_value=mock_batch_container):
        res = get_component_lot_genealogy("COMP-501", permissions=qa_all_permissions)
        assert res["status"] == "found"
        assert res["lot_id"] == "COMP-501"
        assert res["traceability_chain"] == "component_lot -> batches -> finished_drugs"
        assert "B-201" in res["batches"]
        assert "FD-555" in res["finished_drugs"]
        assert len(res["potential_impact"]) == 1
        assert res["potential_impact"][0]["impact_type"] == "POTENTIAL"


# ── TEST 3: REVERSE TRACEABILITY CHAIN (FINISHED DRUG -> INPUT LOTS) ───────────


def test_finished_drug_reverse_genealogy(qa_all_permissions: ChatPermissions) -> None:
    """Reverse genealogy: finished_drug -> batch -> input_lots (materials + components)."""
    mock_batch_container = MagicMock()
    mock_batch_container.query_items.return_value = [
        {
            "id": "B-300",
            "batch_number": "B-300",
            "finished_drugs": ["DRUG-901"],
            "materials": ["RM-100", "RM-200"],
            "components": ["COMP-10"],
        }
    ]

    with patch("app.services.facility_api_service._get_business_container", return_value=mock_batch_container):
        res = get_finished_drug_genealogy("DRUG-901", permissions=qa_all_permissions)
        assert res["status"] == "found"
        assert res["finished_drug_id"] == "DRUG-901"
        assert res["traceability_chain"] == "finished_drug -> batch -> input_lots"
        assert res["batches"] == ["B-300"]
        assert "RM-100" in res["input_material_lots"]
        assert "RM-200" in res["input_material_lots"]
        assert "COMP-10" in res["input_component_lots"]


# ── TEST 4: OPERATOR BATCH HISTORY ─────────────────────────────────────────────


def test_operator_batch_history(qa_all_permissions: ChatPermissions) -> None:
    """Retrieve batches executed by operator, equipment handled, and qualifications."""
    mock_tr_container = MagicMock()
    mock_batch_container = MagicMock()

    mock_tr_container.query_items.return_value = [
        {"operator_id": "OP-017", "course_name": "Aseptic Technique", "status": "CURRENT"}
    ]
    mock_batch_container.query_items.return_value = [
        {"id": "B-401", "operators": ["OP-017"], "equipment": ["EQ-102", "EQ-105"], "deviations": ["DEV-99"]}
    ]

    def container_dispatch(name: str):
        if name == "training":
            return mock_tr_container
        if name == "batches":
            return mock_batch_container
        return MagicMock()

    with patch("app.services.facility_api_service._get_business_container", side_effect=container_dispatch):
        res = get_operator_batch_history("OP-017", permissions=qa_all_permissions)
        assert res["status"] == "found"
        assert res["operator_id"] == "OP-017"
        assert res["traceability_chain"] == "operator -> batches -> equipment -> quality_events"
        assert "B-401" in res["batches"]
        assert "EQ-102" in res["equipment_handled"]
        assert "DEV-99" in res["deviations_linked"]
        assert len(res["qualifications"]) == 1


# ── TEST 5: EQUIPMENT BATCH HISTORY & OVERDUE PM POTENTIAL IMPACT ─────────────


def test_equipment_batch_history_pm_impact(qa_all_permissions: ChatPermissions) -> None:
    """Equipment batch history distinguishes confirmed deviations from overdue PM potential impact."""
    mock_eq_container = MagicMock()
    mock_batch_container = MagicMock()

    mock_eq_container.query_items.return_value = [
        {"id": "EQ-102", "asset_id": "EQ-102", "pm_status": "OVERDUE", "status": "OVERDUE"}
    ]
    mock_batch_container.query_items.return_value = [
        {"id": "B-501", "equipment": ["EQ-102"], "deviations": ["DEV-01"]},
        {"id": "B-502", "equipment": ["EQ-102"], "deviations": []},
    ]

    def container_dispatch(name: str):
        if name == "equipment":
            return mock_eq_container
        if name == "batches":
            return mock_batch_container
        return MagicMock()

    with patch("app.services.facility_api_service._get_business_container", side_effect=container_dispatch):
        res = get_equipment_batch_history("EQ-102", permissions=qa_all_permissions)
        assert res["status"] == "found"
        assert res["traceability_chain"] == "equipment -> batches -> deviations"
        assert len(res["confirmed_impact"]) == 1
        assert res["confirmed_impact"][0]["batch_id"] == "B-501"
        assert res["confirmed_impact"][0]["impact_type"] == "CONFIRMED"

        assert len(res["potential_impact"]) == 1
        assert res["potential_impact"][0]["batch_id"] == "B-502"
        assert res["potential_impact"][0]["impact_type"] == "POTENTIAL"
        assert "overdue PM" in res["potential_impact"][0]["reason"]


# ── TEST 6: DEVIATION FORWARD IMPACT (CONFIRMED VS CO-MANUFACTURED POTENTIAL) ─


def test_deviation_impact_shared_equipment(qa_all_permissions: ChatPermissions) -> None:
    """
    Deviation -> directly affected batch (confirmed) + co-manufactured on shared line (potential).
    """
    mock_dev_container = MagicMock()
    mock_batch_container = MagicMock()

    mock_dev_container.query_items.return_value = [
        {"id": "DEV-445", "equipment_id": "EQ-102", "title": "Temperature Excursion"}
    ]
    # First call: directly affected batch
    # Second call: shared equipment batch
    mock_batch_container.query_items.side_effect = [
        [{"id": "B-601", "deviations": ["DEV-445"]}],
        [{"id": "B-602", "equipment": ["EQ-102"]}],
    ]

    def container_dispatch(name: str):
        if name == "deviations":
            return mock_dev_container
        if name == "batches":
            return mock_batch_container
        return MagicMock()

    with patch("app.services.facility_api_service._get_business_container", side_effect=container_dispatch):
        res = get_deviation_impact("DEV-445", permissions=qa_all_permissions)
        assert res["status"] == "found"
        assert res["traceability_chain"] == "deviation -> affected_batches"
        assert len(res["confirmed_impact"]) == 1
        assert res["confirmed_impact"][0]["batch_id"] == "B-601"
        assert res["confirmed_impact"][0]["impact_type"] == "CONFIRMED"

        assert len(res["potential_impact"]) == 1
        assert res["potential_impact"][0]["batch_id"] == "B-602"
        assert res["potential_impact"][0]["impact_type"] == "POTENTIAL"
        assert "shared equipment" in res["potential_impact"][0]["reason"]


# ── TEST 7: INVESTIGATE ENTITY FROM ANY STARTING TYPE ─────────────────────────


def test_investigate_entity_universal_entry(qa_all_permissions: ChatPermissions) -> None:
    """Verify investigate_entity routes properly for all supported entity types."""
    mock_container = MagicMock()
    mock_container.query_items.return_value = [{"id": "GEN-01", "status": "OK"}]

    with patch("app.services.facility_api_service._get_business_container", return_value=mock_container):
        # 1. Start from material lot
        res_mat = investigate_entity("material_lot", "RM-88321", permissions=qa_all_permissions)
        assert res_mat["traceability_chain"] == "material_lot -> batches -> finished_drugs"

        # 2. Start from component
        res_comp = investigate_entity("component", "COMP-501", permissions=qa_all_permissions)
        assert res_comp["traceability_chain"] == "component_lot -> batches -> finished_drugs"

        # 3. Start from finished drug
        res_drug = investigate_entity("finished_drug", "FD-901", permissions=qa_all_permissions)
        assert res_drug["traceability_chain"] == "finished_drug -> batch -> input_lots"

        # 4. Start from operator
        res_op = investigate_entity("operator", "OP-017", permissions=qa_all_permissions)
        assert res_op["traceability_chain"] == "operator -> batches -> equipment -> quality_events"

        # 5. Start from equipment
        res_eq = investigate_entity("equipment", "EQ-102", permissions=qa_all_permissions)
        assert res_eq["traceability_chain"] == "equipment -> batches -> deviations"

        # 6. Start from deviation
        res_dev = investigate_entity("deviation", "DEV-445", permissions=qa_all_permissions)
        assert res_dev["traceability_chain"] == "deviation -> affected_batches"

        # 7. Start from location
        res_loc = investigate_entity("location", "CLEANROOM-A", permissions=qa_all_permissions)
        assert res_loc["traceability_chain"] == "location -> em -> batches"

        # 8. Start from product
        res_prod = investigate_entity("product", "Aspirin", permissions=qa_all_permissions)
        assert res_prod["traceability_chain"] == "product -> batches"


# ── TEST 8: OFFLINE AI INTENT & EVIDENCE FORMATTING ([CONFIRMED] VS [POTENTIAL]) ─


@pytest.mark.asyncio
async def test_ai_response_formats_confirmed_vs_potential_impact_distinctly(qa_all_permissions: ChatPermissions) -> None:
    """
    Ensure offline AI dispatch formats response with explicit [CONFIRMED IMPACT]
    and [POTENTIAL IMPACT] tags, and never presents potential as confirmed.
    """
    mock_batch_container = MagicMock()
    mock_mat_container = MagicMock()

    mock_mat_container.query_items.return_value = [
        {"id": "RM-88321", "lot_number": "RM-88321", "quality_status": "RELEASED"}
    ]
    mock_batch_container.query_items.return_value = [
        {"id": "B-101", "batch_number": "B-101", "materials": ["RM-88321"], "deviations": ["DEV-1"]},
        {"id": "B-102", "batch_number": "B-102", "materials": ["RM-88321"], "deviations": []},
    ]

    def container_dispatch(name: str):
        if name == "materials":
            return mock_mat_container
        if name == "batches":
            return mock_batch_container
        return MagicMock()

    with patch("app.services.facility_api_service._get_business_container", side_effect=container_dispatch):
        result = await generate_response(
            "Trace material lot RM-88321 genealogy",
            permissions=qa_all_permissions,
        )

        assert "[FACT]" in result.text
        assert "material_lot -> batches -> finished_drugs" in result.text
        assert "[CONFIRMED IMPACT]" in result.text
        assert "Batch B-101" in result.text
        assert "[POTENTIAL IMPACT]" in result.text
        assert "Batch B-102" in result.text
        assert result.entity_type == "material_lot"
        assert result.traceability_chain == "material_lot -> batches -> finished_drugs"


# ── TEST 9: AUDIT LOG CAPTURES ENTITY_TYPE AND TRACEABILITY_CHAIN ──────────────


@pytest.mark.asyncio
async def test_audit_entry_records_entity_type_and_traceability_chain() -> None:
    """Audit log entries must contain entity_type and traceability_chain."""
    entry = await log_audit_entry(
        user_id="usr-compliance-99",
        role="QA Specialist",
        raw_query="Trace component lot COMP-501",
        functions_called=[{"name": "get_component_lot_genealogy", "parameters": {"lot_id": "COMP-501"}}],
        retrieved_record_ids=["COMP-501", "B-201"],
        model_version="gpt-4o-mini",
        final_response="[FACT] Component Lot COMP-501 verified.",
        entity_type="component",
        traceability_chain="component_lot -> batches -> finished_drugs",
    )

    assert entry["entity_type"] == "component"
    assert entry["traceability_chain"] == "component_lot -> batches -> finished_drugs"
    assert entry["user_id"] == "usr-compliance-99"

    recent = get_recent_audit_entries(limit=5)
    matched = [e for e in recent if e.get("id") == entry["id"]]
    assert len(matched) == 1
    assert matched[0]["entity_type"] == "component"
    assert matched[0]["traceability_chain"] == "component_lot -> batches -> finished_drugs"


@pytest.mark.asyncio
async def test_chat_endpoint_populates_phase2_audit_metadata(qa_all_permissions: ChatPermissions) -> None:
    """POST /chat must forward entity_type and traceability_chain to audit_service."""
    import jwt

    token = jwt.encode(
        {"userId": "user-qa-trace", "name": "Trace QA"},
        "dummy-secret-key-at-least-32-chars-long!",
        algorithm="HS256",
    )

    client = TestClient(app)

    mock_mat_container = MagicMock()
    mock_batch_container = MagicMock()
    mock_mat_container.query_items.return_value = [{"id": "RM-88321", "quality_status": "RELEASED"}]
    mock_batch_container.query_items.return_value = [{"id": "B-101", "materials": ["RM-88321"]}]

    def container_dispatch(name: str):
        if name == "materials":
            return mock_mat_container
        if name == "batches":
            return mock_batch_container
        return MagicMock()

    with patch("app.services.facility_api_service._get_business_container", side_effect=container_dispatch), \
         patch("app.api.chat.resolve_permissions", return_value=qa_all_permissions):
        payload = {
            "message": "Trace material lot RM-88321 genealogy",
        }
        resp = client.post(
            "/chat",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        assert resp.status_code == 200

        recent = get_recent_audit_entries(limit=5)
        # Find the audit entry for this request
        matching = [e for e in recent if "RM-88321" in e.get("raw_query", "")]
        assert len(matching) >= 1
        entry = matching[-1]
        assert entry["entity_type"] == "material_lot"
        assert "material_lot -> batches -> finished_drugs" in entry["traceability_chain"]


# ── TEST 10: ROLE-BASED PRE-QUERY PERMISSION GATING ON ALL PHASE 2 ENTITIES ───


def test_permission_gate_blocks_sales_and_unauthorized_roles(
    sales_permissions: ChatPermissions,
    prod_only_permissions: ChatPermissions,
    comp_only_permissions: ChatPermissions,
) -> None:
    """Pre-query verification must reject unauthorized roles without hitting Cosmos DB."""
    mock_container = MagicMock()

    with patch("app.services.facility_api_service._get_business_container", return_value=mock_container):
        # 1. Sales lens is blocked from everything
        with pytest.raises(PermissionDeniedError):
            get_material_lot_genealogy("RM-101", permissions=sales_permissions)
        with pytest.raises(PermissionDeniedError):
            get_operator_batch_history("OP-101", permissions=sales_permissions)
        with pytest.raises(PermissionDeniedError):
            get_deviation_impact("DEV-101", permissions=sales_permissions)
        with pytest.raises(PermissionDeniedError):
            get_finished_drug_genealogy("FD-101", permissions=sales_permissions)
        mock_container.query_items.assert_not_called()

        # 2. Production lens cannot access pure Compliance deviation impact or operator training
        with pytest.raises(PermissionDeniedError):
            get_deviation_impact("DEV-101", permissions=prod_only_permissions)
        with pytest.raises(PermissionDeniedError):
            get_operator_batch_history("OP-101", permissions=prod_only_permissions)
        mock_container.query_items.assert_not_called()

        # 3. Compliance lens cannot access pure Production material or finished drug genealogy
        with pytest.raises(PermissionDeniedError):
            get_material_lot_genealogy("RM-101", permissions=comp_only_permissions)
        with pytest.raises(PermissionDeniedError):
            get_finished_drug_genealogy("FD-101", permissions=comp_only_permissions)
        mock_container.query_items.assert_not_called()
