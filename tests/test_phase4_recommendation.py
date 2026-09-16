"""Integration and compliance tests for CPG AI Phase 4: AI-Assisted Batch Creation Recommendation.

ADVISORY-ONLY: Never creates, releases, or approves an actual batch record (SRS Section 3.2).
Strictly implements SRS Section 7.1, 7.3, and Section 19 Precedence Rules:
- Hard eligibility constraints FIRST: eliminates unavailable, expired, unapproved, or unqualified
  options BEFORE historical ranking occurs.
- Historical success does NOT create authorization or qualification.
- Dedicated test proving an ineligible option is NEVER included despite high historical performance.
- Verifies explicit flag 'approval_status': 'DRAFT — requires authorized human approval' and is_advisory_only: True.
- Verifies audit trail logging and pre-query permission gating.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import audit_service, facility_api_service
from app.services.ai_service import generate_response
from app.services.capabilities import ChatPermissions, ModulePermission
from app.services.permissions_service import PermissionDeniedError
from app.services.recommendation_engine import ADVISORY_STATUS_LABEL, recommend_batch_configuration


@pytest.fixture
def qa_all_permissions() -> ChatPermissions:
    """QA/Production user with full permissions."""
    return ChatPermissions(
        Dashboard_assign="All",
        modules=[
            ModulePermission(Module_name="Batch Record", View=True),
            ModulePermission(Module_name="Compliance", View=True),
            ModulePermission(Module_name="Equipment", View=True),
            ModulePermission(Module_name="Inventory", View=True),
            ModulePermission(Module_name="Production", View=True),
        ],
    )


@pytest.fixture
def sales_permissions() -> ChatPermissions:
    """Sales user lacking production and batch permissions."""
    return ChatPermissions(
        Dashboard_assign="Sales",
        modules=[],
    )


# ── TEST 1: DEDICATED PRECEDENCE TEST (SRS SECTION 19) ────────────────────────
def test_hard_eligibility_precedence_over_historical_performance(qa_all_permissions: ChatPermissions) -> None:
    """
    SRS Section 19 Hard Precedence Rule:
    'Current eligibility rules take precedence over historical performance.'
    'Historical success does not create authorization or qualification.'

    Scenario:
    - Operator OP-STAR has 100% past yield across 50 batches, but training is EXPIRED.
    - Equipment EQ-FAST has 100% past yield across 50 batches, but PM status is OVERDUE.
    - Material RM-OLD has 100% past yield, but quality status is REJECTED.
    - Operator OP-QUALIFIED has 97% past yield, but training is CURRENT.
    - Equipment EQ-QUALIFIED has 97% past yield, but PM is QUALIFIED.
    - Material RM-QUALIFIED is RELEASED.

    VERIFICATION:
    - OP-STAR, EQ-FAST, and RM-OLD MUST BE DISQUALIFIED in Step (d) before ranking!
    - They MUST NEVER appear in recommended_configuration or alternative_configurations!
    - They MUST appear in disqualifying_factors!
    - Only OP-QUALIFIED, EQ-QUALIFIED, and RM-QUALIFIED may be recommended.
    """
    mock_batches = [
        # OP-STAR and EQ-FAST had 100% yield historically!
        {
            "id": "B-PAST-1",
            "product": "Aspirin 500mg",
            "formulation_version": "v2.1",
            "room": "Cleanroom-A",
            "operators": ["OP-STAR"],
            "equipment": ["EQ-FAST"],
            "materials": ["RM-OLD"],
            "yield_percentage": 100.0,
            "status": "COMPLETED",
        },
        {
            "id": "B-PAST-2",
            "product": "Aspirin 500mg",
            "formulation_version": "v2.1",
            "room": "Cleanroom-A",
            "operators": ["OP-QUALIFIED"],
            "equipment": ["EQ-QUALIFIED"],
            "materials": ["RM-QUALIFIED"],
            "yield_percentage": 97.5,
            "status": "COMPLETED",
        },
    ]

    mock_materials = [
        {"id": "RM-OLD", "lot_number": "RM-OLD", "quality_status": "REJECTED"},
        {"id": "RM-QUALIFIED", "lot_number": "RM-QUALIFIED", "quality_status": "RELEASED"},
    ]

    mock_equipment = [
        {"id": "EQ-FAST", "asset_id": "EQ-FAST", "pm_status": "OVERDUE"},
        {"id": "EQ-QUALIFIED", "asset_id": "EQ-QUALIFIED", "pm_status": "QUALIFIED"},
    ]

    mock_operators = [
        {"operator_id": "OP-STAR", "status": "EXPIRED", "course_name": "Aseptic Technique"},
        {"operator_id": "OP-QUALIFIED", "status": "CURRENT", "course_name": "Aseptic Technique"},
    ]

    def mock_get_container(container_name: str) -> MagicMock:
        c = MagicMock()
        if container_name == "batches":
            c.query_items.return_value = iter(mock_batches)
        elif container_name == "materials":
            c.query_items.return_value = iter(mock_materials)
        elif container_name == "equipment":
            c.query_items.return_value = iter(mock_equipment)
        elif container_name == "training":
            c.query_items.return_value = iter(mock_operators)
        else:
            c.query_items.return_value = iter([])
        return c

    with patch.object(facility_api_service, "_get_business_container", side_effect=mock_get_container):
        result = recommend_batch_configuration(
            product_id="Aspirin 500mg",
            target_quantity=1000.0,
            permissions=qa_all_permissions,
            user_id="lead_planner_01",
            role="Manufacturing Planner",
        )

        # 1. Verification of status
        assert result["status"] == "DRAFT_REQUIRES_HUMAN_APPROVAL"
        assert result["approval_status"] == ADVISORY_STATUS_LABEL
        assert result["is_advisory_only"] is True

        # 2. Hard Precedence: OP-STAR, EQ-FAST, RM-OLD eliminated
        disqualified_ids = [d["resource_id"] for d in result["disqualifying_factors"]]
        assert "OP-STAR" in disqualified_ids
        assert "EQ-FAST" in disqualified_ids
        assert "RM-OLD" in disqualified_ids

        # 3. Disqualification reason and step must cite Step (d)
        for d in result["disqualifying_factors"]:
            assert "Step (d)" in d["eliminated_in_step"]

        # 4. Recommended configuration MUST NOT contain any disqualified resource
        rec = result["recommended_configuration"]
        assert rec is not None
        assert "OP-STAR" not in rec["operators"]
        assert "EQ-FAST" not in rec["equipment"]
        assert "RM-OLD" not in rec["materials"]

        # 5. Recommended configuration MUST contain the qualified resource
        assert "OP-QUALIFIED" in rec["operators"]
        assert "EQ-QUALIFIED" in rec["equipment"]
        assert "RM-QUALIFIED" in rec["materials"]

        # 6. Alternative configurations must ALSO be free from disqualified resources
        for alt in result["alternative_configurations"]:
            assert "OP-STAR" not in alt["operators"]
            assert "EQ-FAST" not in alt["equipment"]
            assert "RM-OLD" not in alt["materials"]


# ── TEST 2: MANDATORY ADVISORY & DRAFT FLAGS (SRS 3.2 & 7.3) ───────────────────
def test_mandatory_advisory_flags_and_no_autonomous_release(qa_all_permissions: ChatPermissions) -> None:
    """
    SRS Section 3.2 & 7.3:
    - Never creates, writes, or approves an actual batch record.
    - Explicit flag: 'approval_status': 'DRAFT — requires authorized human approval'
    - is_advisory_only: True
    - Requires human approvals must be listed explicitly.
    """
    def mock_get_container(container_name: str) -> MagicMock:
        c = MagicMock()
        if container_name == "batches":
            c.query_items.return_value = iter([{
                "id": "B-100",
                "product": "Ibuprofen 200mg",
                "formulation_version": "v1.0",
                "room": "Cleanroom-B",
                "operators": ["OP-01"],
                "equipment": ["EQ-01"],
                "materials": ["RM-01"],
                "yield_percentage": 99.0,
            }])
        elif container_name == "materials":
            c.query_items.return_value = iter([{"id": "RM-01", "lot_number": "RM-01", "quality_status": "RELEASED"}])
        elif container_name == "equipment":
            c.query_items.return_value = iter([{"id": "EQ-01", "asset_id": "EQ-01", "pm_status": "QUALIFIED"}])
        elif container_name == "training":
            c.query_items.return_value = iter([{"operator_id": "OP-01", "status": "CURRENT"}])
        return c

    with patch.object(facility_api_service, "_get_business_container", side_effect=mock_get_container) as mock_cont:
        result = recommend_batch_configuration(
            product_id="Ibuprofen 200mg",
            permissions=qa_all_permissions,
        )

        assert result["is_advisory_only"] is True
        assert result["approval_status"] == "DRAFT — requires authorized human approval"
        assert result["status"] == "DRAFT_REQUIRES_HUMAN_APPROVAL"

        # Assert no write methods (create_item, upsert_item, replace_item) were ever called on containers
        for call_args in mock_cont.return_value.method_calls:
            method_name = call_args[0]
            assert method_name not in {"create_item", "upsert_item", "replace_item", "delete_item"}

        # Required approvals must be documented
        assert len(result["required_approvals"]) > 0
        assert any("Production Supervisor" in a or "QA" in a or "Quality" in a for a in result["required_approvals"])


# ── TEST 3: ALL REQUIRED FIELDS OF SRS SECTION 7.3 PRESENT ────────────────────
def test_all_section_7_3_fields_present(qa_all_permissions: ChatPermissions) -> None:
    """
    SRS Section 7.3 requires:
    - recommended configuration
    - alternative eligible configurations
    - risk indicator
    - confidence / uncertainty
    - historical sample size
    - top reasons
    - disqualifying / cautionary factors
    - source records
    - required approvals
    - explicit flag: 'approval_status': 'DRAFT — requires authorized human approval'
    - is_advisory_only: True
    """
    def mock_get_container(container_name: str) -> MagicMock:
        c = MagicMock()
        if container_name == "batches":
            c.query_items.return_value = iter([{
                "id": f"B-{i}",
                "product": "Paracetamol",
                "formulation_version": "v3.0",
                "room": "Cleanroom-C",
                "operators": [f"OP-0{i}"],
                "equipment": [f"EQ-0{i}"],
                "materials": [f"RM-0{i}"],
                "yield_percentage": 98.0 + (i * 0.1),
            } for i in range(1, 4)])
        elif container_name == "materials":
            c.query_items.return_value = iter([{"id": f"RM-0{i}", "lot_number": f"RM-0{i}", "quality_status": "RELEASED"} for i in range(1, 4)])
        elif container_name == "equipment":
            c.query_items.return_value = iter([{"id": f"EQ-0{i}", "asset_id": f"EQ-0{i}", "pm_status": "QUALIFIED"} for i in range(1, 4)])
        elif container_name == "training":
            c.query_items.return_value = iter([{"operator_id": f"OP-0{i}", "status": "CURRENT"} for i in range(1, 4)])
        return c

    with patch.object(facility_api_service, "_get_business_container", side_effect=mock_get_container):
        result = recommend_batch_configuration(
            product_id="Paracetamol",
            target_quantity=500.0,
            permissions=qa_all_permissions,
        )

        required_keys = [
            "recommended_configuration",
            "alternative_configurations",
            "risk_indicator",
            "confidence_uncertainty",
            "historical_sample_size",
            "top_reasons",
            "disqualifying_factors",
            "source_records",
            "required_approvals",
            "approval_status",
            "is_advisory_only",
        ]
        for key in required_keys:
            assert key in result, f"Missing required Section 7.3 field: {key}"

        # Risk indicator structure
        assert "risk_score" in result["risk_indicator"]
        assert "risk_level" in result["risk_indicator"]
        assert "config_version" in result["risk_indicator"]

        # Top reasons must not be empty
        assert len(result["top_reasons"]) > 0


# ── TEST 4: BLOCKED WHEN NO ELIGIBLE RESOURCES SURVIVE HARD CONSTRAINTS ────────
def test_blocked_when_no_eligible_resources_exist(qa_all_permissions: ChatPermissions) -> None:
    """
    If all available resources are disqualified under hard constraints, the recommendation
    engine must block configuration creation and return a descriptive blocked status.
    """
    def mock_get_container(container_name: str) -> MagicMock:
        c = MagicMock()
        if container_name == "batches":
            c.query_items.return_value = iter([])
        elif container_name == "materials":
            c.query_items.return_value = iter([{"id": "RM-DEFECTIVE", "quality_status": "REJECTED"}])
        elif container_name == "equipment":
            c.query_items.return_value = iter([{"id": "EQ-BROKEN", "pm_status": "OUT_OF_SERVICE"}])
        elif container_name == "training":
            c.query_items.return_value = iter([{"operator_id": "OP-UNQUALIFIED", "status": "EXPIRED"}])
        return c

    with patch.object(facility_api_service, "_get_business_container", side_effect=mock_get_container):
        result = recommend_batch_configuration(
            product_id="BlockedDrug",
            permissions=qa_all_permissions,
        )

        assert result["status"] == "BLOCKED_NO_ELIGIBLE_CONFIGURATION"
        assert result["recommended_configuration"] is None
        assert result["approval_status"] == ADVISORY_STATUS_LABEL
        assert result["is_advisory_only"] is True
        assert len(result["disqualifying_factors"]) == 3


# ── TEST 5: PRE-QUERY PERMISSION GATING (SECURITY) ─────────────────────────────
def test_pre_query_permission_gating_blocks_sales_role(sales_permissions: ChatPermissions) -> None:
    """
    Ensures that calling recommend_batch_configuration without Batch and Production
    permissions raises PermissionDeniedError BEFORE executing any database queries.
    """
    with patch.object(facility_api_service, "_get_business_container") as mock_db:
        with pytest.raises(PermissionDeniedError) as exc_info:
            recommend_batch_configuration(
                product_id="Aspirin 500mg",
                permissions=sales_permissions,
            )

        assert "Sales" in str(exc_info.value) or "not accessible" in str(exc_info.value) or "Access denied" in str(exc_info.value)
        # Verify no database interaction took place
        mock_db.assert_not_called()


# ── TEST 6: AUDIT TRAIL LOGGING PER SRS SECTION 13 ────────────────────────────
@pytest.mark.asyncio
async def test_recommendation_logged_to_audit_trail(qa_all_permissions: ChatPermissions) -> None:
    """
    SRS Section 13 controlled action: AI batch recommendations must be logged to the audit trail
    with timestamp, user_id, role, retrieved records, model/config version, and advisory note.
    """
    mock_batch = {
        "id": "B-99",
        "product": "Sterile Saline",
        "formulation_version": "v1.0",
        "room": "Cleanroom-A",
        "operators": ["OP-10"],
        "equipment": ["EQ-10"],
        "materials": ["RM-10"],
        "yield_percentage": 99.1,
    }

    def mock_get_container(name: str) -> MagicMock:
        c = MagicMock()
        if name == "batches":
            c.query_items.return_value = iter([mock_batch])
        elif name == "materials":
            c.query_items.return_value = iter([{"id": "RM-10", "quality_status": "RELEASED"}])
        elif name == "equipment":
            c.query_items.return_value = iter([{"id": "EQ-10", "pm_status": "QUALIFIED"}])
        elif name == "training":
            c.query_items.return_value = iter([{"operator_id": "OP-10", "status": "CURRENT"}])
        return c

    with patch.object(facility_api_service, "_get_business_container", side_effect=mock_get_container):
        with patch.object(audit_service, "log_audit_entry", new_callable=AsyncMock) as mock_audit:
            recommend_batch_configuration(
                product_id="Sterile Saline",
                target_quantity=2000.0,
                permissions=qa_all_permissions,
                user_id="qa_super_01",
                role="QA Supervisor",
            )
            # Give the asyncio task a tick to run
            await asyncio.sleep(0.01)

            mock_audit.assert_called_once()
            call_kwargs = mock_audit.call_args[1]
            assert call_kwargs["user_id"] == "qa_super_01"
            assert call_kwargs["role"] == "QA Supervisor"
            assert call_kwargs["entity_type"] == "batch_recommendation"
            assert ADVISORY_STATUS_LABEL in call_kwargs["final_response"]


# ── TEST 7: AI SERVICE INTENT & OFFLINE ADVISORY FORMATTING ───────────────────
@pytest.mark.asyncio
async def test_ai_service_offline_batch_recommendation_flow(qa_all_permissions: ChatPermissions) -> None:
    """
    Tests end-to-end intent recognition in ai_service.py:
    'Recommend batch configuration for Aspirin 500mg' triggers recommend_batch_configuration,
    and returns a response with the advisory draft header, reasons, and approvals.
    """
    mock_batch = {
        "id": "B-55",
        "product": "Aspirin 500mg",
        "formulation_version": "v2.1",
        "room": "Cleanroom-1",
        "operators": ["OP-05"],
        "equipment": ["EQ-05"],
        "materials": ["RM-05"],
        "yield_percentage": 98.9,
    }

    def mock_get_container(name: str) -> MagicMock:
        c = MagicMock()
        if name == "batches":
            c.query_items.return_value = iter([mock_batch])
        elif name == "materials":
            c.query_items.return_value = iter([{"id": "RM-05", "quality_status": "RELEASED"}])
        elif name == "equipment":
            c.query_items.return_value = iter([{"id": "EQ-05", "pm_status": "QUALIFIED"}])
        elif name == "training":
            c.query_items.return_value = iter([{"operator_id": "OP-05", "status": "CURRENT"}])
        return c

    with patch.object(facility_api_service, "_get_business_container", side_effect=mock_get_container):
        with patch.object(audit_service, "log_audit_entry", new_callable=AsyncMock):
            res = await generate_response(
                "Recommend batch configuration for Aspirin 500mg with quantity 1000",
                permissions=qa_all_permissions,
                username="planner_user",
            )

            text = res.text
            # 1. Must carry explicit advisory label
            assert ADVISORY_STATUS_LABEL in text
            # 2. Must state advisory notice
            assert "advisory-only" in text.lower()
            # 3. Must specify recommendation details
            assert "Recommended Configuration" in text
            assert "Risk Indicator" in text
            assert "Required Human Approvals" in text

