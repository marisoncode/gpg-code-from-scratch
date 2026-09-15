"""Integration and compliance tests for CPG AI Phase 3: Batch Similarity, Risk Scoring, and Trend Detection.

Verifies:
1. Batch Similarity (SRS FR-006): Similarity results include percentage, disposition, and top
   explaining factors (e.g. '96% similarity, passed'), not just a bare number.
2. Configurable Risk Scoring (SRS Section 19): Risk score changes when weight configuration changes,
   proving weights are configurable/approved and not hardcoded.
3. Trend & Anomaly Detection (SRS Section 13): Trend output never uses causal language for
   correlational findings; strictly enforces non-causal language and mandatory correlation disclaimer.
4. Trend output completeness: Must include period, sample size, method/threshold, evidence,
   and explicit uncertainty/confidence indicator.
5. Audit trail records risk-weight configuration version for audit reproducibility.
6. Pre-query permission gating enforced across all Phase 3 analytics functions.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import audit_service, facility_api_service
from app.services.ai_service import generate_response
from app.services.analytics_service import (
    calculate_risk_score,
    detect_trends,
    find_similar_batches,
    load_risk_weights,
)
from app.services.audit_service import get_recent_audit_entries, log_audit_entry
from app.services.capabilities import ChatPermissions, ModulePermission
from app.services.permissions_service import PermissionDeniedError


@pytest.fixture
def qa_all_permissions() -> ChatPermissions:
    """QA user with full view access."""
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
def sales_permissions() -> ChatPermissions:
    """Sales lens (blocked)."""
    return ChatPermissions(
        Dashboard_assign="Sales",
        modules=[],
    )


# ── TEST 1: BATCH SIMILARITY INCLUDES TOP EXPLAINING FACTORS (SRS FR-006) ─────


def test_batch_similarity_includes_top_explaining_factors_and_disposition(qa_all_permissions: ChatPermissions) -> None:
    """
    SRS FR-006:
    Returns similarity percentage, disposition, and the top factors driving similarity.
    Must not be just a bare number. Example format: '96% similarity, passed'.
    """
    target_batch = {
        "id": "B-1021",
        "batch_number": "B-1021",
        "product": "ProductAlpha",
        "formulation_version": "v2.0",
        "quantity": 1000,
        "materials": ["RM-101", "RM-102"],
        "operators": ["OP-01", "OP-02"],
        "equipment": ["EQ-101", "EQ-102"],
        "room": "Cleanroom-1",
        "status": "IN_PROGRESS",
    }

    candidate_batches = [
        {
            "id": "B-1020",
            "batch_number": "B-1020",
            "product": "ProductAlpha",
            "formulation_version": "v2.0",
            "quantity": 995,
            "materials": ["RM-101", "RM-102"],
            "operators": ["OP-01"],
            "equipment": ["EQ-101", "EQ-102"],
            "room": "Cleanroom-1",
            "status": "RELEASED",
        },
        {
            "id": "B-1010",
            "batch_number": "B-1010",
            "product": "ProductAlpha",
            "formulation_version": "v1.0",
            "quantity": 500,
            "materials": ["RM-101"],
            "operators": ["OP-09"],
            "equipment": ["EQ-101"],
            "room": "Cleanroom-2",
            "status": "REJECTED",
        },
    ]

    mock_container = MagicMock()
    # First call: target batch fetch; Second call: candidate batches fetch
    mock_container.query_items.side_effect = [
        [target_batch],
        candidate_batches,
    ]

    with patch("app.services.facility_api_service._get_business_container", return_value=mock_container):
        res = find_similar_batches("B-1021", top_n=5, permissions=qa_all_permissions)

        assert res["status"] == "found"
        assert res["target_batch_id"] == "B-1021"
        assert len(res["similar_batches"]) == 2

        top_match = res["similar_batches"][0]
        assert top_match["batch_id"] == "B-1020"
        assert top_match["similarity_percentage"] >= 80
        assert top_match["disposition"] == "passed"
        # Assert format matches SRS FR-006: "96% similarity, passed" style
        assert re.match(r"\d+%\s+similarity,\s+passed", top_match["summary"])

        # Assert top explaining factors are present and descriptive
        factors = top_match["top_factors"]
        assert isinstance(factors, list)
        assert len(factors) >= 2
        factors_text = " ".join(factors).lower()
        assert "productalpha" in factors_text or "product" in factors_text
        assert "formulation" in factors_text or "equipment" in factors_text or "materials" in factors_text

        # Second match was rejected
        second_match = res["similar_batches"][1]
        assert second_match["disposition"] == "failed"
        assert second_match["similarity_percentage"] < top_match["similarity_percentage"]


# ── TEST 2: RISK SCORE CHANGES WHEN CONFIGURABLE WEIGHTS CHANGE ───────────────


def test_risk_score_changes_with_configurable_weights(qa_all_permissions: ChatPermissions) -> None:
    """
    SRS Section 19: Risk weights must be loaded from a configurable source, not hardcoded.
    Proves that risk score changes dynamically when weight configuration changes.
    """
    batch_config = {
        "id": "B-RISK-1",
        "equipment": ["EQ-OVERDUE-1"],
        "operators": ["OP-EXPIRED-1"],
        "materials": ["RM-DEFECTIVE-1"],
        "deviations": ["DEV-OPEN-1"],
    }

    # Mock dependent record looks
    mock_eq_container = MagicMock()
    mock_eq_container.query_items.return_value = [{"id": "EQ-OVERDUE-1", "pm_status": "OVERDUE"}]

    mock_tr_container = MagicMock()
    mock_tr_container.query_items.return_value = [{"operator_id": "OP-EXPIRED-1", "status": "EXPIRED", "course_name": "GMP-101"}]

    mock_mat_container = MagicMock()
    mock_mat_container.query_items.return_value = [{"id": "RM-DEFECTIVE-1", "quality_status": "REJECTED"}]

    mock_batches_container = MagicMock()
    mock_batches_container.query_items.return_value = [batch_config]

    def container_dispatch(name: str):
        if name == "equipment":
            return mock_eq_container
        if name == "training":
            return mock_tr_container
        if name == "materials":
            return mock_mat_container
        if name == "batches":
            return mock_batches_container
        return MagicMock()

    with patch("app.services.facility_api_service._get_business_container", side_effect=container_dispatch):
        # Configuration A: Baseline weights
        config_a = {
            "config_version": "v1.0-baseline-qa",
            "weights": {
                "operator_unqualified_or_expired": 10,
                "equipment_pm_overdue": 10,
                "material_lot_defective_or_oos": 10,
                "unresolved_deviation": 10,
            },
            "thresholds": {"low_risk_max": 25, "medium_risk_max": 60, "high_risk_min": 61},
        }
        res_a = calculate_risk_score(
            batch_id_or_config=batch_config,
            permissions=qa_all_permissions,
            weight_config=config_a,
        )

        # Configuration B: High-severity weights
        config_b = {
            "config_version": "v2.0-high-severity-qa",
            "weights": {
                "operator_unqualified_or_expired": 25,
                "equipment_pm_overdue": 25,
                "material_lot_defective_or_oos": 25,
                "unresolved_deviation": 25,
            },
            "thresholds": {"low_risk_max": 25, "medium_risk_max": 60, "high_risk_min": 61},
        }
        res_b = calculate_risk_score(
            batch_id_or_config=batch_config,
            permissions=qa_all_permissions,
            weight_config=config_b,
        )

        # Assert scores reflect weights and are strictly NOT hardcoded
        assert res_a["risk_score"] == 40  # 10 + 10 + 10 + 10
        assert res_a["risk_level"] == "MEDIUM"
        assert res_a["config_version"] == "v1.0-baseline-qa"

        assert res_b["risk_score"] == 100  # 25 + 25 + 25 + 25
        assert res_b["risk_level"] == "HIGH"
        assert res_b["config_version"] == "v2.0-high-severity-qa"

        assert res_a["risk_score"] != res_b["risk_score"]


# ── TEST 3: CORRELATION VS CAUSATION STRICT ENFORCEMENT (SRS SECTION 13) ──────


def test_trend_output_strictly_bans_causal_language(qa_all_permissions: ChatPermissions) -> None:
    """
    SRS Section 13: Correlation must NEVER be presented as confirmed causation.
    Banned terms: 'caused by', 'the root cause of', 'responsible for causing', 'led to'.
    Required: explicit correlation disclaimer and correlational terminology.
    """
    mock_batch_container = MagicMock()
    mock_batch_container.query_items.return_value = [
        {"id": f"B-YIELD-{i}", "product": "DrugZ", "yield_percentage": 98.0 - (i * 0.5), "created_at": f"2026-05-{i:02d}"}
        for i in range(1, 11)
    ]

    with patch("app.services.facility_api_service._get_business_container", return_value=mock_batch_container):
        res = detect_trends(
            product_id="DrugZ",
            metric="yield",
            window="90d",
            permissions=qa_all_permissions,
        )

        assert res["status"] == "found"
        assert res["product_id"] == "DrugZ"
        assert res["metric"] == "yield"

        analysis_text = res["correlation_analysis"].lower()
        disclaimer_text = res["causation_disclaimer"].lower()

        # BANNED causal terms check
        banned_causal_terms = [
            "caused by",
            "the root cause of",
            "responsible for causing",
            "led to the failure",
            "proven cause",
        ]
        for term in banned_causal_terms:
            assert term not in analysis_text, f"Forbidden causal claim found in trend output: '{term}'"

        # REQUIRED correlational terminology check
        assert "correlation" in analysis_text or "associated" in analysis_text or "concurrent" in analysis_text
        assert "correlation notice" in disclaimer_text or "correlation" in disclaimer_text


# ── TEST 4: TREND OUTPUT COMPLETENESS PER SRS SECTION 13 ──────────────────────


def test_trend_output_completeness(qa_all_permissions: ChatPermissions) -> None:
    """
    SRS Section 13 requires:
    - Period (window)
    - Sample size (N records)
    - Method / threshold used
    - Evidence citations
    - Explicit uncertainty / confidence indicator
    """
    mock_batch_container = MagicMock()
    mock_batch_container.query_items.return_value = [
        {"id": "B-DEV-1", "product": "AntibioticA", "deviations": ["DEV-101"]},
        {"id": "B-DEV-2", "product": "AntibioticA", "deviations": ["DEV-102"]},
        {"id": "B-DEV-3", "product": "AntibioticA", "deviations": []},
    ]

    with patch("app.services.facility_api_service._get_business_container", return_value=mock_batch_container):
        res = detect_trends(
            product_id="AntibioticA",
            metric="equipment_deviations",
            window="6m",
            permissions=qa_all_permissions,
        )

        assert res["status"] == "found"
        # 1. Period
        assert res["period"] == "6m"
        # 2. Sample size
        assert res["sample_size"] == 3
        # 3. Method / threshold used
        assert "threshold" in res["method_threshold"].lower()
        # 4. Evidence citations
        assert len(res["evidence"]) >= 1
        assert any("B-DEV" in str(e) for e in res["evidence"])
        # 5. Uncertainty / confidence indicator
        assert "confidence" in res["uncertainty_confidence"].lower() or "uncertainty" in res["uncertainty_confidence"].lower()


# ── TEST 5: PRE-QUERY PERMISSION GATING FOR PHASE 3 ANALYTICS ─────────────────


def test_phase3_permission_gating_blocks_sales_and_unauthorized_users(sales_permissions: ChatPermissions) -> None:
    """Pre-query verification must block Sales lens from similarity, risk scoring, and trend detection."""
    mock_container = MagicMock()

    with patch("app.services.facility_api_service._get_business_container", return_value=mock_container):
        with pytest.raises(PermissionDeniedError):
            find_similar_batches("B-1021", permissions=sales_permissions)
        mock_container.query_items.assert_not_called()

        with pytest.raises(PermissionDeniedError):
            calculate_risk_score("B-1021", permissions=sales_permissions)
        mock_container.query_items.assert_not_called()

        with pytest.raises(PermissionDeniedError):
            detect_trends("ProductX", metric="yield", permissions=sales_permissions)
        mock_container.query_items.assert_not_called()


# ── TEST 6: AUDIT TRAIL CAPTURES RISK-WEIGHT CONFIGURATION VERSION ────────────


@pytest.mark.asyncio
async def test_audit_log_captures_risk_config_version() -> None:
    """
    SRS Section 19: Log every risk score calculation to the audit trail,
    including which risk-weight configuration version was used for reproducibility.
    """
    entry = await log_audit_entry(
        user_id="user-qa-auditor-1",
        role="QA Manager",
        raw_query="Calculate risk score for batch B-1021",
        functions_called=[{"name": "calculate_risk_score", "parameters": {"batch_id": "B-1021"}}],
        retrieved_record_ids=["B-1021", "EQ-102"],
        model_version="gpt-4o-mini",
        final_response="[FACT] Risk Score: 65/100 (HIGH Risk)",
        risk_config_version="v1.0-approved-2026",
    )

    assert entry["risk_config_version"] == "v1.0-approved-2026"
    assert entry["user_id"] == "user-qa-auditor-1"

    recent = get_recent_audit_entries(limit=5)
    matched = [e for e in recent if e.get("id") == entry["id"]]
    assert len(matched) == 1
    assert matched[0]["risk_config_version"] == "v1.0-approved-2026"


# ── TEST 7: CHAT ENDPOINT POPULATES PHASE 3 AUDIT RECORD & RISK VERSION ───────


@pytest.mark.asyncio
async def test_chat_endpoint_similarity_and_risk_audit(qa_all_permissions: ChatPermissions) -> None:
    """POST /chat with risk scoring query must populate risk_config_version in audit trail."""
    import jwt

    secret = settings.jwt_secret or "dev-jwt-secret-testing-only-12345"
    token = jwt.encode(
        {"userId": "user-qa-phase3", "name": "Phase3 QA", "role": "QA"},
        secret,
        algorithm="HS256",
    )

    client = TestClient(app)

    mock_batch_container = MagicMock()
    mock_batch_container.query_items.return_value = [
        {"id": "B-999", "batch_number": "B-999", "product": "VaccineY", "equipment": [], "operators": [], "materials": []}
    ]

    with patch("app.services.facility_api_service._get_business_container", return_value=mock_batch_container):
        payload = {
            "message": "Calculate risk score for batch B-999",
            "permissions": qa_all_permissions.model_dump(),
        }
        resp = client.post(
            "/chat",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        assert resp.status_code == 200

        recent = get_recent_audit_entries(limit=5)
        matched = [e for e in recent if "B-999" in e.get("raw_query", "")]
        assert len(matched) >= 1
        latest = matched[-1]
        assert latest["risk_config_version"] != ""
        assert "v1.0" in latest["risk_config_version"]

