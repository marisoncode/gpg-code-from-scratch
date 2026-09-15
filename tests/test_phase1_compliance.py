"""Integration and compliance tests for CPG AI Phase 1.

Verifies:
1. Production startup fails without required env vars (passes in dev).
2. A user without permission never triggers a Cosmos read.
3. Write-scoped credential is never used against business-data containers.
4. Missing data produces explicit 'unavailable' response, not fabrication.
5. Every chat request produces an audit log entry with function(s) called.
6. The LLM cannot bypass predefined functions to run arbitrary queries.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from azure.cosmos import CosmosClient
from fastapi.testclient import TestClient

from app.agent.prompts import SYSTEM_PROMPT
from app.core.config import Settings, settings
from app.main import app, get_cors_origins, validate_production_environment
from app.services import audit_service, facility_api_service
from app.services.ai_service import GenerationResult, generate_response
from app.services.audit_service import (
    AuditSecurityError,
    get_recent_audit_entries,
    get_write_container,
    log_audit_entry,
)
from app.services.capabilities import ChatPermissions, ModulePermission
from app.services.facility_api_service import (
    FacilitySecurityError,
    execute_predefined_tool,
    get_batch_by_id,
    get_equipment_status,
    get_operator_training_status,
    get_readonly_cosmos_client,
)
from app.services.permissions_service import PermissionDeniedError


# ── TEST 1: PRODUCTION STARTUP CHECK & CORS ───────────────────────────────────


def test_production_startup_fails_without_required_env_vars() -> None:
    """Production mode must refuse to start if any required variable is empty."""
    required_keys = [
        "JWT_SECRET",
        "PERMISSIONS_API",
        "COSMOS_DB_READONLY_KEY",
        "COSMOS_DB_WRITE_KEY",
        "MONGODB_URI",
    ]

    base_env = {
        "ENVIRONMENT": "production",
        "JWT_SECRET": "sec-123",
        "PERMISSIONS_API": "https://perm.api/api",
        "COSMOS_DB_READONLY_KEY": "ro-key",
        "COSMOS_DB_WRITE_KEY": "wr-key",
        "MONGODB_URI": "mongodb://localhost:27017",
    }

    # All present -> should pass without error
    with patch.dict(os.environ, base_env, clear=True):
        with patch.object(settings, "environment", "production"), \
             patch.object(settings, "jwt_secret", "sec-123"), \
             patch.object(settings, "permissions_api", "https://perm.api/api"), \
             patch.object(settings, "cosmos_db_readonly_key", "ro-key"), \
             patch.object(settings, "cosmos_db_write_key", "wr-key"):
            validate_production_environment()

    # Each missing variable must raise RuntimeError naming that variable
    for key in required_keys:
        incomplete_env = dict(base_env)
        incomplete_env[key] = ""
        with patch.dict(os.environ, incomplete_env, clear=True):
            with patch.object(settings, "environment", "production"):
                patcher = None
                if hasattr(settings, key.lower()):
                    patcher = patch.object(settings, key.lower(), "")
                    patcher.start()
                try:
                    with pytest.raises(RuntimeError) as exc_info:
                        validate_production_environment()
                    assert key in str(exc_info.value), f"Expected error to name missing variable {key}"
                finally:
                    if patcher:
                        patcher.stop()


def test_development_startup_skips_validation() -> None:
    """Development mode must start cleanly even when required production keys are empty."""
    dev_env = {
        "ENVIRONMENT": "development",
        "JWT_SECRET": "",
        "PERMISSIONS_API": "",
        "COSMOS_DB_READONLY_KEY": "",
        "COSMOS_DB_WRITE_KEY": "",
        "MONGODB_URI": "",
    }
    with patch.dict(os.environ, dev_env, clear=True):
        with patch.object(settings, "environment", "development"):
            # Must not raise
            validate_production_environment()


def test_cors_locking_by_environment() -> None:
    """CORS is locked to FRONTEND_ORIGIN in production, but allows localhost in development."""
    with patch.object(settings, "environment", "production"), \
         patch.object(settings, "frontend_origin", "https://app.cpguardian.com"):
        origins = get_cors_origins()
        assert origins == ["https://app.cpguardian.com"]

    with patch.object(settings, "environment", "development"), \
         patch.object(settings, "frontend_origin", "http://localhost:4200"):
        dev_origins = get_cors_origins()
        assert "http://localhost:4200" in dev_origins
        assert "http://127.0.0.1:4200" in dev_origins


# ── TEST 2: PERMISSION GATE PREVENTS COSMOS READ ──────────────────────────────


def test_permission_gate_prevents_cosmos_read() -> None:
    """A user lacking permission must be rejected BEFORE any Cosmos read is issued."""
    mock_container = MagicMock()
    mock_container.query_items.return_value = []

    # 1. Sales lens trying to read batch data
    sales_perms = ChatPermissions(Dashboard_assign="Sales")
    with patch("app.services.facility_api_service._get_business_container", return_value=mock_container):
        with pytest.raises(PermissionDeniedError):
            get_batch_by_id("B-1021", permissions=sales_perms)
        # Verify container was NEVER queried
        mock_container.query_items.assert_not_called()

    # 2. Production user without Compliance view trying to read operator training records
    prod_perms = ChatPermissions(
        Dashboard_assign="Production",
        modules=[ModulePermission(Module_name="Batch Record", View=True)],
    )
    with patch("app.services.facility_api_service._get_business_container", return_value=mock_container):
        with pytest.raises(PermissionDeniedError):
            get_operator_training_status("OP-017", permissions=prod_perms)
        # Verify container was NEVER queried
        mock_container.query_items.assert_not_called()


# ── TEST 3: WRITE-SCOPED CREDENTIAL ISOLATION ──────────────────────────────────


def test_write_credential_never_used_against_business_containers() -> None:
    """Write client and write credential must NEVER access business data containers."""
    # Attempting to access business containers with write container getter must raise
    business_containers = ["batches", "operators", "equipment", "training", "materials", "deviations", "inventory"]
    for container_name in business_containers:
        with pytest.raises(AuditSecurityError) as exc_info:
            get_write_container(container_name)
        assert "Security violation" in str(exc_info.value)
        assert container_name in str(exc_info.value)

    # Read-only service attempting to query audit container must also be rejected
    with pytest.raises(FacilitySecurityError) as exc_info2:
        facility_api_service._get_business_container("audit_trail")
    assert "forbidden" in str(exc_info2.value)


def test_readonly_service_rejects_write_key() -> None:
    """If facility service is configured with write key, it must refuse to initialize."""
    with patch.object(settings, "cosmos_db_readonly_key", "same_key"), \
         patch.object(settings, "cosmos_db_write_key", "same_key"), \
         patch.object(settings, "cosmos_db_endpoint", "https://mock.cosmos.azure.com:443/"):
        facility_api_service.set_cosmos_readonly_client(None)
        with pytest.raises(FacilitySecurityError):
            get_readonly_cosmos_client()


# ── TEST 4: MISSING DATA PRODUCES EXPLICIT 'UNAVAILABLE' (NO FABRICATION) ─────


@pytest.mark.asyncio
async def test_missing_data_produces_explicit_unavailable_response() -> None:
    """Missing data must be stated explicitly as unavailable, never fabricated."""
    mock_container = MagicMock()
    # Cosmos returns empty list for nonexistent batch
    mock_container.query_items.return_value = []

    qa_perms = ChatPermissions(
        Dashboard_assign="All",
        modules=[
            ModulePermission(Module_name="Batch Record", View=True),
            ModulePermission(Module_name="Compliance", View=True),
        ],
    )

    with patch("app.services.facility_api_service._get_business_container", return_value=mock_container):
        result = get_batch_by_id("B-99999", permissions=qa_perms)
        assert result["status"] == "unavailable"
        assert "unavailable or not found" in result["error"]
        assert result["record_id"] == "B-99999"

        # Via AI generator
        ai_res = await generate_response("Investigate Batch B-99999", permissions=qa_perms)
        res_text = str(ai_res)
        assert "unavailable" in res_text.lower()
        # Verify no fabricated success or status
        assert "released" not in res_text.lower()
        assert "passed" not in res_text.lower()


def test_prompt_enforces_no_fabrication_and_citations() -> None:
    """System prompt must enforce citation standards, categorical labels, and no-fabrication."""
    assert "CITATION REQUIREMENT" in SYSTEM_PROMPT
    assert "MISSING / INACCESSIBLE DATA" in SYSTEM_PROMPT
    assert "STRICT NO-FABRICATION" in SYSTEM_PROMPT
    assert "[FACT]" in SYSTEM_PROMPT
    assert "[CALCULATION]" in SYSTEM_PROMPT
    assert "[RECOMMENDATION]" in SYSTEM_PROMPT
    assert "[HYPOTHESIS]" in SYSTEM_PROMPT
    assert "Never invent, hallucinate, or fabricate" in SYSTEM_PROMPT


# ── TEST 5: AUDIT LOG ENTRY ON EVERY REQUEST ──────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_entry_created_with_function_calls() -> None:
    """Every request produces an immutable audit log entry recording query and tool calls."""
    user_id = "user_test_42"
    role = "QA Manager"
    raw_query = "Investigate Batch B-1021 with Bearer eyJhbGciOiJIUzI1NiJ9.secret"

    entry = await log_audit_entry(
        user_id=user_id,
        role=role,
        raw_query=raw_query,
        functions_called=[{"name": "get_batch_by_id", "parameters": {"batch_id": "B-1021"}}],
        retrieved_record_ids=["B-1021"],
        model_version="gpt-4o-mini",
        final_response="[FACT] Batch B-1021 [Source: Batch Record B-1021]: Status is IN_PROGRESS.",
        permission_denied=False,
    )

    assert entry["user_id"] == user_id
    assert entry["role"] == role
    assert entry["functions_called"][0]["name"] == "get_batch_by_id"
    assert entry["retrieved_record_ids"] == ["B-1021"]
    assert entry["permission_denied"] is False
    # Verify raw token is sanitized and never stored in audit log
    assert "eyJhbGciOiJIUzI1NiJ9" not in entry["raw_query"]
    assert "[REDACTED_JWT]" in entry["raw_query"]


@pytest.mark.asyncio
async def test_chat_endpoint_produces_audit_log(monkeypatch) -> None:
    """Sending a request to the chat endpoint logs to the audit trail."""
    import jwt

    secret = settings.jwt_secret or "dev-jwt-secret-testing-only-12345"
    token = jwt.encode(
        {"userId": "user-gmp-101", "name": "Jane QA", "role": "QA"},
        secret,
        algorithm="HS256",
    )

    client = TestClient(app)
    response = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "message": "Investigate Batch B-1021",
            "permissions": {
                "Dashboard_assign": "All",
                "modules": [{"Module_name": "Batch Record", "View": True}],
            },
        },
    )

    assert response.status_code == 200
    entries = get_recent_audit_entries(limit=10)
    assert len(entries) > 0
    latest = entries[0]
    assert latest["user_id"] == "user-gmp-101"
    assert "B-1021" in latest["raw_query"]


# ── TEST 6: LLM CANNOT BYPASS PREDEFINED FUNCTIONS ────────────────────────────


def test_llm_cannot_bypass_predefined_functions() -> None:
    """FastAPI rejects arbitrary queries and enforces predefined functions only."""
    # Attempting to call an arbitrary function name
    with pytest.raises(ValueError) as exc_info:
        execute_predefined_tool(
            "run_raw_query",
            {"query": "SELECT * FROM cpg_compliance.batches"},
            permissions=None,
        )
    assert "Unauthorized function 'run_raw_query'" in str(exc_info.value)
    assert "Arbitrary database queries are strictly prohibited" in str(exc_info.value)

    # Attempting SQL injection / DROP TABLE
    with pytest.raises(ValueError) as exc_info2:
        execute_predefined_tool("DROP TABLE batches", {}, permissions=None)
    assert "Unauthorized function" in str(exc_info2.value)


@pytest.mark.asyncio
async def test_raw_query_in_user_prompt_is_rejected() -> None:
    """A user attempting to pass raw SQL statements in chat is refused immediately."""
    result = await generate_response("SELECT * FROM batches WHERE 1=1")
    assert "strictly prohibited" in str(result).lower() or "cannot construct or execute raw database queries" in str(result).lower()


# ── TEST 7: REAL COSMOS READ-ONLY CLIENT CONFIGURATION ─────────────────────────


def test_real_readonly_cosmos_client_instantiation() -> None:
    """CosmosClient instantiates with read-only key and endpoint without error."""
    from azure.cosmos._global_endpoint_manager import _GlobalEndpointManager

    with patch.object(settings, "cosmos_db_endpoint", "https://test-cpg.documents.azure.com:443/"), \
         patch.object(settings, "cosmos_db_readonly_key", "dGVzdC1yZWFkb25seS1rZXk="), \
         patch.object(settings, "cosmos_db_write_key", "dGVzdC13cml0ZS1rZXk="), \
         patch.object(_GlobalEndpointManager, "_GetDatabaseAccount"):
        facility_api_service.set_cosmos_readonly_client(None)
        client = get_readonly_cosmos_client()
        assert isinstance(client, CosmosClient)

