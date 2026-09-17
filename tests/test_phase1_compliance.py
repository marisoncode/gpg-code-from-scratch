"""Integration and compliance tests for CPG AI Phase 1.

Verifies:
1. Production startup fails without required env vars (passes in dev).
2. A user without permission never triggers a downstream HTTP call.
3. Downstream HTTP calls forward Bearer token in headers.
4. Missing data produces explicit 'unavailable' response, not fabrication.
5. Every chat request produces an audit log entry with function(s) called.
6. The LLM cannot bypass predefined functions to run arbitrary queries.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient

from app.agent.prompts import SYSTEM_PROMPT
from app.core.config import settings
from app.main import app, create_app, get_cors_origins, validate_production_environment
from app.services import audit_service, facility_api_service
from app.services.ai_service import GenerationResult, generate_response
from app.services.audit_service import (
    get_recent_audit_entries,
    log_audit_entry,
)
from app.services.capabilities import ChatPermissions, ModulePermission
from app.services.facility_api_service import (
    FacilitySecurityError,
    execute_predefined_tool,
    get_batch_by_id,
    get_equipment_status,
    get_operator_training_status,
)
from app.services.permissions_service import PermissionDeniedError


# ── TEST 1: PRODUCTION STARTUP CHECK & CORS ───────────────────────────────────


def test_production_startup_fails_without_required_env_vars() -> None:
    """Production mode must refuse to start if any required variable is empty."""
    from app.core.config import Settings

    required_keys = [
        "PERMISSIONS_API",
        "FACILITY_API",
        "PRODUCTION_API",
        "COMPLIANCE_API",
        "ORDER_API",
        "NOTIFICATION_HUB_API",
        "FRONTEND_ORIGIN",
        "MONGO_CONNECTION_STRING",
        "MONGO_DATABASE_NAME",
    ]

    base_env = {
        "ENVIRONMENT": "production",
        "PERMISSIONS_API": "https://perm.api/api",
        "FACILITY_API": "https://facility.api/api",
        "PRODUCTION_API": "https://prod.api/api",
        "COMPLIANCE_API": "https://comp.api/api",
        "ORDER_API": "https://order.api/api",
        "NOTIFICATION_HUB_API": "https://notif.api/api",
        "FRONTEND_ORIGIN": "https://app.cpguardian.com",
        "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
        "MONGO_DATABASE_NAME": "cpg_ai_prod",
    }

    # All present -> should pass without error using a freshly constructed Settings
    with patch.dict(os.environ, base_env, clear=True):
        full_settings = Settings(_env_file=None)
        validate_production_environment(full_settings)

    # Each missing variable from environment must result in empty setting and raise RuntimeError
    for key in required_keys:
        incomplete_env = dict(base_env)
        incomplete_env.pop(key, None)
        with patch.dict(os.environ, incomplete_env, clear=True):
            fresh_settings = Settings(_env_file=None)
            # Ensure Settings() actually produces an empty string when the env var is not set
            assert getattr(fresh_settings, key.lower()) == "", f"Expected {key} to default to empty string"
            with pytest.raises(RuntimeError) as exc_info:
                validate_production_environment(fresh_settings)
            assert key in str(exc_info.value), f"Expected error to name missing variable {key}"


def test_production_fails_when_all_env_vars_missing() -> None:
    """A production deployment with no env vars and no .env file must fail immediately at validation."""
    from app.core.config import Settings

    with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
        fresh_settings = Settings(_env_file=None)
        # All 9 required fields must be empty by default (no hardcoded fallback URLs/hosts)
        assert fresh_settings.permissions_api == ""
        assert fresh_settings.facility_api == ""
        assert fresh_settings.production_api == ""
        assert fresh_settings.compliance_api == ""
        assert fresh_settings.order_api == ""
        assert fresh_settings.notification_hub_api == ""
        assert fresh_settings.frontend_origin == ""
        assert fresh_settings.mongo_connection_string == ""
        assert fresh_settings.mongo_database_name == ""

        with pytest.raises(RuntimeError) as exc_info:
            validate_production_environment(fresh_settings)
        assert "Missing required environment variable" in str(exc_info.value)


def test_production_app_instantiation_fails_without_frontend_origin() -> None:
    """Instantiating the app in production with FRONTEND_ORIGIN unset must fail loudly at CORS setup."""
    with patch.dict(os.environ, {"ENVIRONMENT": "production", "FRONTEND_ORIGIN": ""}, clear=True), \
         patch.object(settings, "environment", "production"), \
         patch.object(settings, "frontend_origin", ""):
        with pytest.raises(RuntimeError) as exc_info:
            create_app()
        assert "FRONTEND_ORIGIN" in str(exc_info.value)
        assert "wildcard fallback is prohibited" in str(exc_info.value).lower() or "prohibited" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_production_app_startup_succeeds_with_frontend_origin() -> None:
    """App instantiation and lifespan startup in production succeeds when FRONTEND_ORIGIN and required vars are set."""
    prod_env = {
        "ENVIRONMENT": "production",
        "PERMISSIONS_API": "https://perm.api/api",
        "FACILITY_API": "https://facility.api/api",
        "PRODUCTION_API": "https://prod.api/api",
        "COMPLIANCE_API": "https://comp.api/api",
        "ORDER_API": "https://order.api/api",
        "NOTIFICATION_HUB_API": "https://notif.api/api",
        "FRONTEND_ORIGIN": "https://app.cpguardian.com",
        "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
        "MONGO_DATABASE_NAME": "cpg_ai_prod",
    }
    with patch.dict(os.environ, prod_env, clear=True), \
         patch.object(settings, "environment", "production"), \
         patch.object(settings, "permissions_api", "https://perm.api/api"), \
         patch.object(settings, "facility_api", "https://facility.api/api"), \
         patch.object(settings, "production_api", "https://prod.api/api"), \
         patch.object(settings, "compliance_api", "https://comp.api/api"), \
         patch.object(settings, "order_api", "https://order.api/api"), \
         patch.object(settings, "notification_hub_api", "https://notif.api/api"), \
         patch.object(settings, "frontend_origin", "https://app.cpguardian.com"), \
         patch.object(settings, "mongo_connection_string", "mongodb://localhost:27017"), \
         patch.object(settings, "mongo_database_name", "cpg_ai_prod"), \
         patch("app.main.init_mongo", new_callable=AsyncMock), \
         patch("app.main.close_mongo", new_callable=AsyncMock), \
         patch("app.main.init_http_client", new_callable=AsyncMock), \
         patch("app.main.close_http_client", new_callable=AsyncMock), \
         patch("app.services.facility_api_service.init_http_client", new_callable=AsyncMock), \
         patch("app.services.facility_api_service.close_http_client", new_callable=AsyncMock):
        test_app = create_app()
        from app.main import lifespan
        async with lifespan(test_app):
            client = TestClient(test_app)
            resp = client.get("/health", headers={"Origin": "https://app.cpguardian.com"})
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok", "service": "cpg-ai-backend"}
            assert resp.headers.get("access-control-allow-origin") == "https://app.cpguardian.com"


def test_development_startup_skips_validation() -> None:
    """Development mode must start cleanly even when required production keys are empty."""
    dev_env = {
        "ENVIRONMENT": "development",
        "PERMISSIONS_API": "",
        "FACILITY_API": "",
        "PRODUCTION_API": "",
        "COMPLIANCE_API": "",
        "ORDER_API": "",
        "NOTIFICATION_HUB_API": "",
        "FRONTEND_ORIGIN": "",
    }
    with patch.dict(os.environ, dev_env, clear=True):
        with patch.object(settings, "environment", "development"), \
             patch.object(settings, "frontend_origin", ""):
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


def test_development_cors_allows_localhost_and_wildcard_behavior() -> None:
    """Development mode still allows localhost origins and does not require FRONTEND_ORIGIN."""
    dev_env = {
        "ENVIRONMENT": "development",
        "FRONTEND_ORIGIN": "",
    }
    with patch.dict(os.environ, dev_env, clear=True), \
         patch.object(settings, "environment", "development"), \
         patch.object(settings, "frontend_origin", ""):
        test_app = create_app()
        client = TestClient(test_app)
        resp = client.get("/health", headers={"Origin": "http://localhost:4200"})
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:4200"


def test_openai_compatibility_routes_disabled_in_production() -> None:
    """OpenAI compatibility routes (/v1/chat/completions, /v1/models, /models) are not registered in production."""
    prod_env = {
        "ENVIRONMENT": "production",
        "FRONTEND_ORIGIN": "https://app.cpguardian.com",
    }
    with patch.dict(os.environ, prod_env, clear=True), \
         patch.object(settings, "environment", "production"), \
         patch.object(settings, "frontend_origin", "https://app.cpguardian.com"):
        test_app = create_app()
        client = TestClient(test_app)

        # /health remains accessible without authentication
        assert client.get("/health").status_code == 200

        # OpenAI compatibility routes must be 404 (not registered) in production
        assert client.get("/v1/models").status_code == 404
        assert client.get("/models").status_code == 404
        assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 404


@pytest.mark.asyncio
async def test_openai_compatibility_routes_active_in_development() -> None:
    """OpenAI compatibility routes (/v1/chat/completions, /v1/models, /models) are available in development."""
    dev_env = {
        "ENVIRONMENT": "development",
        "FRONTEND_ORIGIN": "",
    }
    with patch.dict(os.environ, dev_env, clear=True), \
         patch.object(settings, "environment", "development"), \
         patch.object(settings, "frontend_origin", ""), \
         patch("app.services.ai_service.generate_response", return_value="assistant reply"):
        test_app = create_app()
        client = TestClient(test_app)

        # GET /v1/models & /models
        res_v1 = client.get("/v1/models")
        assert res_v1.status_code == 200
        assert res_v1.json()["object"] == "list"

        res_models = client.get("/models")
        assert res_models.status_code == 200
        assert res_models.json()["object"] == "list"

        # POST /v1/chat/completions
        res_chat = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})
        assert res_chat.status_code == 200
        data = res_chat.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "assistant reply"


# ── TEST 2: PERMISSION GATE PREVENTS DOWNSTREAM READ ──────────────────────────


@pytest.mark.asyncio
async def test_permission_gate_prevents_downstream_read() -> None:
    """A user lacking permission must be rejected BEFORE any downstream HTTP call is issued."""
    mock_client = MagicMock()
    facility_api_service.set_http_client(mock_client)

    try:
        # 1. Sales lens trying to read batch data
        sales_perms = ChatPermissions(Dashboard_assign="Sales")
        with pytest.raises(PermissionDeniedError):
            await get_batch_by_id("B-1021", permissions=sales_perms)
        # Verify downstream HTTP client was NEVER called
        mock_client.get.assert_not_called()

        # 2. Production user without Compliance view trying to read operator training records
        prod_perms = ChatPermissions(
            Dashboard_assign="Production",
            modules=[ModulePermission(Module_name="Batch Record", View=True)],
        )
        with pytest.raises(PermissionDeniedError):
            await get_operator_training_status("OP-017", permissions=prod_perms)
        # Verify downstream HTTP client was NEVER called
        mock_client.get.assert_not_called()
    finally:
        facility_api_service.set_http_client(None)


# ── TEST 3: BEARER TOKEN FORWARDING & DOWNSTREAM HEADERS ──────────────────────


@pytest.mark.asyncio
async def test_downstream_bearer_token_forwarding() -> None:
    """Downstream calls forward the Authorization Bearer token."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "B-1021", "status": "IN_PROGRESS"}
    mock_client.get.return_value = mock_response

    facility_api_service.set_http_client(mock_client)
    facility_api_service.set_current_user(user_id="test-operator-01", username="Test Operator")
    try:
        qa_perms = ChatPermissions(
            Dashboard_assign="All",
            modules=[ModulePermission(Module_name="Batch Record", View=True)],
        )
        res = await get_batch_by_id("B-1021", permissions=qa_perms, token="forwarded-test-token")
        assert res["status"] == "found"
        assert mock_client.get.called
        headers = mock_client.get.call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer forwarded-test-token"
        assert headers.get("userid") == "test-operator-01"
    finally:
        facility_api_service.set_http_client(None)
        facility_api_service.set_current_user(user_id="", username="")


# ── TEST 4: MISSING DATA PRODUCES EXPLICIT 'UNAVAILABLE' (NO FABRICATION) ─────


@pytest.mark.asyncio
async def test_missing_data_produces_explicit_unavailable_response() -> None:
    """Missing data must be stated explicitly as unavailable, never fabricated."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_client.get.return_value = mock_response
    facility_api_service.set_http_client(mock_client)
    facility_api_service.set_current_user(user_id="test-operator-01", username="Test Operator")

    qa_perms = ChatPermissions(
        Dashboard_assign="All",
        modules=[
            ModulePermission(Module_name="Batch Record", View=True),
            ModulePermission(Module_name="Compliance", View=True),
        ],
    )

    try:
        result = await get_batch_by_id("B-99999", permissions=qa_perms)
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
    finally:
        facility_api_service.set_http_client(None)
        facility_api_service.set_current_user(user_id="", username="")


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
async def test_chat_endpoint_produces_audit_log() -> None:
    """Sending a request to the chat endpoint logs to the audit trail."""
    token = jwt.encode(
        {"userId": "user-gmp-101", "name": "Jane QA", "role": "QA"},
        "dummy-secret-key-at-least-32-chars-long!",
        algorithm="HS256",
    )

    client = TestClient(app)
    with patch("app.api.chat.resolve_permissions", return_value=ChatPermissions(
        Dashboard_assign="All",
        modules=[ModulePermission(Module_name="Batch Record", View=True)],
    )):
        response = client.post(
            "/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "message": "Investigate Batch B-1021",
            },
        )

    assert response.status_code == 200
    entries = get_recent_audit_entries(limit=10)
    assert len(entries) > 0
    latest = entries[0]
    assert latest["user_id"] == "user-gmp-101"
    assert "B-1021" in latest["raw_query"]


# ── TEST 6: LLM CANNOT BYPASS PREDEFINED FUNCTIONS ────────────────────────────


@pytest.mark.asyncio
async def test_llm_cannot_bypass_predefined_functions() -> None:
    """FastAPI rejects arbitrary queries and enforces predefined functions only."""
    # Attempting to call an arbitrary function name
    with pytest.raises(ValueError) as exc_info:
        await execute_predefined_tool(
            "run_raw_query",
            {"query": "SELECT * FROM cpg_compliance.batches"},
            permissions=None,
        )
    assert "Unauthorized function 'run_raw_query'" in str(exc_info.value)
    assert "Arbitrary database queries are strictly prohibited" in str(exc_info.value)

    # Attempting SQL injection / DROP TABLE
    with pytest.raises(ValueError) as exc_info2:
        await execute_predefined_tool("DROP TABLE batches", {}, permissions=None)
    assert "Unauthorized function" in str(exc_info2.value)


@pytest.mark.asyncio
async def test_raw_query_in_user_prompt_is_rejected() -> None:
    """A user attempting to pass raw SQL statements in chat is refused immediately."""
    result = await generate_response("SELECT * FROM batches WHERE 1=1")
    assert "strictly prohibited" in str(result).lower() or "cannot construct or execute raw database queries" in str(result).lower()


@pytest.mark.asyncio
async def test_conversational_greeting_does_not_invoke_tools() -> None:
    """Sending a simple greeting like 'hi' or 'hello' responds conversationally without tool calls."""
    result = await generate_response("hi", username="Alex")
    assert result.functions_called == []
    assert str(result)
