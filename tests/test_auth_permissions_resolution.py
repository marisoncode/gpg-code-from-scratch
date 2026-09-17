"""Tests for Endpoint-Based Identity and Permission Resolution (No Local JWT Secret).

Verifies:
1. Expired token (checked locally via 'exp') is rejected without any network call.
2. Request where PERMISSIONS_API returns 401 is rejected with 401.
3. Request where PERMISSIONS_API returns 403 is rejected with 401.
4. Request where PERMISSIONS_API is unreachable / times out fails closed (denies access).
5. No code path makes an authorization decision using decoded-but-unverified token claims
   (spoofed claims in token are ignored in favor of PERMISSIONS_API authority).
6. Valid PERMISSIONS_API response properly authorizes user actions.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from app.core.auth import AuthenticatedUser, decode_facility_token, require_facility_user
from app.core.config import settings
from app.main import app
from app.services.capabilities import ChatPermissions, ModulePermission
from app.services.permissions_service import (
    PermissionDeniedError,
    parse_permissions_payload,
    resolve_permissions,
    verify_resource_permission,
)


# ── TEST 1: EXPIRED TOKEN REJECTED LOCALLY WITHOUT ANY NETWORK CALL ───────────
def test_expired_token_rejected_locally_without_network_call() -> None:
    """A token whose 'exp' has passed must be rejected with 401 before any network call."""
    past_timestamp = time.time() - 3600  # Expired 1 hour ago
    expired_token = jwt.encode(
        {"sub": "user-expired-01", "name": "Expired User", "exp": past_timestamp},
        "dummy-secret-key-at-least-32-chars-long!",
        algorithm="HS256",
    )

    client = TestClient(app)

    with patch("httpx.AsyncClient.get") as mock_http_get:
        response = client.post(
            "/chat",
            headers={"Authorization": f"Bearer {expired_token}"},
            json={"message": "Investigate Batch B-1021"},
        )

        assert response.status_code == 401
        assert "expired" in response.text.lower()
        # Verify NO network call was made to PERMISSIONS_API or any other service
        mock_http_get.assert_not_called()


# ── TEST 2: PERMISSIONS_API 401/403 REJECTS REQUEST ───────────────────────────
@pytest.mark.asyncio
async def test_permissions_api_401_rejects_request() -> None:
    """When PERMISSIONS_API returns 401, the request is rejected as invalid/expired."""
    future_timestamp = time.time() + 3600
    valid_exp_token = jwt.encode(
        {"sub": "user-401-test", "name": "User 401", "exp": future_timestamp},
        "dummy-secret-key-at-least-32-chars-long!",
        algorithm="HS256",
    )

    client = TestClient(app)

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized by downstream authority"

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        response = client.post(
            "/chat",
            headers={"Authorization": f"Bearer {valid_exp_token}"},
            json={"message": "Investigate Batch B-1021"},
        )

        assert response.status_code == 401
        assert "rejected by permissions authority" in response.text.lower() or "invalid or expired" in response.text.lower()


@pytest.mark.asyncio
async def test_permissions_api_403_rejects_request() -> None:
    """When PERMISSIONS_API returns 403, the request is rejected as unauthorized."""
    future_timestamp = time.time() + 3600
    token = jwt.encode(
        {"sub": "user-403-test", "name": "User 403", "exp": future_timestamp},
        "dummy-secret-key-at-least-32-chars-long!",
        algorithm="HS256",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "Forbidden"

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        with pytest.raises(Exception) as exc_info:
            await resolve_permissions(token)

        assert "401" in str(exc_info.value) or "rejected" in str(exc_info.value).lower()


# ── TEST 3: PERMISSIONS_API UNREACHABLE FAILS CLOSED (DENIES ACCESS) ──────────
@pytest.mark.asyncio
async def test_permissions_api_unreachable_fails_closed() -> None:
    """If PERMISSIONS_API times out or fails (5xx), the service fails closed (no access)."""
    future_timestamp = time.time() + 3600
    token = jwt.encode(
        {"sub": "user-timeout", "name": "Timeout User", "exp": future_timestamp},
        "dummy-secret-key-at-least-32-chars-long!",
        algorithm="HS256",
    )

    # Simulate network timeout
    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectTimeout("Connection timed out")):
        perms = await resolve_permissions(token)

        # Must fail closed: returns 'none' lens and empty modules
        assert perms.Dashboard_assign == "none"
        assert perms.modules == []

        # When checking permissions on any resource, access MUST be denied
        with pytest.raises(PermissionDeniedError) as exc_info:
            verify_resource_permission(perms, "batch")
        assert "Permission denied" in str(exc_info.value)

        with pytest.raises(PermissionDeniedError):
            verify_resource_permission(perms, "equipment")

        with pytest.raises(PermissionDeniedError):
            verify_resource_permission(perms, "compliance")


# ── TEST 4: NO AUTHORIZATION DECISION USES DECODED-BUT-UNVERIFIED CLAIMS ───────
@pytest.mark.asyncio
async def test_no_authorization_decision_uses_decoded_token_claims() -> None:
    """
    CRITICAL SECURITY TEST:
    A user attempts to spoof admin privileges inside the unverified JWT claims:
    `role: "SuperAdmin"`, `Dashboard_assign: "All"`, `modules: [View: True]`.
    However, PERMISSIONS_API returns `Dashboard_assign: "Sales"` (no production access).

    The backend MUST obey the PERMISSIONS_API response and ignore the spoofed token claims!
    """
    future_timestamp = time.time() + 3600
    # Client creates a self-signed token claiming full admin permissions
    spoofed_token = jwt.encode(
        {
            "sub": "attacker-01",
            "name": "Attacker Trying Privilege Escalation",
            "role": "SuperAdmin",
            "Dashboard_assign": "All",
            "modules": [{"Module_name": "Batch Record", "View": True}],
            "exp": future_timestamp,
        },
        "attacker-secret-key-at-least-32-chars-long!",
        algorithm="HS256",
    )

    # Downstream authority PERMISSIONS_API returns the true role: Sales
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "Dashboard_assign": "Sales",
        "modules": [],
    }

    client = TestClient(app)

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        response = client.post(
            "/chat",
            headers={"Authorization": f"Bearer {spoofed_token}"},
            json={
                "message": "Investigate Batch B-1021",
                # Even if client also sends spoofed permissions in body:
                "permissions": {
                    "Dashboard_assign": "All",
                    "modules": [{"Module_name": "Batch Record", "View": True}],
                },
            },
        )

        assert response.status_code == 200
        # The AI response must indicate permission denied because true role is Sales
        response_text = response.json()["message"]
        assert "permission denied" in response_text.lower() or "sales" in response_text.lower() or "not accessible" in response_text.lower()


# ── TEST 5: VALID PERMISSIONS_API RESPONSE AUTHORIZES USER ────────────────────
@pytest.mark.asyncio
async def test_valid_permissions_api_response_authorizes_user() -> None:
    """Valid PERMISSIONS_API 200 response authorizes access to corresponding resources."""
    token = jwt.encode(
        {"sub": "qa-user-01", "name": "Certified QA", "exp": time.time() + 3600},
        "dummy-secret-key-at-least-32-chars-long!",
        algorithm="HS256",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "Dashboard_assign": "Production",
        "modules": [
            {"Module_name": "Batch Record", "View": True},
            {"Module_name": "Inventory", "View": True},
        ],
    }

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        perms = await resolve_permissions(token)
        assert perms.Dashboard_assign == "Production"
        assert len(perms.modules) == 2

        # Verification: Production resource passes
        verify_resource_permission(perms, "batch")

        # But Compliance resource without compliance view is denied
        with pytest.raises(PermissionDeniedError):
            verify_resource_permission(perms, "compliance")


# ── TEST 6: AUTHENTICATION ENFORCEMENT & DEPENDENCY OVERRIDE ISOLATION ─────────
def test_require_facility_user_unauthenticated_returns_401() -> None:
    """Real require_facility_user rejects requests lacking Authorization headers."""
    client = TestClient(app)
    response = client.post("/v1/chat", json={"message": "hello"})
    assert response.status_code == 401
    assert "authentication required" in response.text.lower()


def test_require_facility_user_overridable_via_dependency_overrides() -> None:
    """FastAPI standard dependency_overrides replaces require_facility_user cleanly without production mock checks."""
    fake_user = AuthenticatedUser(
        user_id="fake-dep-user-99",
        name="Fake Dep User",
        raw_claims={"userId": "fake-dep-user-99"},
        token="fake-override-token",
    )

    app.dependency_overrides[require_facility_user] = lambda: fake_user
    client = TestClient(app)

    with patch("app.api.chat.resolve_permissions", return_value=ChatPermissions(Dashboard_assign="All")):
        response = client.post(
            "/v1/chat",
            json={"message": "hello"},
        )
    assert response.status_code == 200


def test_dependency_override_does_not_leak_to_subsequent_test() -> None:
    """Fixture cleanup ensures dependency overrides from previous tests are cleared."""
    client = TestClient(app)
    # Without Authorization header and without dependency override, must return 401
    response = client.post("/v1/chat", json={"message": "hello"})
    assert response.status_code == 401

