"""Security tests verifying prevention of facility_user_id identity spoofing / privilege escalation.

Verifies:
1. `_facility_identity` strictly uses the verified token user_id and ignores client-supplied `facility_user_id`.
2. POST `/v1/chat` resolves permissions using User A's token-derived identity, ignoring spoofed User B `facility_user_id`.
3. POST `/v1/chat/dashboard-summary` resolves permissions using User A's token identity.
4. `resolve_permissions` in `permissions_service.py` strictly derives user_id from token claims over any unverified parameter.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient

from app.api.chat import _facility_identity
from app.core.auth import AuthenticatedUser
from app.main import app
from app.services.capabilities import ChatPermissions, ModulePermission
from app.services.permissions_service import resolve_permissions


@pytest.fixture
def user_a_token() -> str:
    """User A token (Sales role, low privilege)."""
    return jwt.encode(
        {"userId": "user-a-sales-id", "name": "User A Sales", "sub": "user_a@cpg.com"},
        "dummy-secret-key-at-least-32-chars-long!",
        algorithm="HS256",
    )


def test_facility_identity_strictly_enforces_token_user_id(caplog: pytest.LogCaptureFixture) -> None:
    """_facility_identity must ignore client-supplied facility_user_id and log a security warning."""
    user = AuthenticatedUser(
        user_id="token-user-123",
        name="Token User",
        raw_claims={"userId": "token-user-123"},
        token="dummy-token",
    )

    with caplog.at_level(logging.WARNING):
        resolved_id, resolved_name = _facility_identity(
            user,
            facility_user_id="spoofed-admin-456",
            facility_user_name="Display Name Override",
        )

    # Authoritative ID must be strictly from token
    assert resolved_id == "token-user-123"
    # Display name may be overridden for UI presentation
    assert resolved_name == "Display Name Override"
    # Security warning must be logged
    assert any("differs from token user_id" in record.message for record in caplog.records)


def test_chat_endpoint_ignores_spoofed_facility_user_id(user_a_token: str) -> None:
    """
    User A (Sales) sends a request with higher-privileged User B's facility_user_id.
    The endpoint must resolve permissions using User A's token identity and reject access.
    """
    client = TestClient(app)

    sales_perms = ChatPermissions(
        Dashboard_assign="Sales",
        modules=[],
    )
    admin_perms = ChatPermissions(
        Dashboard_assign="All",
        modules=[
            ModulePermission(Module_name="Batch Record", View=True),
            ModulePermission(Module_name="Compliance", View=True),
        ],
    )

    resolved_user_ids = []

    async def mock_resolve(token: str, user_id: str | None = None) -> ChatPermissions:
        resolved_user_ids.append(user_id)
        if user_id == "user-b-admin-id":
            # Vulnerable path would grant full admin access
            return admin_perms
        return sales_perms

    with patch("app.api.chat.resolve_permissions", side_effect=mock_resolve):
        response = client.post(
            "/v1/chat",
            headers={"Authorization": f"Bearer {user_a_token}"},
            json={
                "message": "show me today's production summary",
                "facility_user_id": "user-b-admin-id",
                "facility_user_name": "Admin Impersonator",
            },
        )

    assert response.status_code == 200
    data = response.json()
    msg = data["message"].lower()

    # Access must be denied based on User A's sales permissions
    assert "sales" in msg or "cannot access" in msg or "not available" in msg
    # Crucially: resolve_permissions must NEVER have been called with User B's ID
    assert "user-b-admin-id" not in resolved_user_ids
    assert "user-a-sales-id" in resolved_user_ids


def test_dashboard_summary_ignores_spoofed_facility_user_id(user_a_token: str) -> None:
    """
    POST /v1/chat/dashboard-summary must not allow privilege escalation
    via client-supplied facility_user_id.
    """
    client = TestClient(app)

    sales_perms = ChatPermissions(Dashboard_assign="Sales", modules=[])
    admin_perms = ChatPermissions(Dashboard_assign="All", modules=[ModulePermission(Module_name="Batch Record", View=True)])

    resolved_user_ids = []

    async def mock_resolve(token: str, user_id: str | None = None) -> ChatPermissions:
        resolved_user_ids.append(user_id)
        if user_id == "user-b-admin-id":
            return admin_perms
        return sales_perms

    with patch("app.api.chat.resolve_permissions", side_effect=mock_resolve):
        response = client.post(
            "/v1/chat/dashboard-summary",
            headers={"Authorization": f"Bearer {user_a_token}"},
            json={
                "message": "dashboard summary",
                "facility_user_id": "user-b-admin-id",
            },
        )

    assert response.status_code == 200
    data = response.json()
    # Scopes must be empty because User A has Sales role
    assert data["allowed"] == []
    assert "sales" in data["refused"].lower() or "cannot access" in data["refused"].lower()
    # Confirm User B's ID was never used for permission resolution
    assert "user-b-admin-id" not in resolved_user_ids
    assert "user-a-sales-id" in resolved_user_ids


@pytest.mark.asyncio
async def test_resolve_permissions_service_prioritizes_token_claims(user_a_token: str) -> None:
    """
    Even when called directly with an adversarial user_id, resolve_permissions
    strictly uses the token-derived identity.
    """
    captured_headers = {}

    from unittest.mock import MagicMock

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.text = '{"Dashboard_assign": "Sales", "modules": []}'
    mock_resp.json.return_value = {"Dashboard_assign": "Sales", "modules": []}

    with patch("httpx.AsyncClient.get", return_value=mock_resp) as mock_get:
        perms = await resolve_permissions(user_a_token, user_id="user-b-admin-id")
        assert perms.Dashboard_assign == "Sales"

        # Check the headers sent in the HTTP request to the permissions authority
        _, kwargs = mock_get.call_args
        headers = kwargs.get("headers", {})
        # The userid header sent to the upstream authority must be User A's token ID, not User B's ID
        assert headers.get("userid") == "user-a-sales-id"
