"""Security tests verifying explicit failure on missing user identity.

Verifies:
1. `_get_headers` raises `IdentityResolutionError` (HTTP 500) when user identity is empty.
2. `resolve_permissions` raises `IdentityResolutionError` (HTTP 500) when user identity is empty.
3. Neither function falls back to legacy hardcoded values ('Pukazh Vel' or '3ea3f803-2512-4cee-821c-5357504bd2a0').
4. Valid context (`set_current_user`) or token claims correctly attribute requests to that user.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt
import pytest

from app.services import facility_api_service
from app.services.capabilities import ChatPermissions, ModulePermission
from app.services.facility_api_service import (
    _get_headers,
    get_batch_by_id,
    set_current_token,
    set_current_user,
)
from app.services.permissions_service import (
    IdentityResolutionError,
    resolve_permissions,
)

LEGACY_NAME = "Pukazh Vel"
LEGACY_GUID = "3ea3f803-2512-4cee-821c-5357504bd2a0"


# ── TEST 1: _get_headers FAILS WHEN IDENTITY MISSING ─────────────────────────
def test_get_headers_raises_when_no_user_identity() -> None:
    """_get_headers must raise IdentityResolutionError instead of using hardcoded fallback."""
    set_current_user(user_id="", username="")
    set_current_token("")

    with pytest.raises(IdentityResolutionError) as exc_info:
        _get_headers()

    assert exc_info.value.status_code == 500
    assert "user_id is missing from context and token" in str(exc_info.value.detail)


def test_get_headers_raises_when_token_has_no_user_claims() -> None:
    """_get_headers must raise IdentityResolutionError if token provides no user identity."""
    set_current_user(user_id="", username="")
    token_without_user = jwt.encode(
        {"some_claim": "value"},
        "dummy-secret",
        algorithm="HS256",
    )

    with pytest.raises(IdentityResolutionError) as exc_info:
        _get_headers(token=token_without_user)

    assert exc_info.value.status_code == 500
    assert "user_id is missing" in str(exc_info.value.detail)


# ── TEST 2: _get_headers SUCCEEDS WITH VALID CONTEXT OR TOKEN ────────────────
def test_get_headers_uses_context_identity_and_never_hardcoded() -> None:
    """_get_headers must use context identity and NEVER use legacy hardcoded identity."""
    set_current_user(user_id="active-user-999", username="Active Operator", collection_id="prod-col")
    try:
        headers = _get_headers()
        assert headers["userid"] == "active-user-999"
        assert headers["username"] == "Active Operator"
        assert headers["collectionid"] == "prod-col"

        # Explicitly verify legacy hardcoded values are NOT present
        assert headers["userid"] != LEGACY_GUID
        assert headers["username"] != LEGACY_NAME
    finally:
        set_current_user(user_id="", username="", collection_id="")


def test_get_headers_extracts_identity_from_valid_token() -> None:
    """_get_headers extracts user from token claims when context is empty."""
    set_current_user(user_id="", username="")
    token = jwt.encode(
        {"sub": "token-extracted-user", "name": "Token Extracted Name"},
        "dummy-secret",
        algorithm="HS256",
    )

    headers = _get_headers(token=token)
    assert headers["userid"] == "token-extracted-user"
    assert headers["username"] == "Token Extracted Name"
    assert headers["userid"] != LEGACY_GUID
    assert headers["username"] != LEGACY_NAME


# ── TEST 3: resolve_permissions FAILS WHEN IDENTITY MISSING ───────────────────
@pytest.mark.asyncio
async def test_resolve_permissions_raises_when_identity_cannot_be_resolved() -> None:
    """resolve_permissions must raise IdentityResolutionError when user_id is missing."""
    set_current_user(user_id="", username="")
    token_without_user = jwt.encode(
        {"exp": 9999999999},
        "dummy-secret",
        algorithm="HS256",
    )

    with pytest.raises(IdentityResolutionError) as exc_info:
        await resolve_permissions(token_without_user, user_id="")

    assert exc_info.value.status_code == 500
    assert "user_id is missing from token, parameters, and context" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_resolve_permissions_succeeds_with_token_identity() -> None:
    """resolve_permissions resolves with token claims and does not fall back to legacy values."""
    token = jwt.encode(
        {"userId": "user-valid-123", "name": "Valid User", "exp": 9999999999},
        "dummy-secret",
        algorithm="HS256",
    )

    captured_headers = {}

    async def mock_get(url: str, headers: dict[str, str], **kwargs):
        nonlocal captured_headers
        captured_headers = dict(headers)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {
            "Dashboard_assign": "All",
            "modules": [{"Module_name": "Batch Record", "View": True}],
        }
        return mock_resp

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        perms = await resolve_permissions(token)
        assert perms.Dashboard_assign == "All"
        assert captured_headers["userid"] == "user-valid-123"
        assert captured_headers["username"] == "Valid User"
        assert captured_headers["userid"] != LEGACY_GUID
        assert captured_headers["username"] != LEGACY_NAME


# ── TEST 4: DOWNSTREAM API CALL FAILS LOUDLY WITH NO IDENTITY IN CONTEXT ──────
@pytest.mark.asyncio
async def test_downstream_api_call_raises_identity_error_when_context_missing() -> None:
    """Calling downstream API functions without user identity raises IdentityResolutionError."""
    set_current_user(user_id="", username="")
    set_current_token("")

    mock_client = MagicMock()
    facility_api_service.set_http_client(mock_client)

    qa_perms = ChatPermissions(
        Dashboard_assign="All",
        modules=[ModulePermission(Module_name="Batch Record", View=True)],
    )

    try:
        with pytest.raises(IdentityResolutionError) as exc_info:
            await get_batch_by_id("B-100", permissions=qa_perms)

        assert exc_info.value.status_code == 500
        mock_client.get.assert_not_called()
    finally:
        facility_api_service.set_http_client(None)

