"""Integration tests verifying dashboard chat permission gating and context loading.

Verifies:
1. Valid Production user requesting dashboard summary gets 200 with dashboard data.
2. Sales user requesting dashboard summary gets 200 clean denial (not 500).
3. User with empty/none permissions gets 200 clean denial (not 500).
4. No 500 Internal Server Error is ever raised on permission denials.
"""

from __future__ import annotations

from unittest.mock import patch

import jwt
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import facility_api_service
from app.services.capabilities import ChatPermissions, ModulePermission


@pytest.fixture
def auth_token() -> str:
    return jwt.encode(
        {"userId": "user-test-101", "name": "Test User", "sub": "test@cpg.com"},
        "dummy-secret-key-at-least-32-chars-long!",
        algorithm="HS256",
    )


def test_production_dashboard_chat_with_valid_permissions(auth_token: str) -> None:
    """User with Production assignment and View permission gets 200 with dashboard content."""
    client = TestClient(app)
    prod_permissions = ChatPermissions(
        Dashboard_assign="Production",
        modules=[
            ModulePermission(Module_name="Batch Record", View=True),
            ModulePermission(Module_name="Inventory", View=True),
        ],
    )

    mock_batches = [
        {"id": "B-2001", "product": "Aspirin 500mg", "status": "IN_PROGRESS"},
        {"id": "B-2002", "product": "Paracetamol 650mg", "status": "ACTIVE"},
    ]

    with patch("app.api.chat.resolve_permissions", return_value=prod_permissions), \
         patch.object(facility_api_service, "_get_business_items", return_value=mock_batches):

        response = client.post(
            "/v1/chat",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"message": "show me today's production dashboard summary"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "thread_id" in data
    msg = data["message"].lower()
    # Confirm it returned dashboard content without permission denial or 500 error
    assert "cannot access" not in msg
    assert "access denied" not in msg
    assert "chat failed" not in msg


def test_production_dashboard_chat_with_sales_only_permissions(auth_token: str) -> None:
    """User with Sales-only role gets a clean user-facing denial, not a 500 error."""
    client = TestClient(app)
    sales_permissions = ChatPermissions(
        Dashboard_assign="Sales",
        modules=[],
    )

    with patch("app.api.chat.resolve_permissions", return_value=sales_permissions):
        response = client.post(
            "/v1/chat",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"message": "show me today's production dashboard summary"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    msg = data["message"].lower()
    # Must be a clean, user-facing denial message, NOT a 500 or stack trace
    assert "sales" in msg or "cannot access" in msg or "not available" in msg
    assert "internal server error" not in msg
    assert "chat failed" not in msg


def test_production_dashboard_chat_with_empty_or_none_permissions(auth_token: str) -> None:
    """User with empty/None permissions gets a clean user-facing denial, not a 500 error."""
    client = TestClient(app)
    empty_permissions = ChatPermissions(
        Dashboard_assign=None,
        modules=[],
    )

    with patch("app.api.chat.resolve_permissions", return_value=empty_permissions):
        response = client.post(
            "/v1/chat",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"message": "show me today's production dashboard summary"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    msg = data["message"].lower()
    # Must be a clean denial message
    assert "no active" in msg or "denied" in msg or "cannot access" in msg
    assert "internal server error" not in msg
    assert "chat failed" not in msg

