"""Production + Compliance dashboard reads for the chatbot.

Powered by facility_api_service (Cosmos DB read-only queries) instead of HTTP APIs.
Maintains existing function signatures for seamless integration with app/api/chat.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status

from app.services.capabilities import ChatPermissions, DashboardScope
from app.services.facility_api_service import (
    get_compliance_dashboard_data,
    get_production_dashboard_data,
)
from app.services.permissions_service import PermissionDeniedError


def _date_range() -> tuple[str, str]:
    return "2000-01-01", datetime.now(timezone.utc).date().isoformat()


async def fetch_production(
    client: Any = None,
    *,
    user_id: str = "",
    start_date: str = "",
    end_date: str = "",
    permissions: ChatPermissions | None = None,
    token: str = "",
) -> dict[str, Any]:
    """Fetch production snapshot from facility_api_service."""
    try:
        return get_production_dashboard_data(user_id=user_id, permissions=permissions, token=token)
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)
    except Exception as exc:
        return {"scope": "production", "errors": [str(exc)]}


async def fetch_compliance(
    client: Any = None,
    *,
    user_id: str = "",
    start_date: str = "",
    end_date: str = "",
    permissions: ChatPermissions | None = None,
    token: str = "",
) -> dict[str, Any]:
    """Fetch compliance snapshot from facility_api_service."""
    try:
        return get_compliance_dashboard_data(user_id=user_id, permissions=permissions, token=token)
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)
    except Exception as exc:
        return {"scope": "compliance", "errors": [str(exc)]}


async def fetch_dashboard_snapshot(
    *,
    token: str = "",
    user_id: str = "",
    username: str = "",
    scopes: set[DashboardScope],
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Fetch dashboard snapshot for requested scopes using facility_api_service.
    Signature maintained for app/api/chat.py compatibility.
    """
    start_date, end_date = _date_range()
    snapshot: dict[str, Any] = {
        "range": {"startDate": start_date, "endDate": end_date},
        "scopes": sorted(scopes),
    }

    if "production" in scopes:
        snapshot["production"] = await fetch_production(user_id=user_id, permissions=permissions, token=token)

    if "compliance" in scopes:
        snapshot["compliance"] = await fetch_compliance(user_id=user_id, permissions=permissions, token=token)

    return snapshot


def format_snapshot_for_prompt(snapshot: dict[str, Any]) -> str:
    """Format dashboard snapshot into grounded context for the system prompt."""
    compact = json.dumps(snapshot, default=str, separators=(",", ":"))
    if len(compact) > 6000:
        compact = compact[:6000] + "…"

    return (
        "Live dashboard snapshot (authoritative — do not invent numbers).\n"
        "Summarize only what is present. If a scope is missing or unavailable, do not guess.\n"
        "Every factual statement must reference the specific record or section.\n"
        "Deep links: Production /dashboard/dashboard-production ; "
        "Compliance /dashboard/dashboard-compliance.\n"
        f"{compact}"
    )
