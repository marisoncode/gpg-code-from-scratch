"""Facility API client for communicating with CPG Facility User microservices."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

from app.clients.base_client import _fetch_from_api
from app.core.config import settings

logger = logging.getLogger(__name__)


class FacilityClient:
    """Client for CPG Facility microservice (DashboardProduction, DashboardMonitor, ManageUser)."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.facility_api or "").strip()

    async def get_production_dashboard(
        self,
        token: str | None = None,
    ) -> tuple[int, Any]:
        """Fetch production dashboard snapshot."""
        year_start = f"{datetime.now(timezone.utc).year}-01-01"
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        params = {"startDate": year_start, "endDate": now_date}
        return await _fetch_from_api(
            self.base_url,
            "DashboardProduction/GetProductionDetails",
            params=params,
            token=token,
        )

    async def get_compliance_dashboard(
        self,
        token: str | None = None,
    ) -> tuple[int, Any]:
        """Fetch compliance dashboard snapshot via verified live endpoint."""
        return await _fetch_from_api(
            self.base_url,
            "DashboardMonitor/GetTodayTask",
            token=token,
        )

    async def get_user_roles(
        self,
        token: str | None = None,
    ) -> tuple[int, Any]:
        """Fetch all defined user roles from CPG Facility User API."""
        return await _fetch_from_api(
            self.base_url,
            "UserRole",
            token=token,
        )

    async def get_user_role_by_id(
        self,
        role_id: str,
        token: str | None = None,
    ) -> tuple[int, Any]:
        """Fetch a specific user role definition by ID or Role_id."""
        clean_id = (role_id or "").strip()
        return await _fetch_from_api(
            self.base_url,
            f"UserRole/{clean_id}",
            token=token,
        )

    async def get_user_by_id(
        self,
        user_id: str,
        token: str | None = None,
    ) -> tuple[int, Any]:
        """Fetch user profile and permission settings by GUID."""
        clean_id = (user_id or "").strip()
        return await _fetch_from_api(
            self.base_url,
            f"ManageUser/{clean_id}",
            token=token,
        )

