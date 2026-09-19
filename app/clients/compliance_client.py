"""Compliance API client for communicating with CPG Compliance microservices."""

from __future__ import annotations

import logging
from typing import Any

from app.clients.base_client import _fetch_from_api
from app.core.config import settings

logger = logging.getLogger(__name__)


class ComplianceClient:
    """Client for CPG Compliance microservice (Training, Equipment, Deviations, Environmental Monitoring)."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.compliance_api or "").strip()

    async def get_training_by_operator(
        self,
        operator_id: str,
        token: str | None = None,
    ) -> tuple[int, Any]:
        """Fetch operator qualification / training records directly from Compliance API."""
        clean_id = (operator_id or "").strip()
        code, data = await _fetch_from_api(
            self.base_url,
            "Training",
            params={"Name": clean_id},
            token=token,
        )
        if code != 200 or not data:
            code, data = await _fetch_from_api(
                self.base_url,
                f"Training/{clean_id}",
                token=token,
            )
        return code, data

    async def get_equipment(
        self,
        equipment_id: str,
        token: str | None = None,
    ) -> tuple[int, Any]:
        """Fetch equipment status and PM records directly from Compliance API."""
        clean_id = (equipment_id or "").strip()
        is_guid = len(clean_id) == 36 and clean_id.count("-") == 4
        if is_guid:
            return await _fetch_from_api(
                self.base_url,
                f"Equipment/{clean_id}",
                token=token,
            )
        code, data = await _fetch_from_api(
            self.base_url,
            "Equipment",
            params={"Name": clean_id},
            token=token,
        )
        if code != 200 or not data:
            code, data = await _fetch_from_api(
                self.base_url,
                f"Equipment/{clean_id}",
                token=token,
            )
        return code, data

    async def get_deviation(
        self,
        deviation_id: str,
        token: str | None = None,
    ) -> tuple[int, Any]:
        """Fetch deviation record from TaskManagement directly from Compliance API."""
        clean_id = (deviation_id or "").strip()
        is_guid = len(clean_id) == 36 and clean_id.count("-") == 4
        if is_guid:
            return await _fetch_from_api(
                self.base_url,
                f"v1/TaskManagement/{clean_id}",
                token=token,
            )
        code, data = await _fetch_from_api(
            self.base_url,
            "v1/TaskManagement",
            params={"searchKey": clean_id, "Module_type": "DEV"},
            token=token,
        )
        if code != 200 or not data:
            code, data = await _fetch_from_api(
                self.base_url,
                f"v1/TaskManagement/{clean_id}",
                token=token,
            )
        return code, data

    async def get_environmental_monitoring(
        self,
        location_id: str | None = None,
        limit: int = 10,
        token: str | None = None,
    ) -> tuple[int, Any]:
        """Fetch environmental monitoring records directly from Compliance API."""
        params: dict[str, Any] = {"pageSize": min(limit, 50)}
        if location_id:
            params["Location_name"] = location_id.strip()
        return await _fetch_from_api(
            self.base_url,
            "EnvironmentalMonitoring",
            params=params,
            token=token,
        )
