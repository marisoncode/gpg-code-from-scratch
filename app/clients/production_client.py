"""Production API client for communicating with CPG Production microservices."""

from __future__ import annotations

import logging
from typing import Any

from app.clients.base_client import _fetch_from_api, normalize_cpg_record
from app.core.config import settings

logger = logging.getLogger(__name__)


class ProductionClient:
    """Client for CPG Production microservice (BatchRecord, Inventories)."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.production_api or "").strip()

    async def get_batch_by_id(
        self,
        batch_id: str,
        token: str | None = None,
    ) -> tuple[int, Any]:
        """Fetch batch record by ID or Lot Number directly from Production API."""
        clean_id = (batch_id or "").strip()
        if not clean_id:
            return 400, None

        is_guid = len(clean_id) == 36 and clean_id.count("-") == 4
        if is_guid:
            code, data = await _fetch_from_api(
                self.base_url,
                f"BatchRecord/{clean_id}",
                token=token,
            )
            if code == 200 and data and isinstance(data, dict):
                return 200, data

        def _matches(item: Any) -> bool:
            if not isinstance(item, dict):
                return False
            for k in ("Batch_number", "Lot_number", "batch_number", "lot_number", "id", "BatchNumber", "LotNumber"):
                val = str(item.get(k) or "").strip()
                if val and val.lower() == clean_id.lower():
                    return True
            return False

        # 1. Try direct Lot_number filter
        code, data = await _fetch_from_api(
            self.base_url,
            "BatchRecord",
            params={"Lot_number": clean_id},
            token=token,
        )
        if code == 200 and data:
            records = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
            for item in records:
                if _matches(item):
                    return 200, item
            if len(records) == 1 and isinstance(records[0], dict):
                if clean_id.lower() in str(records[0]).lower():
                    return 200, records[0]

        # 2. Try direct REST path /BatchRecord/{clean_id}
        if not is_guid:
            code_d, data_d = await _fetch_from_api(
                self.base_url,
                f"BatchRecord/{clean_id}",
                token=token,
            )
            if code_d == 200 and data_d and isinstance(data_d, dict):
                if _matches(data_d) or clean_id.lower() in str(data_d).lower():
                    return 200, data_d

        # 3. Fallback: query paginated batches and match locally
        code_p, data_p = await _fetch_from_api(
            self.base_url,
            "BatchRecord",
            params={"page": 1, "pageSize": 100},
            token=token,
        )
        if code_p == 200 and data_p:
            records_p = data_p if isinstance(data_p, list) else ([data_p] if isinstance(data_p, dict) else [])
            for item in records_p:
                if _matches(item):
                    return 200, item

        return 404, None

    async def get_batches(
        self,
        status: str | None = None,
        limit: int = 10,
        token: str | None = None,
    ) -> tuple[int, Any]:
        """Query batch records with optional status filter directly from Production API."""
        params: dict[str, Any] = {"page": 1, "pageSize": min(limit, 50)}
        if status:
            params["Status"] = status.strip()
        return await _fetch_from_api(
            self.base_url,
            "BatchRecord",
            params=params,
            token=token,
        )

    async def get_material_inventory(
        self,
        lot_number: str,
        token: str | None = None,
    ) -> tuple[int, Any]:
        """Query component or chemical child inventory by lot number directly from Production API."""
        clean_lot = (lot_number or "").strip()
        code, data = await _fetch_from_api(
            self.base_url,
            "ComponentChildInventory",
            params={"Lot_number": clean_lot},
            token=token,
        )
        if code != 200 or not data:
            code, data = await _fetch_from_api(
                self.base_url,
                "ChemicalChildInventory",
                params={"Lot_number": clean_lot},
                token=token,
            )
        return code, data
