"""Order API client for communicating with CPG Order microservices."""

from __future__ import annotations

import logging
from typing import Any

from app.clients.base_client import _fetch_from_api
from app.core.config import settings

logger = logging.getLogger(__name__)


class OrderClient:
    """Client for CPG Order microservice."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.order_api or "").strip()

    async def get_order_by_id(
        self,
        order_id: str,
        token: str | None = None,
    ) -> tuple[int, Any]:
        clean_id = (order_id or "").strip()
        return await _fetch_from_api(
            self.base_url,
            f"Order/{clean_id}",
            token=token,
            collection_fallback="orders",
        )

