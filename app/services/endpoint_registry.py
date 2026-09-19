"""Centralized, Production-Grade Downstream API Endpoint Registry.

Indexes and manages all 316 CPG microservice endpoints across:
- Facility User API (facility_user_api_endpoints.json)
- Production API (production_api_endpoints.json)
- Compliance API (compliance_api_endpoints.json)
- Order API (order_api_endpoints.json)

Provides O(1) in-memory resolution, type-safe definitions, and centralized
configuration aliases via app/config/endpoints_config.json.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EndpointDefinition(BaseModel):
    service: str
    method: str = "GET"
    path: str
    tags: list[str] = Field(default_factory=list)
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    description: str = ""
    operation_id: str = ""

    @property
    def clean_path(self) -> str:
        """Returns path without leading '/api/' prefix for clean URL concatenation."""
        p = self.path.strip().lstrip("/")
        if p.startswith("api/"):
            return p[4:]
        return p


class EndpointRegistry:
    """In-memory singleton registry indexing all verified CPG microservice endpoints."""

    def __init__(self, endpoints_dir: str | Path | None = None) -> None:
        self.endpoints_dir = Path(
            endpoints_dir
            or os.getenv("ENDPOINTS_DIR")
            or Path(__file__).parent.parent / "config" / "endpoints"
        )
        self.fallback_dir = Path(__file__).resolve().parent.parent.parent
        self.config_file = Path(__file__).parent.parent / "config" / "endpoints_config.json"

        # In-memory indices
        self._endpoints: list[EndpointDefinition] = []
        self._by_service_and_path: dict[tuple[str, str, str], EndpointDefinition] = {}
        self._by_service_and_tag: dict[tuple[str, str, str], list[EndpointDefinition]] = {}
        self._aliases: dict[str, dict[str, str]] = {}
        self._loaded: bool = False

        self.load()

    def _find_file(self, filename: str) -> Path | None:
        primary = self.endpoints_dir / filename
        if primary.exists():
            return primary
        fallback = self.fallback_dir / filename
        if fallback.exists():
            return fallback
        return None

    def load(self) -> None:
        """Loads and indexes all endpoint JSON files and configuration aliases."""
        self._endpoints.clear()
        self._by_service_and_path.clear()
        self._by_service_and_tag.clear()
        self._aliases.clear()

        # 1. Load logical aliases from config
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self._aliases = cfg.get("aliases", {})
            except Exception as exc:
                logger.warning("Could not load endpoints_config.json: %s", exc)

        # 2. Service definitions mapping
        service_files = {
            "facility_user": "facility_user_api_endpoints.json",
            "production": "production_api_endpoints.json",
            "compliance": "compliance_api_endpoints.json",
            "order": "order_api_endpoints.json",
        }

        for service, filename in service_files.items():
            filepath = self._find_file(filename)
            if not filepath:
                logger.warning("Endpoint specification file %s not found.", filename)
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_list = json.load(f)

                if isinstance(raw_list, list):
                    for item in raw_list:
                        method = str(item.get("method", "GET")).upper()
                        path = str(item.get("path", "")).strip()
                        tags = item.get("tags", [])
                        params = item.get("parameters", [])
                        summary = item.get("summary", "")
                        desc = item.get("description", "")
                        op_id = item.get("operationId", "")

                        endpoint = EndpointDefinition(
                            service=service,
                            method=method,
                            path=path,
                            tags=tags,
                            parameters=params,
                            summary=summary,
                            description=desc,
                            operation_id=op_id,
                        )
                        self._endpoints.append(endpoint)

                        # Index by (service, method, path)
                        norm_path = "/" + path.strip().lstrip("/")
                        self._by_service_and_path[(service, method, norm_path)] = endpoint
                        # Also index by lowercase for case-insensitive resolution
                        self._by_service_and_path[(service, method, norm_path.lower())] = endpoint

                        # Index by (service, method, tag)
                        for tag in tags:
                            tag_key = (service, method, tag.lower())
                            self._by_service_and_tag.setdefault(tag_key, []).append(endpoint)

                logger.info("Loaded %d endpoints for %s from %s", len(raw_list), service, filepath.name)

            except Exception as exc:
                logger.error("Failed to parse endpoint file %s: %s", filepath, exc)

        self._loaded = True
        logger.info("Endpoint Registry initialized with %d total endpoints.", len(self._endpoints))

    def get_total_count(self) -> int:
        return len(self._endpoints)

    def resolve_path(self, service: str, key_or_tag_or_path: str, method: str = "GET") -> str:
        """
        Resolves an endpoint path string from:
        1. An alias key defined in endpoints_config.json (e.g. 'production_dashboard_details')
        2. A verified endpoint path (e.g. '/api/DashboardProduction/GetProductionDetails')
        3. A controller/tag name (e.g. 'BatchRecord')

        Returns clean relative path (without '/api/' prefix) ready for _build_api_url.
        """
        srv = service.lower().strip()
        meth = method.upper().strip()
        query = key_or_tag_or_path.strip()

        # A. Check logical aliases first
        if srv in self._aliases and query in self._aliases[srv]:
            full_path = self._aliases[srv][query]
            return self._strip_api_prefix(full_path)

        # B. Check exact or normalized path in registry
        norm_path = "/" + query.lstrip("/")
        if not norm_path.startswith("/api/"):
            norm_path = "/api" + norm_path

        key = (srv, meth, norm_path)
        if key in self._by_service_and_path:
            return self._strip_api_prefix(self._by_service_and_path[key].path)

        key_lower = (srv, meth, norm_path.lower())
        if key_lower in self._by_service_and_path:
            return self._strip_api_prefix(self._by_service_and_path[key_lower].path)

        # C. Check by Tag
        tag_key = (srv, meth, query.lower())
        if tag_key in self._by_service_and_tag:
            return self._strip_api_prefix(self._by_service_and_tag[tag_key][0].path)

        # D. Safe Fallback: return stripped path directly
        return self._strip_api_prefix(query)

    @staticmethod
    def _strip_api_prefix(path: str) -> str:
        p = path.strip().lstrip("/")
        if p.startswith("api/"):
            return p[4:]
        return p

    def find_endpoints(
        self,
        service: str | None = None,
        tag: str | None = None,
        method: str | None = None,
    ) -> list[EndpointDefinition]:
        """Find endpoints filtered by service, tag, or HTTP method."""
        results = self._endpoints
        if service:
            s = service.lower().strip()
            results = [e for e in results if e.service == s]
        if method:
            m = method.upper().strip()
            results = [e for e in results if e.method == m]
        if tag:
            t = tag.lower().strip()
            results = [e for e in results if any(tag.lower() == t for tag in e.tags)]
        return results


# Global singleton instance
registry = EndpointRegistry()


def get_endpoint_path(service: str, key_or_tag_or_path: str, method: str = "GET") -> str:
    """Public helper to resolve a clean endpoint path string."""
    return registry.resolve_path(service, key_or_tag_or_path, method=method)

