"""Read-only data access facade and service bridge for CPG microservices.

This module acts as a facade over:
- `app.clients`: Low-level HTTP communication, header management, and response unwrap
- `app.ai`: Predefined business tools, schemas, and secure dispatch
"""

from __future__ import annotations

import logging
from typing import Any

# Re-export client infrastructure and contextvars
from app.clients.base_client import (
    _build_api_url,
    _fetch_from_api,
    _get_headers,
    _get_http_client,
    _http_client_override,
    _infer_service,
    _request_collection_id,
    _request_token,
    _request_user_id,
    _request_user_name,
    _safe_fetch_items,
    _shared_async_client,
    _triggered_endpoints,
    close_http_client,
    get_current_collection_id,
    get_current_token,
    get_current_user_id,
    get_current_user_name,
    get_triggered_endpoints,
    init_http_client,
    normalize_cpg_record,
    record_triggered_endpoint,
    reset_triggered_endpoints,
    set_current_token,
    set_current_user,
    set_http_client,
    unwrap_cpg_response,
)
from app.clients.compliance_client import ComplianceClient
from app.clients.facility_client import FacilityClient
from app.clients.order_client import OrderClient
from app.clients.production_client import ProductionClient

# Re-export AI tools and executor
from app.ai.executor import PREDEFINED_FUNCTIONS, execute_predefined_tool
from app.ai.schemas import PREDEFINED_TOOL_SCHEMAS
from app.ai.tools import (
    calculate_risk_score,
    detect_trends,
    find_similar_batches,
    get_batch_by_id,
    get_batches,
    get_compliance_dashboard_data,
    get_component_lot_genealogy,
    get_deviation_by_id,
    get_deviation_impact,
    get_environmental_monitoring,
    get_equipment_batch_history,
    get_equipment_status,
    get_finished_drug_genealogy,
    get_material_lot_genealogy,
    get_material_lot_trace,
    get_operator_batch_history,
    get_operator_training_status,
    get_production_dashboard_data,
    get_related_batches,
    investigate_entity,
    recommend_batch_configuration,
)
from app.services.permissions_service import (
    IdentityResolutionError,
    PermissionDeniedError,
    verify_resource_permission,
)

logger = logging.getLogger(__name__)


class FacilitySecurityError(Exception):
    """Raised when security boundaries are violated."""


__all__ = [
    # Clients & Context
    "reset_triggered_endpoints",
    "record_triggered_endpoint",
    "get_triggered_endpoints",
    "set_current_token",
    "get_current_token",
    "set_current_user",
    "get_current_user_id",
    "get_current_user_name",
    "get_current_collection_id",
    "init_http_client",
    "close_http_client",
    "set_http_client",
    "normalize_cpg_record",
    "unwrap_cpg_response",
    "ProductionClient",
    "ComplianceClient",
    "FacilityClient",
    "OrderClient",
    # Tools & Execution
    "PREDEFINED_FUNCTIONS",
    "PREDEFINED_TOOL_SCHEMAS",
    "execute_predefined_tool",
    "get_batch_by_id",
    "get_batches",
    "get_operator_training_status",
    "get_equipment_status",
    "get_material_lot_trace",
    "get_deviation_by_id",
    "get_environmental_monitoring",
    "get_production_dashboard_data",
    "get_compliance_dashboard_data",
    "get_related_batches",
    "get_material_lot_genealogy",
    "get_component_lot_genealogy",
    "get_operator_batch_history",
    "get_equipment_batch_history",
    "get_finished_drug_genealogy",
    "get_deviation_impact",
    "investigate_entity",
    "find_similar_batches",
    "calculate_risk_score",
    "detect_trends",
    "recommend_batch_configuration",
    "FacilitySecurityError",
    "verify_resource_permission",
    "PermissionDeniedError",
    "IdentityResolutionError",
]
