"""CPG Microservice API Clients layer.

Provides dedicated client classes handling:
- HTTP requests
- URLs and endpoint resolution
- Header construction and token forwarding
- Timeouts and HTTP errors
- Response unwrap and key normalization
"""

from app.clients.base_client import (
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

__all__ = [
    "ProductionClient",
    "ComplianceClient",
    "FacilityClient",
    "OrderClient",
    "init_http_client",
    "close_http_client",
    "set_http_client",
    "get_current_token",
    "set_current_token",
    "get_current_user_id",
    "get_current_user_name",
    "get_current_collection_id",
    "set_current_user",
    "get_triggered_endpoints",
    "record_triggered_endpoint",
    "reset_triggered_endpoints",
    "normalize_cpg_record",
    "unwrap_cpg_response",
]
