"""Read-only data access service using Azure Cosmos DB and COSMOS_DB_READONLY_KEY.

Exposes a fixed set of named, parameterized, READ-ONLY functions only.
No create/update/delete/upsert methods anywhere.
Every function verifies the caller's ChatPermissions BEFORE running the query.
The LLM selects which predefined function to call and with what parameters.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError

from app.core.config import settings
from app.services.capabilities import ChatPermissions
from app.services.permissions_service import PermissionDeniedError, verify_resource_permission

logger = logging.getLogger(__name__)

# Disallow write-scoped credentials or audit/chat containers
_FORBIDDEN_CONTAINERS = {"audit_trail", "chat_history"}


class FacilitySecurityError(Exception):
    """Raised when security boundaries are violated."""


_client_override: CosmosClient | None = None


def set_cosmos_readonly_client(client: CosmosClient | None) -> None:
    """Used in integration tests to inject a dev/test CosmosClient."""
    global _client_override
    _client_override = client


def get_readonly_cosmos_client() -> CosmosClient:
    """
    Returns a CosmosClient configured strictly with COSMOS_DB_READONLY_KEY.
    Refuses to use COSMOS_DB_WRITE_KEY under any circumstances.
    """
    global _client_override
    if _client_override is not None:
        return _client_override

    endpoint = (settings.cosmos_db_endpoint or "").strip()
    key = (settings.cosmos_db_readonly_key or "").strip()

    if not endpoint:
        raise ValueError("COSMOS_DB_ENDPOINT is not configured.")
    if not key:
        raise ValueError("COSMOS_DB_READONLY_KEY is not configured.")

    # Explicit protection: ensure the write key is NEVER used
    if settings.cosmos_db_write_key and key == settings.cosmos_db_write_key:
        raise FacilitySecurityError(
            "Security violation: COSMOS_DB_WRITE_KEY cannot be used for business data queries."
        )

    return CosmosClient(endpoint, credential=key)


def _get_business_container(container_name: str):
    """Get a read-only container proxy, guarding against audit/chat containers."""
    c_name = container_name.strip()
    if c_name in _FORBIDDEN_CONTAINERS:
        raise FacilitySecurityError(
            f"Read-only facility service is forbidden from accessing write container: '{c_name}'"
        )
    client = get_readonly_cosmos_client()
    db = client.get_database_client(settings.cosmos_db_database)
    return db.get_container_client(c_name)


# ── PREDEFINED READ-ONLY FUNCTIONS ─────────────────────────────────────────────


def get_batch_by_id(
    batch_id: str,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Retrieve single batch record by ID.
    Enforces permission check BEFORE query.
    """
    verify_resource_permission(permissions, "batch")
    clean_id = (batch_id or "").strip()
    if not clean_id:
        return {"error": "batch_id parameter is required.", "status": "missing_parameter"}

    try:
        container = _get_business_container("batches")
        query = "SELECT * FROM c WHERE c.id = @id OR c.batch_number = @id"
        parameters = [{"name": "@id", "value": clean_id}]
        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
        if items:
            return {"batch": items[0], "record_id": items[0].get("id", clean_id), "status": "found"}
        return {
            "error": f"Batch record {clean_id} is unavailable or not found.",
            "status": "unavailable",
            "record_id": clean_id,
        }
    except FacilitySecurityError:
        raise
    except CosmosHttpResponseError as exc:
        logger.warning(f"Cosmos read error for batch {clean_id}: {exc}")
        return {"error": f"Database query failed: {exc}", "status": "unavailable", "record_id": clean_id}
    except Exception as exc:
        logger.warning(f"Cosmos connection error for batch {clean_id}: {exc}")
        return {"error": f"Batch record {clean_id} is unavailable.", "status": "unavailable", "record_id": clean_id}


def get_batches(
    status: str | None = None,
    limit: int = 10,
    permissions: ChatPermissions | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve batch records, optionally filtered by status.
    Enforces permission check BEFORE query.
    """
    verify_resource_permission(permissions, "batch")
    container = _get_business_container("batches")

    max_items = max(1, min(limit or 10, 50))
    if status and status.strip():
        query = f"SELECT TOP {max_items} * FROM c WHERE c.status = @status ORDER BY c.created_at DESC"
        parameters = [{"name": "@status", "value": status.strip()}]
    else:
        query = f"SELECT TOP {max_items} * FROM c ORDER BY c.created_at DESC"
        parameters = []

    try:
        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
        return items
    except Exception as exc:
        logger.warning(f"Cosmos read error in get_batches: {exc}")
        return []


def get_operator_training_status(
    operator_id: str,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Retrieve training status for an operator.
    Enforces permission check BEFORE query.
    """
    verify_resource_permission(permissions, "training")
    clean_id = (operator_id or "").strip()
    if not clean_id:
        return {"error": "operator_id parameter is required.", "status": "missing_parameter"}

    container = _get_business_container("training")
    query = "SELECT * FROM c WHERE c.operator_id = @op_id OR c.id = @op_id"
    parameters = [{"name": "@op_id", "value": clean_id}]

    try:
        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
        if items:
            return {"training_records": items, "record_id": clean_id, "status": "found"}
        return {
            "error": f"Operator training record for {clean_id} is unavailable or not found.",
            "status": "unavailable",
            "record_id": clean_id,
        }
    except Exception as exc:
        logger.warning(f"Cosmos read error for operator {clean_id}: {exc}")
        return {"error": str(exc), "status": "error"}


def get_equipment_status(
    equipment_id: str,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Retrieve equipment PM, calibration, and operational status.
    Enforces permission check BEFORE query.
    """
    verify_resource_permission(permissions, "equipment")
    clean_id = (equipment_id or "").strip()
    if not clean_id:
        return {"error": "equipment_id parameter is required.", "status": "missing_parameter"}

    container = _get_business_container("equipment")
    query = "SELECT * FROM c WHERE c.id = @eq_id OR c.asset_id = @eq_id"
    parameters = [{"name": "@eq_id", "value": clean_id}]

    try:
        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
        if items:
            return {"equipment": items[0], "record_id": clean_id, "status": "found"}
        return {
            "error": f"Equipment record for {clean_id} is unavailable or not found.",
            "status": "unavailable",
            "record_id": clean_id,
        }
    except Exception as exc:
        logger.warning(f"Cosmos read error for equipment {clean_id}: {exc}")
        return {"error": str(exc), "status": "error"}


def get_material_lot_trace(
    lot_number: str,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Trace a chemical or component lot through batches.
    Enforces permission check BEFORE query.
    """
    verify_resource_permission(permissions, "material")
    clean_lot = (lot_number or "").strip()
    if not clean_lot:
        return {"error": "lot_number parameter is required.", "status": "missing_parameter"}

    container = _get_business_container("materials")
    query = "SELECT * FROM c WHERE c.lot_number = @lot OR c.id = @lot"
    parameters = [{"name": "@lot", "value": clean_lot}]

    try:
        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
        if items:
            return {"material_lot": items[0], "record_id": clean_lot, "status": "found"}
        return {
            "error": f"Material lot record {clean_lot} is unavailable or not found.",
            "status": "unavailable",
            "record_id": clean_lot,
        }
    except Exception as exc:
        logger.warning(f"Cosmos read error for lot {clean_lot}: {exc}")
        return {"error": str(exc), "status": "error"}


def get_deviation_by_id(
    deviation_id: str,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Retrieve deviation / CAPA record by ID.
    Enforces permission check BEFORE query.
    """
    verify_resource_permission(permissions, "deviation")
    clean_id = (deviation_id or "").strip()
    if not clean_id:
        return {"error": "deviation_id parameter is required.", "status": "missing_parameter"}

    container = _get_business_container("deviations")
    query = "SELECT * FROM c WHERE c.id = @dev_id OR c.deviation_number = @dev_id"
    parameters = [{"name": "@dev_id", "value": clean_id}]

    try:
        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
        if items:
            return {"deviation": items[0], "record_id": clean_id, "status": "found"}
        return {
            "error": f"Deviation record {clean_id} is unavailable or not found.",
            "status": "unavailable",
            "record_id": clean_id,
        }
    except Exception as exc:
        logger.warning(f"Cosmos read error for deviation {clean_id}: {exc}")
        return {"error": str(exc), "status": "error"}


def get_environmental_monitoring(
    location_id: str | None = None,
    permissions: ChatPermissions | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve environmental monitoring records for a room / location.
    Enforces permission check BEFORE query.
    """
    verify_resource_permission(permissions, "environmental_monitoring")
    container = _get_business_container("environmental_monitoring")

    if location_id and location_id.strip():
        query = "SELECT TOP 20 * FROM c WHERE c.location_id = @loc ORDER BY c.timestamp DESC"
        parameters = [{"name": "@loc", "value": location_id.strip()}]
    else:
        query = "SELECT TOP 20 * FROM c ORDER BY c.timestamp DESC"
        parameters = []

    try:
        return list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
    except Exception as exc:
        logger.warning(f"Cosmos read error for EM: {exc}")
        return []


def get_production_dashboard_data(
    user_id: str = "",
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Read production KPIs, alerts, batches, and inventory from Cosmos DB.
    Enforces permission check BEFORE query.
    """
    verify_resource_permission(permissions, "production")
    out: dict[str, Any] = {
        "scope": "production",
        "kpis": {"totalBatch": 0, "inProgress": 0, "released": 0, "pending": 0, "hold": 0},
        "lowInventory": [],
        "alerts": [],
        "todaysBatches": [],
        "operators": [],
        "newProducts": [],
    }

    try:
        batches_c = _get_business_container("batches")
        batches = list(
            batches_c.query_items(
                query="SELECT TOP 10 * FROM c ORDER BY c.created_at DESC",
                enable_cross_partition_query=True,
            )
        )
        out["todaysBatches"] = [
            {
                "lot": str(b.get("lot_number") or b.get("id") or ""),
                "product": str(b.get("product") or ""),
                "status": str(b.get("status") or ""),
                "qty": b.get("quantity", 0),
            }
            for b in batches
        ]
        out["kpis"]["totalBatch"] = len(batches)
        out["kpis"]["inProgress"] = sum(1 for b in batches if b.get("status") == "IN_PROGRESS")
        out["kpis"]["released"] = sum(1 for b in batches if b.get("status") == "RELEASED")
    except Exception as exc:
        logger.warning(f"Cosmos production batches read error: {exc}")

    try:
        inv_c = _get_business_container("inventory")
        items = list(
            inv_c.query_items(
                query="SELECT TOP 5 * FROM c WHERE c.available_quantity < 50",
                enable_cross_partition_query=True,
            )
        )
        out["lowInventory"] = [
            {
                "name": str(i.get("name") or ""),
                "sku": str(i.get("sku") or ""),
                "available": i.get("available_quantity", 0),
            }
            for i in items
        ]
    except Exception as exc:
        logger.warning(f"Cosmos production inventory read error: {exc}")

    return out


def get_compliance_dashboard_data(
    user_id: str = "",
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Read compliance OOC details, tasks, and alerts from Cosmos DB.
    Enforces permission check BEFORE query.
    """
    verify_resource_permission(permissions, "compliance")
    out: dict[str, Any] = {
        "scope": "compliance",
        "ooc": {
            "equipmentCount": 0,
            "locationCount": 0,
            "trainingCount": 0,
            "total": 0,
            "equipment": [],
            "locations": [],
            "trainings": [],
        },
        "todayTasks": [],
        "futureTasks": [],
        "upcoming": {"equipment": [], "training": [], "locations": []},
        "personnelAlerts": [],
        "environmentalAlerts": [],
    }

    try:
        eq_c = _get_business_container("equipment")
        overdue_eq = list(
            eq_c.query_items(
                query="SELECT * FROM c WHERE c.status = 'OVERDUE' OR c.pm_status = 'OVERDUE'",
                enable_cross_partition_query=True,
            )
        )
        out["ooc"]["equipment"] = [str(e.get("name") or e.get("id")) for e in overdue_eq]
        out["ooc"]["equipmentCount"] = len(overdue_eq)
    except Exception as exc:
        logger.warning(f"Cosmos compliance equipment read error: {exc}")

    try:
        tr_c = _get_business_container("training")
        expired_tr = list(
            tr_c.query_items(
                query="SELECT * FROM c WHERE c.status = 'EXPIRED'",
                enable_cross_partition_query=True,
            )
        )
        out["ooc"]["trainings"] = [str(t.get("course_name") or t.get("operator_id")) for t in expired_tr]
        out["ooc"]["trainingCount"] = len(expired_tr)
    except Exception as exc:
        logger.warning(f"Cosmos compliance training read error: {exc}")

    out["ooc"]["total"] = out["ooc"]["equipmentCount"] + out["ooc"]["trainingCount"]
    return out


# ── TOOL REGISTRY & SAFE LLM FUNCTION DISPATCH ────────────────────────────────

PREDEFINED_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_batch_by_id": get_batch_by_id,
    "get_batches": get_batches,
    "get_operator_training_status": get_operator_training_status,
    "get_equipment_status": get_equipment_status,
    "get_material_lot_trace": get_material_lot_trace,
    "get_deviation_by_id": get_deviation_by_id,
    "get_environmental_monitoring": get_environmental_monitoring,
    "get_production_dashboard_data": get_production_dashboard_data,
    "get_compliance_dashboard_data": get_compliance_dashboard_data,
}

PREDEFINED_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_batch_by_id",
            "description": "Fetch a single batch record by its batch ID (e.g. B-1021).",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "The unique batch identifier (e.g. B-1021)"}
                },
                "required": ["batch_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_batches",
            "description": "Retrieve recent batch records, optionally filtered by status (e.g. IN_PROGRESS, RELEASED).",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Batch status to filter by"},
                    "limit": {"type": "integer", "description": "Maximum records to return (max 50)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_operator_training_status",
            "description": "Retrieve training and qualification records for an operator (e.g. OP-017).",
            "parameters": {
                "type": "object",
                "properties": {
                    "operator_id": {"type": "string", "description": "The operator ID (e.g. OP-017)"}
                },
                "required": ["operator_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_equipment_status",
            "description": "Retrieve PM, calibration, and status for equipment (e.g. EQ-102).",
            "parameters": {
                "type": "object",
                "properties": {
                    "equipment_id": {"type": "string", "description": "The equipment asset ID (e.g. EQ-102)"}
                },
                "required": ["equipment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_material_lot_trace",
            "description": "Trace a chemical or component lot (e.g. RM-88321) through batches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_number": {"type": "string", "description": "The lot or chemical code"}
                },
                "required": ["lot_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_deviation_by_id",
            "description": "Retrieve deviation and CAPA details by deviation ID (e.g. DEV-445).",
            "parameters": {
                "type": "object",
                "properties": {
                    "deviation_id": {"type": "string", "description": "The deviation ID"}
                },
                "required": ["deviation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_environmental_monitoring",
            "description": "Retrieve environmental monitoring excursions and readings for a location or room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_id": {"type": "string", "description": "Location or room identifier (e.g. R-204)"}
                },
            },
        },
    },
]


def execute_predefined_tool(
    function_name: str,
    arguments: dict[str, Any],
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Execute a predefined function requested via LLM function calling.
    Rejects any function not in the strict PREDEFINED_FUNCTIONS registry.
    The LLM has NO direct query-writing ability.
    """
    fn_name = (function_name or "").strip()
    if fn_name not in PREDEFINED_FUNCTIONS:
        raise ValueError(
            f"Unauthorized function '{fn_name}'. The LLM can only invoke predefined functions: "
            f"{list(PREDEFINED_FUNCTIONS.keys())}. Arbitrary database queries are strictly prohibited."
        )

    fn = PREDEFINED_FUNCTIONS[fn_name]
    # Pass permissions to the function for pre-query gating
    args = dict(arguments)
    args["permissions"] = permissions

    try:
        result = fn(**args)
        return {"function": fn_name, "arguments": arguments, "result": result, "status": "success"}
    except PermissionDeniedError as exc:
        return {
            "function": fn_name,
            "arguments": arguments,
            "error": exc.message,
            "status": "permission_denied",
        }
    except Exception as exc:
        logger.error(f"Error executing predefined function {fn_name}: {exc}", exc_info=True)
        return {"function": fn_name, "arguments": arguments, "error": str(exc), "status": "error"}

