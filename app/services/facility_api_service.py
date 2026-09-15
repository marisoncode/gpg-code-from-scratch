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


# ── PHASE 2: MULTI-ENTITY INVESTIGATION & TRACEABILITY CHAINS ─────────────────


def get_related_batches(
    entity_type: str,
    entity_id: str,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Find all batches related to any entity type (operator, equipment, material, component, location, deviation, product).
    Enforces permission check BEFORE query.
    """
    etype = (entity_type or "").strip().lower()
    eid = (entity_id or "").strip()

    # Pre-query permission gating
    verify_resource_permission(permissions, "batch")
    if etype in {"training", "operator"}:
        verify_resource_permission(permissions, "training")
    elif etype in {"equipment"}:
        verify_resource_permission(permissions, "equipment")
    elif etype in {"deviation", "oos", "oot", "location", "em"}:
        verify_resource_permission(permissions, "compliance")

    if not eid:
        return {"error": "entity_id parameter is required.", "status": "missing_parameter"}

    try:
        container = _get_business_container("batches")
        query = "SELECT * FROM c WHERE ARRAY_CONTAINS(c.operators, @eid) OR ARRAY_CONTAINS(c.equipment, @eid) OR ARRAY_CONTAINS(c.materials, @eid) OR ARRAY_CONTAINS(c.components, @eid) OR c.product = @eid OR c.product_id = @eid OR c.location_id = @eid OR c.room = @eid OR c.operator_id = @eid OR c.equipment_id = @eid OR ARRAY_CONTAINS(c.deviations, @eid) OR c.id = @eid"
        parameters = [{"name": "@eid", "value": eid}]

        batches = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
        return {
            "entity_type": etype,
            "entity_id": eid,
            "count": len(batches),
            "related_batches": batches,
            "traceability_chain": f"{etype} -> batches",
            "status": "found" if batches else "unavailable",
        }
    except FacilitySecurityError:
        raise
    except Exception as exc:
        logger.warning(f"Cosmos error in get_related_batches for {etype}:{eid}: {exc}")
        return {
            "entity_type": etype,
            "entity_id": eid,
            "count": 0,
            "related_batches": [],
            "error": f"Related batches for {etype} {eid} are unavailable.",
            "status": "unavailable",
        }


def get_material_lot_genealogy(
    lot_id: str,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Trace chemical/raw material lot genealogy:
    material_lot -> batches -> finished_drugs.
    Strictly distinguishes [CONFIRMED IMPACT] from [POTENTIAL IMPACT].
    """
    verify_resource_permission(permissions, "material")
    verify_resource_permission(permissions, "batch")
    clean_lot = (lot_id or "").strip()
    if not clean_lot:
        return {"error": "lot_id parameter is required.", "status": "missing_parameter"}

    try:
        # Step 1: Fetch material lot record
        mat_container = _get_business_container("materials")
        mat_items = list(
            mat_container.query_items(
                query="SELECT * FROM c WHERE c.lot_number = @lot OR c.id = @lot",
                parameters=[{"name": "@lot", "value": clean_lot}],
                enable_cross_partition_query=True,
            )
        )
        lot_record = mat_items[0] if mat_items else {"lot_number": clean_lot, "status": "RECORD_NOT_FOUND"}

        # Step 2: Fetch all batches that used this material lot
        batch_container = _get_business_container("batches")
        batches = list(
            batch_container.query_items(
                query="SELECT * FROM c WHERE ARRAY_CONTAINS(c.materials, @lot) OR ARRAY_CONTAINS(c.material_lots, @lot) OR c.lot_number = @lot",
                parameters=[{"name": "@lot", "value": clean_lot}],
                enable_cross_partition_query=True,
            )
        )

        # Step 3: Classify into confirmed vs potential impact
        confirmed_impact: list[dict[str, Any]] = []
        potential_impact: list[dict[str, Any]] = []
        finished_drugs: list[str] = []

        is_defective = str(lot_record.get("quality_status") or lot_record.get("status") or "").upper() in {"REJECTED", "DEFECTIVE", "OOS", "RECALLED"}

        for b in batches:
            bid = str(b.get("id") or b.get("batch_number") or "")
            b_status = str(b.get("status") or "").upper()
            drugs = b.get("finished_drugs") or []
            if isinstance(drugs, list):
                finished_drugs.extend(str(d) for d in drugs if d)

            # Confirmed impact: direct verified consumption of confirmed bad lot or batch had linked OOS/deviation
            has_quality_event = bool(b.get("deviations") or b.get("oos_events") or b_status in {"FAILED", "REJECTED"})
            if is_defective or has_quality_event:
                confirmed_impact.append({
                    "batch_id": bid,
                    "product": b.get("product"),
                    "status": b_status,
                    "reason": "Direct verified consumption of defective material lot" if is_defective else "Batch has linked quality event / deviation",
                    "impact_type": "CONFIRMED",
                })
            else:
                # Potential impact: used same lot but not confirmed defective or currently in progress
                potential_impact.append({
                    "batch_id": bid,
                    "product": b.get("product"),
                    "status": b_status,
                    "reason": "Batch used lot; downstream impact unconfirmed pending QA disposition",
                    "impact_type": "POTENTIAL",
                })

        return {
            "lot_id": clean_lot,
            "material_type": "raw_material",
            "lot_details": lot_record,
            "batches_count": len(batches),
            "batches": [b.get("id") or b.get("batch_number") for b in batches],
            "finished_drugs": list(set(finished_drugs)),
            "confirmed_impact": confirmed_impact,
            "potential_impact": potential_impact,
            "traceability_chain": "material_lot -> batches -> finished_drugs",
            "status": "found" if batches or mat_items else "unavailable",
        }
    except FacilitySecurityError:
        raise
    except Exception as exc:
        logger.warning(f"Cosmos error in get_material_lot_genealogy for {clean_lot}: {exc}")
        return {
            "lot_id": clean_lot,
            "error": f"Genealogy data for material lot {clean_lot} is unavailable.",
            "status": "unavailable",
            "confirmed_impact": [],
            "potential_impact": [],
            "traceability_chain": "material_lot -> batches -> finished_drugs",
        }


def get_component_lot_genealogy(
    lot_id: str,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Trace packaging / component lot genealogy:
    component_lot -> batches -> finished_drugs.
    Strictly distinguishes [CONFIRMED IMPACT] from [POTENTIAL IMPACT].
    """
    verify_resource_permission(permissions, "component")
    verify_resource_permission(permissions, "batch")
    clean_lot = (lot_id or "").strip()
    if not clean_lot:
        return {"error": "lot_id parameter is required.", "status": "missing_parameter"}

    try:
        batch_container = _get_business_container("batches")
        batches = list(
            batch_container.query_items(
                query="SELECT * FROM c WHERE ARRAY_CONTAINS(c.components, @lot) OR ARRAY_CONTAINS(c.component_lots, @lot)",
                parameters=[{"name": "@lot", "value": clean_lot}],
                enable_cross_partition_query=True,
            )
        )

        confirmed_impact: list[dict[str, Any]] = []
        potential_impact: list[dict[str, Any]] = []
        finished_drugs: list[str] = []

        for b in batches:
            bid = str(b.get("id") or b.get("batch_number") or "")
            drugs = b.get("finished_drugs") or []
            if isinstance(drugs, list):
                finished_drugs.extend(str(d) for d in drugs if d)

            if b.get("status") in {"REJECTED", "FAILED"}:
                confirmed_impact.append({"batch_id": bid, "impact_type": "CONFIRMED", "reason": "Batch rejected"})
            else:
                potential_impact.append({"batch_id": bid, "impact_type": "POTENTIAL", "reason": "Component lot used; inspection pending"})

        return {
            "lot_id": clean_lot,
            "component_type": "packaging_component",
            "batches_count": len(batches),
            "batches": [b.get("id") or b.get("batch_number") for b in batches],
            "finished_drugs": list(set(finished_drugs)),
            "confirmed_impact": confirmed_impact,
            "potential_impact": potential_impact,
            "traceability_chain": "component_lot -> batches -> finished_drugs",
            "status": "found" if batches else "unavailable",
        }
    except FacilitySecurityError:
        raise
    except Exception as exc:
        logger.warning(f"Cosmos error in get_component_lot_genealogy for {clean_lot}: {exc}")
        return {
            "lot_id": clean_lot,
            "error": f"Genealogy for component lot {clean_lot} is unavailable.",
            "status": "unavailable",
            "confirmed_impact": [],
            "potential_impact": [],
            "traceability_chain": "component_lot -> batches -> finished_drugs",
        }


def get_operator_batch_history(
    operator_id: str,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Retrieve operator history: batches executed, equipment handled,
    training qualification status as of batch date, and linked quality events.
    """
    verify_resource_permission(permissions, "training")
    verify_resource_permission(permissions, "batch")
    clean_op = (operator_id or "").strip()
    if not clean_op:
        return {"error": "operator_id parameter is required.", "status": "missing_parameter"}

    try:
        # Step 1: Operator training records
        tr_container = _get_business_container("training")
        trainings = list(
            tr_container.query_items(
                query="SELECT * FROM c WHERE c.operator_id = @op_id OR c.id = @op_id",
                parameters=[{"name": "@op_id", "value": clean_op}],
                enable_cross_partition_query=True,
            )
        )

        # Step 2: Batches participated in
        batch_container = _get_business_container("batches")
        batches = list(
            batch_container.query_items(
                query="SELECT * FROM c WHERE ARRAY_CONTAINS(c.operators, @op_id) OR c.operator_id = @op_id",
                parameters=[{"name": "@op_id", "value": clean_op}],
                enable_cross_partition_query=True,
            )
        )

        equipment_used: set[str] = set()
        deviations_linked: list[str] = []
        for b in batches:
            for eq in b.get("equipment") or []:
                if eq:
                    equipment_used.add(str(eq))
            for dev in b.get("deviations") or []:
                if dev:
                    deviations_linked.append(str(dev))

        return {
            "operator_id": clean_op,
            "batches_count": len(batches),
            "batches": [b.get("id") or b.get("batch_number") for b in batches],
            "equipment_handled": sorted(equipment_used),
            "deviations_linked": deviations_linked,
            "qualifications": trainings,
            "traceability_chain": "operator -> batches -> equipment -> quality_events",
            "status": "found" if batches or trainings else "unavailable",
        }
    except FacilitySecurityError:
        raise
    except Exception as exc:
        logger.warning(f"Cosmos error in get_operator_batch_history for {clean_op}: {exc}")
        return {
            "operator_id": clean_op,
            "error": f"Operator batch history for {clean_op} is unavailable.",
            "status": "unavailable",
            "traceability_chain": "operator -> batches",
        }


def get_equipment_batch_history(
    equipment_id: str,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Retrieve equipment history: batches processed on asset, PM/calibration status as of date,
    and deviations linked to asset.
    """
    verify_resource_permission(permissions, "equipment")
    verify_resource_permission(permissions, "batch")
    clean_eq = (equipment_id or "").strip()
    if not clean_eq:
        return {"error": "equipment_id parameter is required.", "status": "missing_parameter"}

    try:
        # Step 1: Equipment asset details
        eq_container = _get_business_container("equipment")
        eq_items = list(
            eq_container.query_items(
                query="SELECT * FROM c WHERE c.id = @eq_id OR c.asset_id = @eq_id",
                parameters=[{"name": "@eq_id", "value": clean_eq}],
                enable_cross_partition_query=True,
            )
        )
        eq_record = eq_items[0] if eq_items else {"asset_id": clean_eq, "status": "UNKNOWN"}

        # Step 2: Batches processed on this equipment
        batch_container = _get_business_container("batches")
        batches = list(
            batch_container.query_items(
                query="SELECT * FROM c WHERE ARRAY_CONTAINS(c.equipment, @eq_id) OR c.equipment_id = @eq_id",
                parameters=[{"name": "@eq_id", "value": clean_eq}],
                enable_cross_partition_query=True,
            )
        )

        # Distinguish confirmed vs potential impact from equipment status
        pm_overdue = str(eq_record.get("pm_status") or eq_record.get("status") or "").upper() == "OVERDUE"
        confirmed_impact: list[dict[str, Any]] = []
        potential_impact: list[dict[str, Any]] = []

        for b in batches:
            bid = str(b.get("id") or b.get("batch_number") or "")
            if b.get("deviations"):
                confirmed_impact.append({
                    "batch_id": bid,
                    "impact_type": "CONFIRMED",
                    "reason": "Batch has confirmed deviation linked to equipment event",
                })
            elif pm_overdue:
                potential_impact.append({
                    "batch_id": bid,
                    "impact_type": "POTENTIAL",
                    "reason": "Batch processed on equipment with overdue PM/calibration; requires QA review",
                })
            else:
                potential_impact.append({
                    "batch_id": bid,
                    "impact_type": "POTENTIAL",
                    "reason": "Batch processed on equipment; within standard parameters",
                })

        return {
            "equipment_id": clean_eq,
            "asset_details": eq_record,
            "batches_count": len(batches),
            "batches": [b.get("id") or b.get("batch_number") for b in batches],
            "confirmed_impact": confirmed_impact,
            "potential_impact": potential_impact,
            "traceability_chain": "equipment -> batches -> deviations",
            "status": "found" if batches or eq_items else "unavailable",
        }
    except FacilitySecurityError:
        raise
    except Exception as exc:
        logger.warning(f"Cosmos error in get_equipment_batch_history for {clean_eq}: {exc}")
        return {
            "equipment_id": clean_eq,
            "error": f"Equipment batch history for {clean_eq} is unavailable.",
            "status": "unavailable",
            "traceability_chain": "equipment -> batches",
        }


def get_finished_drug_genealogy(
    drug_id_or_lot: str,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Reverse traceability chain:
    finished_drug -> batch -> input_lots (raw materials + components).
    """
    verify_resource_permission(permissions, "batch")
    verify_resource_permission(permissions, "material")
    clean_id = (drug_id_or_lot or "").strip()
    if not clean_id:
        return {"error": "drug_id_or_lot parameter is required.", "status": "missing_parameter"}

    try:
        batch_container = _get_business_container("batches")
        batches = list(
            batch_container.query_items(
                query="SELECT * FROM c WHERE ARRAY_CONTAINS(c.finished_drugs, @did) OR c.finished_drug_lot = @did OR c.id = @did",
                parameters=[{"name": "@did", "value": clean_id}],
                enable_cross_partition_query=True,
            )
        )

        all_materials: set[str] = set()
        all_components: set[str] = set()
        for b in batches:
            for m in b.get("materials") or b.get("material_lots") or []:
                if m:
                    all_materials.add(str(m))
            for c in b.get("components") or b.get("component_lots") or []:
                if c:
                    all_components.add(str(c))

        return {
            "finished_drug_id": clean_id,
            "batches": [b.get("id") or b.get("batch_number") for b in batches],
            "input_material_lots": sorted(all_materials),
            "input_component_lots": sorted(all_components),
            "traceability_chain": "finished_drug -> batch -> input_lots",
            "status": "found" if batches else "unavailable",
        }
    except FacilitySecurityError:
        raise
    except Exception as exc:
        logger.warning(f"Cosmos error in get_finished_drug_genealogy for {clean_id}: {exc}")
        return {
            "finished_drug_id": clean_id,
            "error": f"Genealogy for finished drug {clean_id} is unavailable.",
            "status": "unavailable",
            "traceability_chain": "finished_drug -> batch -> input_lots",
        }


def get_deviation_impact(
    deviation_id: str,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Forward impact analysis for deviation / OOS event:
    deviation -> directly_affected_batch (confirmed) + co-manufactured_batches (potential).
    """
    verify_resource_permission(permissions, "deviation")
    clean_id = (deviation_id or "").strip()
    if not clean_id:
        return {"error": "deviation_id parameter is required.", "status": "missing_parameter"}

    try:
        dev_container = _get_business_container("deviations")
        dev_items = list(
            dev_container.query_items(
                query="SELECT * FROM c WHERE c.id = @did OR c.deviation_number = @did",
                parameters=[{"name": "@did", "value": clean_id}],
                enable_cross_partition_query=True,
            )
        )
        dev_record = dev_items[0] if dev_items else {"id": clean_id, "status": "RECORD_NOT_FOUND"}

        batch_container = _get_business_container("batches")
        batches = list(
            batch_container.query_items(
                query="SELECT * FROM c WHERE ARRAY_CONTAINS(c.deviations, @did) OR c.deviation_id = @did",
                parameters=[{"name": "@did", "value": clean_id}],
                enable_cross_partition_query=True,
            )
        )

        confirmed_impact: list[dict[str, Any]] = []
        potential_impact: list[dict[str, Any]] = []

        for b in batches:
            bid = str(b.get("id") or b.get("batch_number") or "")
            confirmed_impact.append({
                "batch_id": bid,
                "impact_type": "CONFIRMED",
                "reason": f"Directly cited in deviation record {clean_id}",
            })

        # Co-manufactured or shared line batches are potential impact
        eq_id = dev_record.get("equipment_id")
        if eq_id:
            shared_eq_batches = list(
                batch_container.query_items(
                    query="SELECT TOP 5 * FROM c WHERE ARRAY_CONTAINS(c.equipment, @eq) AND NOT ARRAY_CONTAINS(c.deviations, @did)",
                    parameters=[{"name": "@eq", "value": eq_id}, {"name": "@did", "value": clean_id}],
                    enable_cross_partition_query=True,
                )
            )
            for b in shared_eq_batches:
                bid = str(b.get("id") or b.get("batch_number") or "")
                potential_impact.append({
                    "batch_id": bid,
                    "impact_type": "POTENTIAL",
                    "reason": f"Manufactured on shared equipment {eq_id} during relevant window",
                })

        return {
            "deviation_id": clean_id,
            "deviation_details": dev_record,
            "confirmed_impact": confirmed_impact,
            "potential_impact": potential_impact,
            "traceability_chain": "deviation -> affected_batches",
            "status": "found" if dev_items or batches else "unavailable",
        }
    except FacilitySecurityError:
        raise
    except Exception as exc:
        logger.warning(f"Cosmos error in get_deviation_impact for {clean_id}: {exc}")
        return {
            "deviation_id": clean_id,
            "error": f"Deviation impact data for {clean_id} is unavailable.",
            "status": "unavailable",
            "confirmed_impact": [],
            "potential_impact": [],
            "traceability_chain": "deviation -> affected_batches",
        }


def investigate_entity(
    entity_type: str,
    entity_id: str,
    date_range: str | None = None,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    """
    Universal multi-entity investigation entry point supporting all 11 SRS entity types:
    batch, product, material_lot, component, operator, equipment, location, em, pm, deviation, oos.
    """
    etype = (entity_type or "").strip().lower()
    eid = (entity_id or "").strip()

    if etype in {"batch"}:
        return get_batch_by_id(batch_id=eid, permissions=permissions)
    if etype in {"material", "material_lot", "chemical", "chemical_lot"}:
        return get_material_lot_genealogy(lot_id=eid, permissions=permissions)
    if etype in {"component", "component_lot", "packaging"}:
        return get_component_lot_genealogy(lot_id=eid, permissions=permissions)
    if etype in {"operator"}:
        return get_operator_batch_history(operator_id=eid, permissions=permissions)
    if etype in {"equipment", "asset"}:
        return get_equipment_batch_history(equipment_id=eid, permissions=permissions)
    if etype in {"finished_drug", "drug"}:
        return get_finished_drug_genealogy(drug_id_or_lot=eid, permissions=permissions)
    if etype in {"deviation", "oos", "oot"}:
        return get_deviation_impact(deviation_id=eid, permissions=permissions)
    if etype in {"location", "room", "em", "environmental_monitoring"}:
        verify_resource_permission(permissions, "compliance")
        em_records = get_environmental_monitoring(location_id=eid, permissions=permissions)
        related = get_related_batches(entity_type="location", entity_id=eid, permissions=permissions)
        return {
            "entity_type": "location",
            "entity_id": eid,
            "environmental_monitoring": em_records,
            "related_batches": related.get("related_batches", []),
            "traceability_chain": "location -> em -> batches",
            "status": "found",
        }
    if etype in {"product"}:
        verify_resource_permission(permissions, "batch")
        related = get_related_batches(entity_type="product", entity_id=eid, permissions=permissions)
        return {
            "entity_type": "product",
            "entity_id": eid,
            "batches": related.get("related_batches", []),
            "traceability_chain": "product -> batches",
            "status": "found",
        }

    # Fallback to generic related batches search
    return get_related_batches(entity_type=etype, entity_id=eid, permissions=permissions)



def find_similar_batches(
    batch_id: str,
    top_n: int = 5,
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    from app.services.analytics_service import find_similar_batches as _fsb
    return _fsb(batch_id=batch_id, top_n=top_n, permissions=permissions)


def calculate_risk_score(
    batch_id: str = "",
    proposed_config: dict[str, Any] | None = None,
    permissions: ChatPermissions | None = None,
    weight_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from app.services.analytics_service import calculate_risk_score as _crs
    target = proposed_config if proposed_config else batch_id
    return _crs(batch_id_or_config=target, permissions=permissions, weight_config=weight_config)


def detect_trends(
    product_id: str,
    metric: str = "yield",
    window: str = "90d",
    permissions: ChatPermissions | None = None,
) -> dict[str, Any]:
    from app.services.analytics_service import detect_trends as _dt
    return _dt(product_id=product_id, metric=metric, window=window, permissions=permissions)


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
    # Phase 2 functions:
    "get_related_batches": get_related_batches,
    "get_material_lot_genealogy": get_material_lot_genealogy,
    "get_component_lot_genealogy": get_component_lot_genealogy,
    "get_operator_batch_history": get_operator_batch_history,
    "get_equipment_batch_history": get_equipment_batch_history,
    "get_finished_drug_genealogy": get_finished_drug_genealogy,
    "get_deviation_impact": get_deviation_impact,
    "investigate_entity": investigate_entity,
    # Phase 3 functions:
    "find_similar_batches": find_similar_batches,
    "calculate_risk_score": calculate_risk_score,
    "detect_trends": detect_trends,
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
    {
        "type": "function",
        "function": {
            "name": "investigate_entity",
            "description": "Start a multi-entity investigation from ANY entity type (batch, product, material lot, component, operator, equipment, location, EM, PM, deviation, OOS, date range).",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "description": "Type of starting entity: 'batch', 'product', 'material_lot', 'component', 'operator', 'equipment', 'location', 'em', 'pm', 'deviation', 'oos'",
                    },
                    "entity_id": {"type": "string", "description": "The identifier of the starting entity"},
                    "date_range": {"type": "string", "description": "Optional date range filter (e.g. '2026-01-01 to 2026-08-31')"},
                },
                "required": ["entity_type", "entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_related_batches",
            "description": "Find all batches related to a specific entity (operator, equipment, material, component, location, deviation, product).",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string", "description": "The entity type (e.g. operator, equipment, material, location)"},
                    "entity_id": {"type": "string", "description": "The entity identifier"},
                },
                "required": ["entity_type", "entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_material_lot_genealogy",
            "description": "Traverse forward genealogy from chemical/material lot -> batches -> finished drugs, labeling confirmed vs potential impact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {"type": "string", "description": "The material lot number (e.g. RM-88321)"}
                },
                "required": ["lot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_component_lot_genealogy",
            "description": "Traverse packaging component genealogy: component_lot -> batches -> finished drugs, labeling confirmed vs potential impact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {"type": "string", "description": "The component lot number (e.g. COMP-501)"}
                },
                "required": ["lot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_operator_batch_history",
            "description": "Retrieve batches executed by an operator, equipment handled, date-aware qualifications, and deviations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operator_id": {"type": "string", "description": "The operator identifier (e.g. OP-017)"}
                },
                "required": ["operator_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_equipment_batch_history",
            "description": "Retrieve batches processed on equipment, PM/calibration history, and deviations, labeling confirmed vs potential impact.",
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
            "name": "get_finished_drug_genealogy",
            "description": "Reverse traceability chain: finished_drug -> batch -> input_lots (materials + components).",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_id_or_lot": {"type": "string", "description": "The finished drug identifier or lot number"}
                },
                "required": ["drug_id_or_lot"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_deviation_impact",
            "description": "Evaluate deviation/OOS impact: deviation -> directly affected batches (confirmed) + co-manufactured batches (potential).",
            "parameters": {
                "type": "object",
                "properties": {
                    "deviation_id": {"type": "string", "description": "The deviation ID (e.g. DEV-445)"}
                },
                "required": ["deviation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_batches",
            "description": "Find historically similar batches comparing product, formulation, size, materials, operators, equipment, and location, returning similarity percentage and top explaining factors (SRS FR-006).",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "Target batch identifier to compare against historical batches"},
                    "top_n": {"type": "integer", "description": "Number of top similar batches to return (default 5)"},
                },
                "required": ["batch_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_risk_score",
            "description": "Calculate a batch risk score (0-100) and risk level using approved, configurable risk weights (SRS Section 19).",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "Batch ID to evaluate for risk"},
                    "proposed_config": {"type": "object", "description": "Optional proposed batch run configuration parameters"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_trends",
            "description": "Detect statistical trends and anomalies (yield drift, recurring EM events, equipment deviations). Enforces non-causal reporting per SRS Section 13.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "The product identifier to analyze"},
                    "metric": {"type": "string", "description": "Metric to analyze (yield, em_events, equipment_deviations)"},
                    "window": {"type": "string", "description": "Time window (e.g. '90d', '6m', '1y')"},
                },
                "required": ["product_id"],
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

