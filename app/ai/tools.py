"""Business-level AI tools exposed to the CPG AI Assistant.

Every tool strictly routes its request through our dedicated backend domain endpoints
(app.api.domain_endpoints), which enforce:
1. Role / lens / module permission verification
2. Interaction with live CPG microservices with active JWT token
3. Response normalization and sanitization
"""

from __future__ import annotations

import logging
from typing import Any

from app.api import domain_endpoints
from app.api.domain_endpoints import (
    DetectTrendsRequest,
    RecommendBatchRequest,
    RiskScoreRequest,
    SimilarBatchesRequest,
)
from app.clients.base_client import set_current_user
from app.services.capabilities import ChatPermissions
from app.services.permissions_service import verify_resource_permission

logger = logging.getLogger(__name__)


# ── CORE BATCH & PRODUCTION TOOLS ─────────────────────────────────────────────

async def get_batch_by_id(
    batch_id: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Fetch a single batch record by ID or Lot Number via dedicated production endpoint."""
    verify_resource_permission(permissions, "batch")
    clean_id = (batch_id or "").strip()
    if not clean_id:
        return {"error": "batch_id parameter is required.", "status": "missing_parameter"}

    res = await domain_endpoints.get_production_batch(batch_id=clean_id, token=token, permissions=permissions)
    if res.status == "success" and res.data:
        return {
            "found": True,
            "batch": res.data,
            "record_id": clean_id,
            "status": "success",
            "endpoints_triggered": res.endpoints_triggered,
        }
    return {
        "found": False,
        "batch": None,
        "record_id": clean_id,
        "status": "not_found",
        "error": f"Batch '{clean_id}' was not found in the production database.",
        "endpoints_triggered": res.endpoints_triggered,
    }


async def get_batches(
    status: str | None = None,
    limit: int = 10,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve batch records via dedicated production endpoint."""
    verify_resource_permission(permissions, "batch")
    res = await domain_endpoints.list_production_batches(batch_status=status, limit=limit, token=token, permissions=permissions)
    return res.data if isinstance(res.data, list) else ([res.data] if res.data else [])


async def get_production_dashboard_data(
    permissions: ChatPermissions | None = None,
    token: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Fetch live production dashboard metrics via dedicated production endpoint."""
    if user_id:
        set_current_user(user_id=user_id)
    verify_resource_permission(permissions, "production")
    res = await domain_endpoints.get_production_dashboard(token=token, permissions=permissions)
    return res.data if isinstance(res.data, dict) else {"details": res.data}


# ── COMPLIANCE & QUALITY TOOLS ────────────────────────────────────────────────

async def get_equipment_status(
    equipment_id: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Retrieve equipment PM status and calibration details via dedicated compliance endpoint."""
    verify_resource_permission(permissions, "equipment")
    clean_id = (equipment_id or "").strip()
    if not clean_id:
        return {"error": "equipment_id parameter is required.", "status": "missing_parameter"}

    res = await domain_endpoints.get_compliance_equipment(equipment_id=clean_id, token=token, permissions=permissions)
    eq_data = res.data if isinstance(res.data, dict) else {"raw": res.data}
    return {
        "equipment": eq_data,
        "equipment_id": clean_id,
        "status": res.status,
        "endpoints_triggered": res.endpoints_triggered,
    }


async def get_operator_training_status(
    operator_id: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Retrieve operator qualification and training records via dedicated compliance endpoint."""
    verify_resource_permission(permissions, "training")
    clean_id = (operator_id or "").strip()
    if not clean_id:
        return {"error": "operator_id parameter is required.", "status": "missing_parameter"}

    res = await domain_endpoints.get_compliance_training(operator_id=clean_id, token=token, permissions=permissions)
    trainings = res.data if isinstance(res.data, list) else ([res.data] if res.data else [])
    return {
        "operator_id": clean_id,
        "training_records": trainings,
        "status": res.status,
        "endpoints_triggered": res.endpoints_triggered,
    }


async def get_deviation_by_id(
    deviation_id: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Retrieve deviation record details via dedicated compliance endpoint."""
    verify_resource_permission(permissions, "deviation")
    clean_id = (deviation_id or "").strip()
    if not clean_id:
        return {"error": "deviation_id parameter is required.", "status": "missing_parameter"}

    res = await domain_endpoints.get_compliance_deviation(deviation_id=clean_id, token=token, permissions=permissions)
    dev_data = res.data if isinstance(res.data, dict) else {"raw": res.data}
    return {
        "deviation": dev_data,
        "deviation_id": clean_id,
        "status": res.status,
        "endpoints_triggered": res.endpoints_triggered,
    }


async def get_environmental_monitoring(
    location_id: str | None = None,
    limit: int = 10,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Retrieve cleanroom environmental monitoring excursions via dedicated compliance endpoint."""
    verify_resource_permission(permissions, "environmental_monitoring")
    res = await domain_endpoints.get_compliance_environmental(location_id=location_id, limit=limit, token=token, permissions=permissions)
    records = res.data if isinstance(res.data, list) else ([res.data] if res.data else [])
    return {
        "records": records,
        "location_filter": location_id,
        "status": res.status,
        "endpoints_triggered": res.endpoints_triggered,
    }


async def get_compliance_dashboard_data(
    permissions: ChatPermissions | None = None,
    token: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Fetch live compliance dashboard metrics via dedicated compliance endpoint."""
    if user_id:
        set_current_user(user_id=user_id)
    verify_resource_permission(permissions, "compliance")
    res = await domain_endpoints.get_compliance_dashboard(token=token, permissions=permissions)
    return res.data if isinstance(res.data, dict) else {"details": res.data}


# ── GENEALOGY & TRACEABILITY TOOLS ────────────────────────────────────────────

async def get_material_lot_trace(
    lot_number: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Trace chemical or component lot through production via dedicated production endpoint."""
    verify_resource_permission(permissions, "material")
    clean_lot = (lot_number or "").strip()
    if not clean_lot:
        return {"error": "lot_number parameter is required.", "status": "missing_parameter"}

    res = await domain_endpoints.get_production_material_trace(lot_number=clean_lot, token=token, permissions=permissions)
    mat_data = res.data if isinstance(res.data, dict) else {"raw": res.data}
    return {
        "material_lot": mat_data,
        "lot_number": clean_lot,
        "status": res.status,
        "endpoints_triggered": res.endpoints_triggered,
    }


async def get_material_lot_genealogy(
    lot_id: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Forward genealogy: chemical lot -> batches -> finished drugs via dedicated genealogy endpoint."""
    verify_resource_permission(permissions, "material")
    verify_resource_permission(permissions, "batch")
    res = await domain_endpoints.get_material_genealogy(lot_id=lot_id, token=token, permissions=permissions)
    return res.data if isinstance(res.data, dict) else {"data": res.data}


async def get_component_lot_genealogy(
    lot_id: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Trace component/packaging lot through batches via dedicated genealogy endpoint."""
    verify_resource_permission(permissions, "material")
    verify_resource_permission(permissions, "batch")
    res = await domain_endpoints.get_material_genealogy(lot_id=lot_id, token=token, permissions=permissions)
    return res.data if isinstance(res.data, dict) else {"data": res.data}


async def get_finished_drug_genealogy(
    drug_id_or_lot: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Reverse genealogy: finished drug -> batch -> input lots via dedicated genealogy endpoint."""
    verify_resource_permission(permissions, "batch")
    res = await domain_endpoints.get_drug_genealogy(drug_id=drug_id_or_lot, token=token, permissions=permissions)
    return res.data if isinstance(res.data, dict) else {"data": res.data}


async def get_operator_batch_history(
    operator_id: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Operator batch execution and qualification history via dedicated genealogy endpoint."""
    verify_resource_permission(permissions, "training")
    verify_resource_permission(permissions, "batch")
    res = await domain_endpoints.get_operator_history(operator_id=operator_id, token=token, permissions=permissions)
    return res.data if isinstance(res.data, dict) else {"data": res.data}


async def get_equipment_batch_history(
    equipment_id: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Equipment asset execution and deviation history via dedicated genealogy endpoint."""
    verify_resource_permission(permissions, "equipment")
    verify_resource_permission(permissions, "batch")
    res = await domain_endpoints.get_equipment_history(equipment_id=equipment_id, token=token, permissions=permissions)
    return res.data if isinstance(res.data, dict) else {"data": res.data}


async def get_deviation_impact(
    deviation_id: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Assess cross-batch impact of a deviation via dedicated compliance endpoint."""
    verify_resource_permission(permissions, "deviation")
    verify_resource_permission(permissions, "batch")
    res = await domain_endpoints.get_compliance_deviation(deviation_id=deviation_id, token=token, permissions=permissions)
    dev_data = res.data if isinstance(res.data, dict) else {}
    return {
        "deviation_id": deviation_id,
        "deviation": dev_data,
        "impact_status": "assessed",
        "endpoints_triggered": res.endpoints_triggered,
    }


async def investigate_entity(
    entity_type: str,
    entity_id: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Deep investigative trace for any entity type routing through dedicated domain endpoints."""
    etype = (entity_type or "").strip().lower()
    eid = (entity_id or "").strip()

    if etype in {"batch", "batches"}:
        res = await domain_endpoints.get_production_batch(batch_id=eid, token=token, permissions=permissions)
        return {"entity_type": "batch", "details": res.data, "status": res.status}
    elif etype in {"equipment"}:
        res = await domain_endpoints.get_equipment_history(equipment_id=eid, token=token, permissions=permissions)
        return {"entity_type": "equipment", "details": res.data, "status": res.status}
    elif etype in {"operator", "training"}:
        res = await domain_endpoints.get_operator_history(operator_id=eid, token=token, permissions=permissions)
        return {"entity_type": "operator", "details": res.data, "status": res.status}
    elif etype in {"material", "lot"}:
        res = await domain_endpoints.get_material_genealogy(lot_id=eid, token=token, permissions=permissions)
        return {"entity_type": "material", "details": res.data, "status": res.status}
    elif etype in {"deviation"}:
        res = await domain_endpoints.get_compliance_deviation(deviation_id=eid, token=token, permissions=permissions)
        return {"entity_type": "deviation", "details": res.data, "status": res.status}
    else:
        res = await domain_endpoints.get_production_batch(batch_id=eid, token=token, permissions=permissions)
        return {"entity_type": etype, "details": res.data, "status": res.status}


async def get_related_batches(
    entity_type: str,
    entity_id: str,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Find all batches related to an entity type via dedicated production/genealogy endpoints."""
    verify_resource_permission(permissions, "batch")
    etype = (entity_type or "").strip().lower()
    eid = (entity_id or "").strip()

    if etype in {"operator", "training"}:
        res = await domain_endpoints.get_operator_history(operator_id=eid, token=token, permissions=permissions)
        batches = res.data.get("batches", []) if isinstance(res.data, dict) else []
        return {"entity_type": etype, "entity_id": eid, "batches": batches, "batches_count": len(batches)}
    elif etype in {"equipment"}:
        res = await domain_endpoints.get_equipment_history(equipment_id=eid, token=token, permissions=permissions)
        batches = res.data.get("batches", []) if isinstance(res.data, dict) else []
        return {"entity_type": etype, "entity_id": eid, "batches": batches, "batches_count": len(batches)}
    elif etype in {"material", "lot"}:
        res = await domain_endpoints.get_material_genealogy(lot_id=eid, token=token, permissions=permissions)
        batches = res.data.get("batches", []) if isinstance(res.data, dict) else []
        return {"entity_type": etype, "entity_id": eid, "batches": batches, "batches_count": len(batches)}

    res = await domain_endpoints.list_production_batches(token=token, permissions=permissions)
    all_b = res.data if isinstance(res.data, list) else []
    return {
        "entity_type": etype,
        "entity_id": eid,
        "batches": [b.get("id") or b.get("batch_number") for b in all_b],
        "batches_count": len(all_b),
    }


# ── ADVANCED ANALYTICS & RECOMMENDATION TOOLS ─────────────────────────────────

async def calculate_risk_score(
    batch_id: str | None = None,
    batch_id_or_config: str | dict[str, Any] | None = None,
    proposed_config: dict[str, Any] | None = None,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute risk score via dedicated analytics endpoint."""
    verify_resource_permission(permissions, "batch")
    target_id = batch_id or (batch_id_or_config if isinstance(batch_id_or_config, str) else None)
    target_config = proposed_config or (batch_id_or_config if isinstance(batch_id_or_config, dict) else None)
    body = RiskScoreRequest(batch_id=target_id, proposed_config=target_config)
    res = await domain_endpoints.calculate_risk(body=body, token=token, permissions=permissions)
    return res.data if isinstance(res.data, dict) else {"result": res.data}


async def find_similar_batches(
    batch_id: str,
    top_n: int = 5,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Find multi-dimensional similar batches via dedicated analytics endpoint."""
    verify_resource_permission(permissions, "batch")
    body = SimilarBatchesRequest(batch_id=batch_id, top_n=top_n)
    res = await domain_endpoints.find_similar(body=body, token=token, permissions=permissions)
    return res.data if isinstance(res.data, dict) else {"result": res.data}


async def detect_trends(
    product_id: str,
    metric: str = "yield",
    window: str = "90d",
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Detect quality drifts and yield anomalies via dedicated analytics endpoint."""
    verify_resource_permission(permissions, "batch")
    body = DetectTrendsRequest(product_id=product_id, metric=metric, window=window)
    res = await domain_endpoints.detect_analytics_trends(body=body, token=token, permissions=permissions)
    return res.data if isinstance(res.data, dict) else {"result": res.data}


async def recommend_batch_configuration(
    product_id: str,
    target_quantity: float = 1000.0,
    target_date: str | None = None,
    target_location: str | None = None,
    requested_overrides: dict[str, Any] | None = None,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
    user_id: str = "system-ai",
    role: str = "User",
    **kwargs: Any,
) -> dict[str, Any]:
    """GAMP-5 batch configuration recommendation via dedicated analytics endpoint."""
    verify_resource_permission(permissions, "batch")
    body = RecommendBatchRequest(
        product_id=product_id,
        target_quantity=target_quantity,
        target_date=target_date,
        target_location=target_location,
        requested_overrides=requested_overrides,
    )
    res = await domain_endpoints.recommend_batch(body=body, token=token, permissions=permissions)
    return res.data if isinstance(res.data, dict) else {"result": res.data}
