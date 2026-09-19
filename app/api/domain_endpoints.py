"""Dedicated domain REST endpoints for Production, Compliance, Genealogy, Analytics, and Tools.

Provides direct, typed REST endpoints for frontend UI and external services:
- Production: /v1/production/*
- Compliance: /v1/compliance/*
- Genealogy: /v1/genealogy/*
- Analytics: /v1/analytics/*
- Tools Catalog & Execution: /v1/tools/*

Directly communicates with CPG downstream microservice clients:
- ProductionClient -> https://production-api.cpguardian.com/api/
- ComplianceClient -> https://compliance-api.cpguardian.com/api/
- FacilityClient   -> https://facility-user-api.cpguardian.com/api/

Every endpoint strictly enforces:
1. Authentication via require_facility_user dependency or active JWT context
2. Authoritative permissions resolution via CPG ManageUser API
3. Application-level Python permission checks (verify_resource_permission)
4. Downstream response data sanitization (stripping secrets and internal metadata)
5. Microservice execution transparency (endpoints_triggered)
"""

from __future__ import annotations

import inspect
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.ai.sanitizer import sanitize_for_llm
from app.clients.base_client import (
    get_current_token,
    get_current_user_id,
    get_current_user_name,
    get_triggered_endpoints,
    normalize_cpg_record,
    reset_triggered_endpoints,
    set_current_token,
    set_current_user,
)
from app.clients.compliance_client import ComplianceClient
from app.clients.facility_client import FacilityClient
from app.clients.production_client import ProductionClient
from app.core.auth import AuthenticatedUser, require_facility_user
from app.services import analytics_service, recommendation_engine
from app.services.capabilities import ChatPermissions
from app.services.permissions_service import (
    PermissionDeniedError,
    resolve_permissions,
    verify_resource_permission,
)

# ── DOWNSTREAM CPG CLIENTS ───────────────────────────────────────────────────

_prod_client = ProductionClient()
_comp_client = ComplianceClient()
_fac_client = FacilityClient()

# ── ROUTERS ───────────────────────────────────────────────────────────────────

production_router = APIRouter(prefix="/v1/production", tags=["Production"])
compliance_router = APIRouter(prefix="/v1/compliance", tags=["Compliance"])
genealogy_router = APIRouter(prefix="/v1/genealogy", tags=["Genealogy & Traceability"])
analytics_router = APIRouter(prefix="/v1/analytics", tags=["Analytics & Recommendations"])
tools_router = APIRouter(prefix="/v1/tools", tags=["AI Tools & Capabilities"])
facility_router = APIRouter(prefix="/v1/facility", tags=["Facility & Roles"])


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────

class DomainResponse(BaseModel):
    status: str = "success"
    data: Any
    endpoints_triggered: list[str] = Field(default_factory=list)


class RiskScoreRequest(BaseModel):
    batch_id: str | None = None
    proposed_config: dict[str, Any] | None = None


class SimilarBatchesRequest(BaseModel):
    batch_id: str
    top_n: int = Field(default=5, ge=1, le=20)


class DetectTrendsRequest(BaseModel):
    product_id: str
    metric: str = "yield"
    window: str = "90d"


class RecommendBatchRequest(BaseModel):
    product_id: str
    target_quantity: float = 1000.0
    target_date: str | None = None
    target_location: str | None = None
    requested_overrides: dict[str, Any] | None = None


class ToolExecuteRequest(BaseModel):
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


def _resolve_caller(
    user: AuthenticatedUser | Any = None,
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> AuthenticatedUser:
    """Resolves authenticated context whether called via FastAPI HTTP or programmatically from AI tools."""
    if isinstance(user, AuthenticatedUser):
        return user

    tok = (token or get_current_token()).strip()
    uid = get_current_user_id()
    uname = get_current_user_name()

    col = ""
    if tok and (not uid or not uname):
        try:
            from app.core.auth import decode_facility_token
            decoded = decode_facility_token(tok)
            if not uid:
                uid = decoded.user_id
            if not uname:
                uname = decoded.name
            if decoded.collection_id:
                col = decoded.collection_id
        except Exception:
            pass

    return AuthenticatedUser(
        user_id=uid or "system-ai",
        name=uname or "AI Assistant",
        collection_id=col or (user.collection_id if isinstance(user, AuthenticatedUser) else "") or get_current_collection_id(),
        raw_claims={},
        token=tok,
    )


# ── PRODUCTION ENDPOINTS (DIRECT CPG CALLS) ───────────────────────────────────

@production_router.get("/batches", response_model=DomainResponse)
async def list_production_batches(
    batch_status: str | None = Query(None, alias="status", description="Optional batch status filter"),
    limit: int = Query(10, ge=1, le=50, description="Max records to return"),
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Directly query batch records from Production API."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "batch")
        code, raw = await _prod_client.get_batches(status=batch_status, limit=limit, token=actual_user.token)
        items = [normalize_cpg_record(x) for x in raw] if isinstance(raw, list) else normalize_cpg_record(raw)
        return DomainResponse(
            status="success" if code == 200 else "partial",
            data=sanitize_for_llm(items),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@production_router.get("/batches/{batch_id}", response_model=DomainResponse)
async def get_production_batch(
    batch_id: str,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Directly fetch batch record from Production API."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "batch")
        code, raw = await _prod_client.get_batch_by_id(batch_id=batch_id, token=actual_user.token)
        if code == 200 and raw:
            batch = normalize_cpg_record(raw)
            return DomainResponse(
                status="success",
                data=sanitize_for_llm(batch),
                endpoints_triggered=get_triggered_endpoints(),
            )
        return DomainResponse(
            status="not_found",
            data=None,
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@production_router.get("/materials/{lot_number}", response_model=DomainResponse)
async def get_production_material_trace(
    lot_number: str,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Directly query chemical/component child inventory from Production API."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "material")
        code, raw = await _prod_client.get_material_inventory(lot_number=lot_number, token=actual_user.token)
        return DomainResponse(
            status="success" if code == 200 else "not_found",
            data=sanitize_for_llm(normalize_cpg_record(raw)),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@production_router.get("/dashboard", response_model=DomainResponse)
async def get_production_dashboard(
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Directly retrieve production dashboard details from Facility API."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "production")
        code, raw = await _fac_client.get_production_dashboard(token=actual_user.token)
        return DomainResponse(
            status="success" if code == 200 else "error",
            data=sanitize_for_llm(raw),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


# ── COMPLIANCE ENDPOINTS (DIRECT CPG CALLS) ───────────────────────────────────

@compliance_router.get("/equipment/{equipment_id}", response_model=DomainResponse)
async def get_compliance_equipment(
    equipment_id: str,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Directly retrieve equipment PM and calibration from Compliance API."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "equipment")
        code, raw = await _comp_client.get_equipment(equipment_id=equipment_id, token=actual_user.token)
        return DomainResponse(
            status="success" if code == 200 else "not_found",
            data=sanitize_for_llm(normalize_cpg_record(raw)),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@compliance_router.get("/training/{operator_id}", response_model=DomainResponse)
async def get_compliance_training(
    operator_id: str,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Directly retrieve operator qualifications from Compliance API."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "training")
        code, raw = await _comp_client.get_training_by_operator(operator_id=operator_id, token=actual_user.token)
        return DomainResponse(
            status="success" if code == 200 else "not_found",
            data=sanitize_for_llm(normalize_cpg_record(raw)),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@compliance_router.get("/deviations/{deviation_id}", response_model=DomainResponse)
async def get_compliance_deviation(
    deviation_id: str,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Directly retrieve deviation details from TaskManagement API."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "deviation")
        code, raw = await _comp_client.get_deviation(deviation_id=deviation_id, token=actual_user.token)
        return DomainResponse(
            status="success" if code == 200 else "not_found",
            data=sanitize_for_llm(normalize_cpg_record(raw)),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@compliance_router.get("/environmental", response_model=DomainResponse)
async def get_compliance_environmental(
    location_id: str | None = Query(None, description="Optional cleanroom/room filter"),
    limit: int = Query(10, ge=1, le=50),
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Directly retrieve environmental monitoring excursions from Compliance API."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "environmental_monitoring")
        code, raw = await _comp_client.get_environmental_monitoring(location_id=location_id, limit=limit, token=actual_user.token)
        items = [normalize_cpg_record(x) for x in raw] if isinstance(raw, list) else normalize_cpg_record(raw)
        return DomainResponse(
            status="success" if code == 200 else "partial",
            data=sanitize_for_llm(items),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@compliance_router.get("/dashboard", response_model=DomainResponse)
async def get_compliance_dashboard(
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Directly retrieve compliance OOC dashboard details from Facility API."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "compliance")
        code, raw = await _fac_client.get_compliance_dashboard(token=actual_user.token)
        return DomainResponse(
            status="success" if code == 200 else "error",
            data=sanitize_for_llm(raw),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


# ── GENEALOGY ENDPOINTS ───────────────────────────────────────────────────────

@genealogy_router.get("/material/{lot_id}", response_model=DomainResponse)
async def get_material_genealogy(
    lot_id: str,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Trace material lot through batches and finished drugs."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "material")
        verify_resource_permission(permissions, "batch")
        code_mat, mat_data = await _prod_client.get_material_inventory(lot_id, token=actual_user.token)
        code_b, batches_data = await _prod_client.get_batches(limit=50, token=actual_user.token)
        all_batches = batches_data if isinstance(batches_data, list) else []
        matching = [
            normalize_cpg_record(b)
            for b in all_batches
            if lot_id in (str(b.get("material_lot") or b.get("materials") or ""))
        ]
        return DomainResponse(
            status="success",
            data=sanitize_for_llm({
                "lot_id": lot_id,
                "material": normalize_cpg_record(mat_data),
                "batches": matching,
                "batches_count": len(matching),
            }),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@genealogy_router.get("/drug/{drug_id}", response_model=DomainResponse)
async def get_drug_genealogy(
    drug_id: str,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Reverse trace finished drug to batches and consumed lots."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "batch")
        code, raw = await _prod_client.get_batches(limit=50, token=actual_user.token)
        all_batches = raw if isinstance(raw, list) else []
        matching = [
            normalize_cpg_record(b)
            for b in all_batches
            if drug_id in (str(b.get("finished_drugs") or b.get("product") or b.get("id") or ""))
        ]
        return DomainResponse(
            status="success",
            data=sanitize_for_llm({
                "drug_id": drug_id,
                "batches": matching,
                "batches_count": len(matching),
            }),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@genealogy_router.get("/operator/{operator_id}", response_model=DomainResponse)
async def get_operator_history(
    operator_id: str,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Operator batch and qualification history."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "training")
        verify_resource_permission(permissions, "batch")

        code_tr, tr_data = await _comp_client.get_training_by_operator(operator_id, token=actual_user.token)
        trainings = tr_data if isinstance(tr_data, list) else ([tr_data] if tr_data else [])

        code_b, b_data = await _prod_client.get_batches(limit=50, token=actual_user.token)
        all_batches = b_data if isinstance(b_data, list) else []

        matching: list[dict[str, Any]] = []
        equipment_used: set[str] = set()
        deviations_linked: list[str] = []

        for b in all_batches:
            ops = [str(o) for o in (b.get("operators") or [])]
            if operator_id in ops or b.get("operator_id") == operator_id:
                matching.append(normalize_cpg_record(b))
                for eq in b.get("equipment") or []:
                    if eq:
                        equipment_used.add(str(eq))
                for dev in b.get("deviations") or []:
                    if dev:
                        deviations_linked.append(str(dev))

        return DomainResponse(
            status="success",
            data=sanitize_for_llm({
                "operator_id": operator_id,
                "batches_count": len(matching),
                "batches": [b.get("id") or b.get("batch_number") for b in matching],
                "equipment_handled": sorted(equipment_used),
                "deviations_linked": deviations_linked,
                "qualifications": trainings,
                "traceability_chain": "operator -> batches -> equipment -> quality_events",
            }),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@genealogy_router.get("/equipment/{equipment_id}", response_model=DomainResponse)
async def get_equipment_history(
    equipment_id: str,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Equipment batch and deviation history."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "equipment")
        verify_resource_permission(permissions, "batch")

        code_eq, eq_data = await _comp_client.get_equipment(equipment_id, token=actual_user.token)
        code_b, b_data = await _prod_client.get_batches(limit=50, token=actual_user.token)
        all_batches = b_data if isinstance(b_data, list) else []

        matching: list[dict[str, Any]] = []
        for b in all_batches:
            eqs = [str(e) for e in (b.get("equipment") or [])]
            if equipment_id in eqs or b.get("equipment_id") == equipment_id:
                matching.append(normalize_cpg_record(b))

        return DomainResponse(
            status="success",
            data=sanitize_for_llm({
                "equipment_id": equipment_id,
                "equipment": normalize_cpg_record(eq_data),
                "batches_count": len(matching),
                "batches": [b.get("id") or b.get("batch_number") for b in matching],
            }),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


# ── ANALYTICS ENDPOINTS ───────────────────────────────────────────────────────

@analytics_router.post("/risk-score", response_model=DomainResponse)
async def calculate_risk(
    body: RiskScoreRequest,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Compute risk score based on versioned weights."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "batch")
        target = body.batch_id or body.proposed_config or ""
        raw = analytics_service.calculate_risk_score(batch_id_or_config=target, permissions=permissions)
        if inspect.isawaitable(raw):
            raw = await raw
        return DomainResponse(
            status="success",
            data=sanitize_for_llm(raw),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@analytics_router.post("/similar-batches", response_model=DomainResponse)
async def find_similar(
    body: SimilarBatchesRequest,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Find multi-dimensional similar batches."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "batch")
        raw = analytics_service.find_similar_batches(batch_id=body.batch_id, top_n=body.top_n, permissions=permissions)
        if inspect.isawaitable(raw):
            raw = await raw
        return DomainResponse(
            status="success",
            data=sanitize_for_llm(raw),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@analytics_router.post("/detect-trends", response_model=DomainResponse)
async def detect_analytics_trends(
    body: DetectTrendsRequest,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Detect quality drifts and yield anomalies."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "batch")
        raw = analytics_service.detect_trends(
            product_id=body.product_id,
            metric=body.metric,
            window=body.window,
            permissions=permissions,
            token=actual_user.token,
        )
        if inspect.isawaitable(raw):
            raw = await raw
        return DomainResponse(
            status="success",
            data=sanitize_for_llm(raw),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


@analytics_router.post("/recommend-batch", response_model=DomainResponse)
async def recommend_batch(
    body: RecommendBatchRequest,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """GAMP-5 batch configuration recommendation with hard precedence."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    try:
        verify_resource_permission(permissions, "batch")
        raw = recommendation_engine.recommend_batch_configuration(
            product_id=body.product_id,
            target_quantity=body.target_quantity,
            target_date=body.target_date,
            target_location=body.target_location,
            requested_overrides=body.requested_overrides,
            permissions=permissions,
            user_id=actual_user.user_id,
            role=permissions.Dashboard_assign or "User",
            token=actual_user.token,
        )
        if inspect.isawaitable(raw):
            raw = await raw
        return DomainResponse(
            status="success",
            data=sanitize_for_llm(raw),
            endpoints_triggered=get_triggered_endpoints(),
        )
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)


# ── TOOLS CATALOG & UNIFIED EXECUTION ─────────────────────────────────────────

@tools_router.get("", response_model=DomainResponse)
async def list_registered_tools(
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Discover all registered AI capabilities, required permissions, and schemas."""
    from app.ai.registry import list_capabilities
    return DomainResponse(
        status="success",
        data=list_capabilities(),
        endpoints_triggered=[],
    )


@tools_router.post("/execute", response_model=DomainResponse)
async def execute_tool(
    body: ToolExecuteRequest,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Unified execution endpoint for dynamic UI tools or testing harnesses."""
    actual_user = _resolve_caller(user, token=token)
    from app.ai.executor import execute_predefined_tool

    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)
    result = await execute_predefined_tool(
        function_name=body.tool_name,
        arguments=body.parameters,
        permissions=permissions,
        token=actual_user.token,
    )

    if result.get("status") == "permission_denied":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=result.get("error", "Permission denied"),
        )
    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Execution error"),
        )

    return DomainResponse(
        status="success",
        data=result.get("result"),
        endpoints_triggered=get_triggered_endpoints(),
    )


# ── FACILITY & USER ROLE ENDPOINTS ───────────────────────────────────────────

@facility_router.get("/roles", response_model=DomainResponse)
async def list_facility_roles(
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Fetch defined user roles and RBAC configurations from Facility User API."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)

    code, raw = await _fac_client.get_user_roles(token=actual_user.token)
    items = [normalize_cpg_record(x) for x in raw] if isinstance(raw, list) else (normalize_cpg_record(raw) if raw else [])
    return DomainResponse(
        status="success" if code == 200 else ("empty" if code == 404 else "partial"),
        data=sanitize_for_llm(items),
        endpoints_triggered=get_triggered_endpoints(),
    )


@facility_router.get("/roles/{role_id}", response_model=DomainResponse)
async def get_facility_role(
    role_id: str,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Fetch a specific user role definition by ID from Facility User API."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)

    code, raw = await _fac_client.get_user_role_by_id(role_id=role_id, token=actual_user.token)
    return DomainResponse(
        status="success" if code == 200 else "not_found",
        data=sanitize_for_llm(normalize_cpg_record(raw)),
        endpoints_triggered=get_triggered_endpoints(),
    )


@facility_router.get("/users/{user_id}", response_model=DomainResponse)
async def get_facility_user_profile(
    user_id: str,
    user: AuthenticatedUser | Any = Depends(require_facility_user),
    token: str | None = None,
    permissions: ChatPermissions | None = None,
) -> DomainResponse:
    """Fetch a specific user profile and permissions by User GUID from Facility User API."""
    actual_user = _resolve_caller(user, token=token)
    reset_triggered_endpoints()
    set_current_token(actual_user.token)
    set_current_user(user_id=actual_user.user_id, username=actual_user.name)

    if permissions is None:
        permissions = await resolve_permissions(actual_user.token, user_id=actual_user.user_id)

    code, raw = await _fac_client.get_user_by_id(user_id=user_id, token=actual_user.token)
    return DomainResponse(
        status="success" if code == 200 else "not_found",
        data=sanitize_for_llm(normalize_cpg_record(raw)),
        endpoints_triggered=get_triggered_endpoints(),
    )


@facility_router.get("/permissions/me", response_model=DomainResponse)
async def get_my_permissions(
    user: AuthenticatedUser = Depends(require_facility_user),
) -> DomainResponse:
    """Resolve and return effective permissions and role lens for the active caller."""
    reset_triggered_endpoints()
    set_current_token(user.token)
    set_current_user(user_id=user.user_id, username=user.name)
    perms = await resolve_permissions(user.token, user_id=user.user_id)
    return DomainResponse(
        status="success",
        data=perms.model_dump(),
        endpoints_triggered=get_triggered_endpoints(),
    )

