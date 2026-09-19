"""Dedicated AI Summary APIs for explicit, deterministic microservice triggering.

Bypasses free-form chat regex by providing direct, strongly-typed endpoints
that explicitly trigger downstream CPG microservices and return both
structured telemetry and AI executive insights with full endpoint execution transparency.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.auth import AuthenticatedUser, require_facility_user
from app.services.ai_service import generate_response
from app.services.dashboard_service import format_snapshot_for_prompt
from app.services.facility_api_service import (
    get_compliance_dashboard_data,
    get_production_dashboard_data,
    get_triggered_endpoints,
    reset_triggered_endpoints,
    set_current_token,
    set_current_user,
    verify_resource_permission,
)
from app.clients.base_client import resolve_user_guid
from app.services.permissions_service import PermissionDeniedError, resolve_permissions

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/summary",
    tags=["AI Summary"],
)


class SummaryRequest(BaseModel):
    facility_user_id: str | None = None
    facility_user_name: str | None = None
    focus_area: str | None = None


class DedicatedSummaryResponse(BaseModel):
    scope: str
    summary: str
    endpoints_triggered: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    status: str = "success"


@router.post("/production", response_model=DedicatedSummaryResponse)
async def production_summary(
    request: SummaryRequest | None = None,
    user: AuthenticatedUser = Depends(require_facility_user),
):
    """
    Explicitly trigger CPG Production microservices:
    - /api/DashboardProduction/GetProductionDetails
    - /api/BatchRecord
    - /api/ChemicalChildInventory

    Synthesizes live telemetry into an executive operational AI summary.
    """
    reset_triggered_endpoints()
    set_current_token(user.token)
    if user.user_id and "@" in user.user_id:
        dyn_guid, dyn_name = await resolve_user_guid(user.user_id, token=user.token)
        if dyn_guid:
            user.user_id = dyn_guid
        if dyn_name and not user.name:
            user.name = dyn_name

    display_name = (request.facility_user_name if request else None) or user.name
    set_current_user(user_id=user.user_id, username=display_name, collection_id=user.collection_id)

    effective_permissions = await resolve_permissions(user.token, user_id=user.user_id)
    try:
        verify_resource_permission(effective_permissions, "production")
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=exc.message,
        ) from exc

    try:
        data = await get_production_dashboard_data(
            user_id=user.user_id,
            permissions=effective_permissions,
            token=user.token,
        )
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=exc.message,
        ) from exc

    if not data or data.get("status") == "error" or (isinstance(data, dict) and data.get("details") is None and len(data) <= 1):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Downstream Production service timed out or failed to return data.",
        )

    formatted_context = format_snapshot_for_prompt({"production": data})
    prompt = "Provide an executive operational summary of today's production KPIs, active batches, and critical alerts."
    if request and request.focus_area:
        prompt += f" Focus specifically on: {request.focus_area}."

    gen_result = await generate_response(
        prompt,
        username=display_name,
        extra_context=formatted_context,
        permissions=effective_permissions,
        token=user.token,
    )
    endpoints = get_triggered_endpoints()

    return DedicatedSummaryResponse(
        scope="production",
        summary=str(gen_result),
        endpoints_triggered=endpoints,
        data=data,
        status="success",
    )


@router.post("/compliance", response_model=DedicatedSummaryResponse)
async def compliance_summary(
    request: SummaryRequest | None = None,
    user: AuthenticatedUser = Depends(require_facility_user),
):
    """
    Explicitly trigger CPG Compliance microservices:
    - /api/DashboardMonitor/GetOOCDetails
    - /api/v1/TaskManagement
    - /api/Equipment

    Synthesizes live telemetry into an executive QA compliance risk assessment.
    """
    reset_triggered_endpoints()
    set_current_token(user.token)
    if user.user_id and "@" in user.user_id:
        dyn_guid, dyn_name = await resolve_user_guid(user.user_id, token=user.token)
        if dyn_guid:
            user.user_id = dyn_guid
        if dyn_name and not user.name:
            user.name = dyn_name

    display_name = (request.facility_user_name if request else None) or user.name
    set_current_user(user_id=user.user_id, username=display_name, collection_id=user.collection_id)

    effective_permissions = await resolve_permissions(user.token, user_id=user.user_id)
    try:
        verify_resource_permission(effective_permissions, "compliance")
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=exc.message,
        ) from exc

    try:
        data = await get_compliance_dashboard_data(
            user_id=user.user_id,
            permissions=effective_permissions,
            token=user.token,
        )
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=exc.message,
        ) from exc

    if not data or data.get("status") == "error" or (isinstance(data, dict) and data.get("details") is None and len(data) <= 1):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Downstream Compliance service (facility-user-api.cpguardian.com/api/DashboardMonitor/GetTodayTask) timed out or failed to return data.",
        )

    formatted_context = format_snapshot_for_prompt({"compliance": data})
    prompt = (
        "Provide an executive QA compliance risk assessment covering open deviations, "
        "out-of-compliance alerts, and overdue equipment or calibrations."
    )
    if request and request.focus_area:
        prompt += f" Focus specifically on: {request.focus_area}."

    gen_result = await generate_response(
        prompt,
        username=display_name,
        extra_context=formatted_context,
        permissions=effective_permissions,
        token=user.token,
    )
    endpoints = get_triggered_endpoints()

    return DedicatedSummaryResponse(
        scope="compliance",
        summary=str(gen_result),
        endpoints_triggered=endpoints,
        data=data,
        status="success",
    )


@router.post("/overall", response_model=DedicatedSummaryResponse)
@router.post("", response_model=DedicatedSummaryResponse)
async def overall_summary(
    request: SummaryRequest | None = None,
    user: AuthenticatedUser = Depends(require_facility_user),
):
    """
    Explicitly triggers both Production and Compliance microservices for a plant-wide operational briefing.
    """
    reset_triggered_endpoints()
    set_current_token(user.token)
    if user.user_id and "@" in user.user_id:
        dyn_guid, dyn_name = await resolve_user_guid(user.user_id, token=user.token)
        if dyn_guid:
            user.user_id = dyn_guid
        if dyn_name and not user.name:
            user.name = dyn_name

    display_name = (request.facility_user_name if request else None) or user.name
    set_current_user(user_id=user.user_id, username=display_name, collection_id=user.collection_id)

    effective_permissions = await resolve_permissions(user.token, user_id=user.user_id)

    combined_data: dict[str, Any] = {}
    # Fetch production if allowed
    try:
        verify_resource_permission(effective_permissions, "production")
        combined_data["production"] = await get_production_dashboard_data(
            user_id=user.user_id,
            permissions=effective_permissions,
            token=user.token,
        )
    except PermissionDeniedError:
        pass

    # Fetch compliance if allowed
    try:
        verify_resource_permission(effective_permissions, "compliance")
        combined_data["compliance"] = await get_compliance_dashboard_data(
            user_id=user.user_id,
            permissions=effective_permissions,
            token=user.token,
        )
    except PermissionDeniedError:
        pass

    if not combined_data:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your role cannot access Production or Compliance summary data.",
        )

    formatted_context = format_snapshot_for_prompt(combined_data)
    gen_result = await generate_response(
        "Provide an executive operational summary covering active production batches and compliance status.",
        username=display_name,
        extra_context=formatted_context,
        permissions=effective_permissions,
        token=user.token,
    )
    endpoints = get_triggered_endpoints()

    return DedicatedSummaryResponse(
        scope="overall",
        summary=str(gen_result),
        endpoints_triggered=endpoints,
        data=combined_data,
        status="success",
    )

