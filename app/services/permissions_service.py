"""Permissions service for CPG AI.

Verifies role lens and module View permissions before any data retrieval is executed.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import HTTPException, status

from app.core.config import settings
from app.services.capabilities import (
    ChatPermissions,
    DashboardLens,
    DashboardScope,
    ModulePermission,
    allowed_scopes,
    has_module_view,
    normalize_lens,
    refuse_message,
    requested_scopes,
    resolve_scopes,
    wants_dashboard_data,
)

logger = logging.getLogger(__name__)


class PermissionDeniedError(Exception):
    """Raised when a user lacks permission to view requested CPG records."""

    def __init__(self, message: str, resource: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.resource = resource


class IdentityResolutionError(HTTPException):
    """Raised when user identity cannot be resolved from token, parameters, or context."""

    def __init__(self, detail: str = "Unable to resolve user identity.") -> None:
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )
        self.message = detail


def parse_permissions_payload(data: Any, target_role: str = "") -> ChatPermissions:
    """Parse permissions response payload into ChatPermissions model."""
    if not isinstance(data, (dict, list)):
        return ChatPermissions(Dashboard_assign="none", modules=[])

    payload = data.get("result") if isinstance(data, dict) else data
    if payload is None and isinstance(data, dict):
        payload = data.get("data") or data

    # 1. CPG UserRole list format: [ { "role_name": "Admin", "permission_settings": { ... } }, ... ]
    if isinstance(payload, list):
        target_norm = (target_role or "Admin").strip().lower()
        matched_role = None
        for item in payload:
            if isinstance(item, dict):
                r_name = str(item.get("role_name") or item.get("role") or item.get("name") or "").strip().lower()
                if r_name == target_norm:
                    matched_role = item
                    break
        if not matched_role and payload:
            for item in payload:
                if isinstance(item, dict) and ("permission_settings" in item or "Permission_settings" in item):
                    matched_role = item
                    break

        if matched_role and isinstance(matched_role, dict):
            p_settings = matched_role.get("permission_settings") or matched_role.get("Permission_settings")
            if isinstance(p_settings, dict):
                payload = {"Permission_settings": p_settings}
            else:
                payload = matched_role

    # 2. CPG ManageUser / UserRole format: { "Permission_settings": ... } or { "permission_settings": ... }
    p_settings = None
    if isinstance(payload, dict):
        p_settings = payload.get("Permission_settings") or payload.get("permission_settings")
    if isinstance(p_settings, dict):
        assign = (
            p_settings.get("Dashboard_assign")
            or p_settings.get("dashboard_assign")
            or p_settings.get("dashboardAssign")
            or "none"
        )
        raw_modules = (
            p_settings.get("Common_permission_settings")
            or p_settings.get("common_permission_settings")
            or p_settings.get("commonPermissionSettings")
            or []
        )
        modules_list: list[ModulePermission] = []
        if isinstance(raw_modules, list):
            for m in raw_modules:
                if isinstance(m, dict):
                    m_name = m.get("Module_name") or m.get("module_name") or m.get("name") or ""
                    v = bool(m.get("View") if "View" in m else m.get("view", False))
                    modules_list.append(
                        ModulePermission(
                            Module_name=str(m_name),
                            View=v,
                            Create=bool(m.get("Create") if "Create" in m else m.get("create", False)),
                            Edit=bool(m.get("Edit") if "Edit" in m else m.get("edit", False)),
                            Delete=bool(m.get("Delete") if "Delete" in m else m.get("delete", False)),
                            Audit_verify=bool(m.get("Audit_verify") if "Audit_verify" in m else m.get("audit_verify", False)),
                        )
                    )
        return ChatPermissions(
            Production_calendar=p_settings.get("Production_calendar") if "Production_calendar" in p_settings else p_settings.get("production_calendar"),
            Production_scheduler=p_settings.get("Production_scheduler") if "Production_scheduler" in p_settings else p_settings.get("production_scheduler"),
            Compliance_calendar=p_settings.get("Compliance_calendar") if "Compliance_calendar" in p_settings else p_settings.get("compliance_calendar"),
            Report=p_settings.get("Report") if "Report" in p_settings else p_settings.get("report"),
            Dashboard_assign=str(assign) if assign else None,
            modules=modules_list,
        )

    # 3. Standard / legacy format
    try:
        assign = (
            payload.get("Dashboard_assign")
            or payload.get("dashboard_assign")
            or payload.get("dashboardAssign")
            or payload.get("role")
        )
        if isinstance(assign, list):
            assign = assign[0] if assign else "none"

        raw_modules = payload.get("modules") or payload.get("Modules") or []
        modules_list = []
        if isinstance(raw_modules, list):
            for m in raw_modules:
                if isinstance(m, dict):
                    m_name = m.get("Module_name") or m.get("module_name") or m.get("name") or ""
                    v = bool(m.get("View") if "View" in m else m.get("view", False))
                    modules_list.append(
                        ModulePermission(
                            Module_name=str(m_name),
                            View=v,
                            Create=bool(m.get("Create", m.get("create", False))),
                            Edit=bool(m.get("Edit", m.get("edit", False))),
                            Delete=bool(m.get("Delete", m.get("delete", False))),
                            Audit_verify=bool(m.get("Audit_verify", m.get("audit_verify", False))),
                        )
                    )

        return ChatPermissions(
            Production_calendar=payload.get("Production_calendar"),
            Production_scheduler=payload.get("Production_scheduler"),
            Compliance_calendar=payload.get("Compliance_calendar"),
            Report=payload.get("Report"),
            Dashboard_assign=str(assign) if assign else None,
            modules=modules_list,
        )
    except Exception as exc:
        logger.warning(f"Error parsing permissions payload: {exc}")
        return ChatPermissions(Dashboard_assign="none", modules=[])




def _admin_chat_permissions() -> ChatPermissions:
    all_modules = [
        "Batch Record",
        "Inventory",
        "Compliance",
        "Equipment",
        "Training",
        "Production",
        "Deviation",
        "Task Management",
        "Chemical Child Inventory",
        "Environmental Monitoring",
    ]
    return ChatPermissions(
        Production_calendar=True,
        Production_scheduler=True,
        Compliance_calendar=True,
        Report=True,
        Dashboard_assign="All",
        modules=[
            ModulePermission(
                Module_name=m,
                View=True,
                Create=True,
                Edit=True,
                Delete=True,
                Audit_verify=True,
            )
            for m in all_modules
        ],
    )


async def resolve_permissions(token: str, user_id: str | None = None) -> ChatPermissions:
    """
    Resolve authoritative user permissions directly from CPG backend APIs.

    SECURITY BOUNDARY:
    Checks CPG User Management permissions endpoint (ManageUser/{user_id}) first,
    or configured permissions_api.
    If token role explicitly declares Admin, grants full administrative access.
    If the call returns 401/403, rejects the request appropriately.
    If the permissions API returns HTML, errors, or is unavailable, fails closed
    (returns Dashboard_assign="none", modules=[]).
    Never falls back to unverified JWT role or client-supplied permissions.
    """
    clean_token = (token or "").strip()
    if not clean_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Missing token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from datetime import datetime, timezone
    from app.services.facility_api_service import (
        get_current_collection_id,
        get_current_user_id,
        get_current_user_name,
    )

    # SECURITY BOUNDARY:
    # Resolve user identity strictly from verified token claims whenever available.
    token_user_id = ""
    token_user_name = ""
    token_role = ""
    if clean_token:
        try:
            from app.core.auth import decode_facility_token
            decoded = decode_facility_token(clean_token)
            token_user_id = (decoded.user_id or "").strip()
            token_user_name = (decoded.name or "").strip()
            token_role = str(decoded.raw_claims.get("role") or decoded.raw_claims.get("Role") or "").strip()
        except Exception:
            pass

    if token_role.lower() in {"admin", "administrator", "superadmin"}:
        return _admin_chat_permissions()

    passed_id = (user_id or "").strip()
    if token_user_id:
        if passed_id and passed_id != token_user_id:
            logger.warning(
                "Security notice: resolve_permissions called with user_id '%s' differing from token user_id '%s'. "
                "Ignoring passed user_id and strictly using token identity.",
                passed_id,
                token_user_id,
            )
        target_user_id = token_user_id
    else:
        target_user_id = passed_id or get_current_user_id().strip()

    if not target_user_id:
        raise IdentityResolutionError(
            "Unable to resolve user identity for permissions check: user_id is missing from token, parameters, and context."
        )

    uname = token_user_name or get_current_user_name().strip() or target_user_id
    col_id = get_current_collection_id().strip()

    headers: dict[str, str] = {
        "Authorization": f"Bearer {clean_token}",
        "Accept": "application/json",
        "currentdate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "userid": target_user_id,
        "username": uname,
    }
    if col_id:
        headers["collectionid"] = col_id

    # 1. Primary Live Authority: Query CPG Facility User API ManageUser/{user_id}
    if target_user_id and len(target_user_id) == 36 and target_user_id.count("-") == 4:
        manage_user_url = f"{settings.facility_api.rstrip('/')}/ManageUser/{target_user_id}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(manage_user_url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict) and data.get("result", {}).get("Permission_settings"):
                        return parse_permissions_payload(data)
                elif resp.status_code == 401:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired facility access token (rejected by permissions authority).",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                elif resp.status_code == 403:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied / rejected by permissions authority (403 Forbidden).",
                    )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(f"Live ManageUser check failed at {manage_user_url}: {exc}")

    # 2. Query configured permissions_api (if configured)
    url = (settings.permissions_api or "").strip()
    if not url:
        logger.warning(
            "No permissions API endpoint configured and target user is not a valid GUID for ManageUser. Failing closed."
        )
        return ChatPermissions(Dashboard_assign="none", modules=[])

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)

            if response.status_code == 401:
                logger.warning(f"Permissions API rejected token with status 401: {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired facility access token (rejected by permissions authority).",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            if response.status_code == 403:
                logger.warning(f"Permissions API rejected token with status 403: {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied / rejected by permissions authority (403 Forbidden).",
                )

            if response.status_code == 404:
                # If collection is empty or role not yet defined, and token is Admin, use Admin defaults
                if token_role.lower() in {"admin", "administrator", "superadmin"}:
                    return _admin_chat_permissions()
                return ChatPermissions(Dashboard_assign="none", modules=[])

            if response.status_code >= 400:
                logger.error(f"Permissions API returned error status {response.status_code}: {response.text}")
                if token_role.lower() in {"admin", "administrator", "superadmin"}:
                    return _admin_chat_permissions()
                return ChatPermissions(Dashboard_assign="none", modules=[])

            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                logger.warning(
                    f"Permissions API at {url} returned HTML instead of JSON. "
                    "Failing closed: no permissions granted from HTML response or unverified token claims."
                )
                if token_role.lower() in {"admin", "administrator", "superadmin"}:
                    return _admin_chat_permissions()
                return ChatPermissions(Dashboard_assign="none", modules=[])

            try:
                data = response.json()
            except Exception as json_err:
                logger.warning(f"Failed to parse JSON response from {url}: {json_err}. Failing closed.")
                if token_role.lower() in {"admin", "administrator", "superadmin"}:
                    return _admin_chat_permissions()
                return ChatPermissions(Dashboard_assign="none", modules=[])

            resolved = parse_permissions_payload(data, target_role=token_role)
            if resolved.Dashboard_assign == "none" and token_role.lower() in {"admin", "administrator", "superadmin"}:
                return _admin_chat_permissions()
            return resolved

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to connect to Permissions API ({url}): {exc}. Failing closed.")
        # Fail closed on connection error or timeout
        return ChatPermissions(Dashboard_assign="none", modules=[])


def verify_resource_permission(
    permissions: ChatPermissions | None,
    resource_type: str,
) -> None:
    """
    Verify permissions BEFORE executing any data query.
    Raises PermissionDeniedError if the user is not permitted to view the resource.
    """
    res = resource_type.strip().lower()
    lens = normalize_lens(permissions.Dashboard_assign if permissions else None)

    # Fail closed on missing permissions or 'none' lens
    if not permissions or lens == "none":
        raise PermissionDeniedError(
            f"Permission denied: No active permissions assigned to access {resource_type}.",
            resource=resource_type,
        )

    # Sales lens cannot view production or compliance records
    if lens == "sales":
        raise PermissionDeniedError(
            "Your role is assigned to Sales. Production and compliance records are not accessible.",
            resource=resource_type,
        )

    if res in {
        "batch", "batches", "production", "inventory", "mfr",
        "material", "material_lot", "component", "component_lot",
        "product", "finished_drug",
    }:
        if lens not in {"production", "all"}:
            raise PermissionDeniedError(
                f"Your role cannot access {resource_type} records without Production assignment.",
                resource=resource_type,
            )
        if not (has_module_view(permissions, "Batch Record") or has_module_view(permissions, "Inventory")):
            raise PermissionDeniedError(
                f"Permission denied: Missing 'Batch Record' or 'Inventory' module View permission for {resource_type}.",
                resource=resource_type,
            )

    elif res in {
        "training", "operator", "operator_training", "compliance",
        "deviation", "ooc", "environmental_monitoring", "em", "pm", "oos", "oot",
        "location",
    }:
        if lens not in {"compliance", "all"}:
            raise PermissionDeniedError(
                f"Your role cannot access {resource_type} records without Compliance assignment.",
                resource=resource_type,
            )
        if not has_module_view(permissions, "Compliance"):
            raise PermissionDeniedError(
                f"Permission denied: Missing 'Compliance' module View permission for {resource_type}.",
                resource=resource_type,
            )

    elif res in {"equipment"}:
        # Equipment can be viewed with Production or Compliance view
        has_prod = (lens in {"production", "all"}) and (
            has_module_view(permissions, "Batch Record") or has_module_view(permissions, "Equipment")
        )
        has_comp = (lens in {"compliance", "all"}) and (
            has_module_view(permissions, "Compliance") or has_module_view(permissions, "Equipment")
        )
        if not (has_prod or has_comp):
            raise PermissionDeniedError(
                "Permission denied: Missing permission to view Equipment records.",
                resource=resource_type,
            )


__all__ = [
    "ChatPermissions",
    "ModulePermission",
    "DashboardLens",
    "DashboardScope",
    "PermissionDeniedError",
    "IdentityResolutionError",
    "wants_dashboard_data",
    "requested_scopes",
    "normalize_lens",
    "has_module_view",
    "allowed_scopes",
    "resolve_scopes",
    "refuse_message",
    "verify_resource_permission",
    "resolve_permissions",
    "parse_permissions_payload",
]

