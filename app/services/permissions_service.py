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


def parse_permissions_payload(data: Any) -> ChatPermissions:
    """Parse permissions response payload into ChatPermissions model."""
    if not isinstance(data, dict):
        return ChatPermissions(Dashboard_assign="none", modules=[])

    payload = data.get("result") or data.get("data") or data

    # 1. CPG ManageUser format: { "Permission_settings": { "Dashboard_assign": "All", "Common_permission_settings": [...] } }
    if isinstance(payload, dict) and "Permission_settings" in payload and isinstance(payload["Permission_settings"], dict):
        p_settings = payload["Permission_settings"]
        assign = p_settings.get("Dashboard_assign") or "none"
        raw_modules = p_settings.get("Common_permission_settings") or []
        modules_list: list[ModulePermission] = []
        if isinstance(raw_modules, list):
            for m in raw_modules:
                if isinstance(m, dict):
                    m_name = m.get("Module_name") or m.get("module_name") or ""
                    v = bool(m.get("View", False))
                    modules_list.append(
                        ModulePermission(
                            Module_name=str(m_name),
                            View=v,
                            Create=bool(m.get("Create", False)),
                            Edit=bool(m.get("Edit", False)),
                            Delete=bool(m.get("Delete", False)),
                            Audit_verify=bool(m.get("Audit_verify", False)),
                        )
                    )
        return ChatPermissions(
            Production_calendar=p_settings.get("Production_calendar"),
            Production_scheduler=p_settings.get("Production_scheduler"),
            Compliance_calendar=p_settings.get("Compliance_calendar"),
            Report=p_settings.get("Report"),
            Dashboard_assign=str(assign) if assign else None,
            modules=modules_list,
        )

    # 2. Standard / legacy format
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


def _permissions_from_token_role(token: str) -> ChatPermissions:
    """Resolve permissions from decoded token role when PERMISSIONS_API returns HTML or fails in development."""
    try:
        from app.core.auth import decode_facility_token
        user = decode_facility_token(token)
        role = str(user.raw_claims.get("role") or "").strip().lower()
        if role in ("admin", "superadmin", "production", "qa", "quality", "manager"):
            all_mods = [
                ModulePermission(Module_name="Batch Record", View=True),
                ModulePermission(Module_name="Inventory", View=True),
                ModulePermission(Module_name="Compliance", View=True),
                ModulePermission(Module_name="Deviation", View=True),
                ModulePermission(Module_name="Equipment", View=True),
                ModulePermission(Module_name="Training", View=True),
            ]
            return ChatPermissions(Dashboard_assign="All", modules=all_mods)
        if role == "sales":
            return ChatPermissions(Dashboard_assign="Sales", modules=[])
    except Exception:
        pass
    return ChatPermissions(Dashboard_assign="none", modules=[])


async def resolve_permissions(token: str, user_id: str | None = None) -> ChatPermissions:
    """
    Resolve authoritative user permissions directly from CPG backend APIs.

    SECURITY BOUNDARY:
    Checks CPG User Management permissions endpoint (ManageUser/{user_id}) first,
    falling back to PERMISSIONS_API or token role claims.
    If the call returns 401/403, rejects the request as invalid/expired token.
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
    if clean_token:
        try:
            from app.core.auth import decode_facility_token
            decoded = decode_facility_token(clean_token)
            token_user_id = (decoded.user_id or "").strip()
            token_user_name = (decoded.name or "").strip()
        except Exception:
            pass

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
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(manage_user_url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict) and data.get("result", {}).get("Permission_settings"):
                        return parse_permissions_payload(data)
                elif resp.status_code in (401, 403):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired facility access token (rejected by permissions authority).",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(f"Live ManageUser check failed at {manage_user_url}: {exc}")

    # 2. Fallback: Query configured permissions_api
    url = (settings.permissions_api or "https://sales.cpguardian.com/users/roles").strip()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)

            if response.status_code in (401, 403):
                logger.warning(f"Permissions API rejected token with status {response.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired facility access token (rejected by permissions authority).",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            if response.status_code >= 400:
                logger.error(f"Permissions API returned error status {response.status_code}: {response.text}")
                if not settings.is_production:
                    return _permissions_from_token_role(clean_token)
                return ChatPermissions(Dashboard_assign="none", modules=[])

            if "text/html" in response.headers.get("content-type", ""):
                logger.warning(f"Permissions API at {url} returned HTML. Resolving role from verified token claims.")
                return _permissions_from_token_role(clean_token)

            data = response.json()
            return parse_permissions_payload(data)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to connect to Permissions API ({url}): {exc}")
        if not settings.is_production:
            return _permissions_from_token_role(clean_token)
        # Fail closed in production
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

