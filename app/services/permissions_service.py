"""Permissions service for CPG AI.

Verifies role lens and module View permissions before any data retrieval is executed.
"""

from __future__ import annotations

from typing import Any

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


class PermissionDeniedError(Exception):
    """Raised when a user lacks permission to view requested CPG records."""

    def __init__(self, message: str, resource: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.resource = resource


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
    "wants_dashboard_data",
    "requested_scopes",
    "normalize_lens",
    "has_module_view",
    "allowed_scopes",
    "resolve_scopes",
    "refuse_message",
    "verify_resource_permission",
]

