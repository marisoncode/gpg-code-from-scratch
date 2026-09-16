"""Role lens + module View gates for dashboard reads."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DashboardLens = Literal["production", "compliance", "sales", "none", "all"]
DashboardScope = Literal["production", "compliance"]

_SUMMARY_RE = re.compile(
    r"\b("
    r"dashboard|summary|summar(?:y|ise|ize)|kpi|kpis|"
    r"alert|alerts|ooc|out of compliance|"
    r"batch status|today'?s batch|upcoming batch|upcoming batches|"
    r"low inventory|operator|operators|"
    r"compliance (?:status|alert|task|tasks|calendar)|"
    r"production (?:status|alert|alerts|summary)"
    r")\b",
    re.IGNORECASE,
)


class ModulePermission(BaseModel):
    model_config = ConfigDict(extra="ignore")

    Module_name: str = ""
    View: bool = False
    Create: bool = False
    Edit: bool = False
    Delete: bool = False
    Audit_verify: bool = False


class ChatPermissions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    Production_calendar: bool | None = None
    Production_scheduler: bool | None = None
    Compliance_calendar: bool | None = None
    Report: bool | None = None
    Dashboard_assign: str | None = None
    modules: list[ModulePermission] = Field(default_factory=list)


def wants_dashboard_data(message: str) -> bool:
    return bool(_SUMMARY_RE.search(message or ""))


def requested_scopes(message: str) -> set[DashboardScope]:
    text = (message or "").lower()
    wants_prod = bool(re.search(r"\bproduction\b|\bbatch\b|\binventory\b|\boperator", text))
    wants_comp = bool(
        re.search(r"\bcompliance\b|\booc\b|\btraining\b|\btask\b|\benvironmental\b|\bpersonnel\b", text)
    )
    if wants_prod and not wants_comp:
        return {"production"}
    if wants_comp and not wants_prod:
        return {"compliance"}
    return {"production", "compliance"}


def normalize_lens(raw: str | None) -> DashboardLens:
    value = (raw or "").strip().lower()
    if value in {"production", "compliance", "sales", "none", "all"}:
        return value  # type: ignore[return-value]
    return "all"


def has_module_view(permissions: ChatPermissions | None, module_name: str) -> bool:
    modules = list(permissions.modules) if permissions else []
    if not modules:
        return True
    needle = module_name.strip().lower()
    match = next((m for m in modules if (m.Module_name or "").strip().lower() == needle), None)
    if match is None:
        return False
    return bool(match.View)


def allowed_scopes(permissions: ChatPermissions | None) -> set[DashboardScope]:
    lens = normalize_lens(permissions.Dashboard_assign if permissions else None)
    allowed: set[DashboardScope] = set()
    if lens in {"production", "all"}:
        if has_module_view(permissions, "Batch Record") or has_module_view(permissions, "Inventory"):
            allowed.add("production")
    if lens in {"compliance", "all"}:
        if has_module_view(permissions, "Compliance"):
            allowed.add("compliance")
    return allowed


def resolve_scopes(message: str, permissions: ChatPermissions | None) -> set[DashboardScope]:
    return requested_scopes(message) & allowed_scopes(permissions)


def refuse_message(requested: set[DashboardScope], allowed: set[DashboardScope], lens: DashboardLens) -> str:
    denied = requested - allowed
    if not denied:
        return ""
    if lens == "sales":
        return (
            "Your role is assigned to Sales. Production and Compliance dashboard "
            "summaries are not available. Ask an admin to change Dashboard_assign."
        )
    if lens == "none":
        return (
            "No active dashboard permissions assigned. Access to production and compliance records is denied."
        )
    labels = " and ".join(sorted(denied)).title()
    return (
        f"Your role cannot access the {labels} dashboard. "
        "Ask an admin to enable the matching module View or Dashboard_assign."
    )
