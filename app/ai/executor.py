"""Predefined tool execution engine for safe LLM function calling.

Dispatches function calls strictly to validated, read-only tools in app.ai.tools.
The LLM cannot execute arbitrary code, queries, or API calls.
Dispatches function calls strictly to validated, read-only tools in app.ai.tools
gated by the authoritative AICapability registry.
The LLM cannot:
- Execute arbitrary code, queries, or API calls
- Bypass permissions or override security context
- Choose arbitrary URLs, endpoints, or HTTP methods
- Supply arbitrary authorization headers or spoof identity
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

from app.ai import tools
from app.ai.registry import CAPABILITY_REGISTRY, get_capability
from app.ai.sanitizer import sanitize_for_llm
from app.services.capabilities import ChatPermissions
from app.services.permissions_service import (
    PermissionDeniedError,
    verify_resource_permission,
)

logger = logging.getLogger(__name__)

# Backward-compatible dictionary reference
PREDEFINED_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_batch_by_id": tools.get_batch_by_id,
    "get_batches": tools.get_batches,
    "get_operator_training_status": tools.get_operator_training_status,
    "get_equipment_status": tools.get_equipment_status,
    "get_material_lot_trace": tools.get_material_lot_trace,
    "get_deviation_by_id": tools.get_deviation_by_id,
    "get_environmental_monitoring": tools.get_environmental_monitoring,
    "get_production_dashboard_data": tools.get_production_dashboard_data,
    "get_compliance_dashboard_data": tools.get_compliance_dashboard_data,
    "get_related_batches": tools.get_related_batches,
    "get_material_lot_genealogy": tools.get_material_lot_genealogy,
    "get_component_lot_genealogy": tools.get_component_lot_genealogy,
    "get_operator_batch_history": tools.get_operator_batch_history,
    "get_equipment_batch_history": tools.get_equipment_batch_history,
    "get_finished_drug_genealogy": tools.get_finished_drug_genealogy,
    "get_deviation_impact": tools.get_deviation_impact,
    "investigate_entity": tools.investigate_entity,
    "find_similar_batches": tools.find_similar_batches,
    "calculate_risk_score": tools.calculate_risk_score,
    "detect_trends": tools.detect_trends,
    "recommend_batch_configuration": tools.recommend_batch_configuration,
    **{cap.name: cap.handler for cap in CAPABILITY_REGISTRY.values()},
}

# Forbidden argument keys that an LLM or client might attempt to inject
_FORBIDDEN_ARGUMENT_KEYS = {
    "permissions",
    "token",
    "user_id",
    "username",
    "headers",
    "url",
    "endpoint",
    "method",
    "http_method",
    "role",
}


def sanitize_tool_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Strip any security context or network injection parameters from tool arguments."""
    if not arguments:
        return {}
    clean: dict[str, Any] = {}
    for k, v in arguments.items():
        if k.strip().lower() in _FORBIDDEN_ARGUMENT_KEYS:
            logger.warning(
                "Security notice: Attempted injection of parameter '%s' into tool execution stripped.",
                k,
            )
            continue
        clean[k] = v
    return clean


async def execute_predefined_tool(
    function_name: str,
    arguments: dict[str, Any] | None = None,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """
    Execute a predefined function requested via LLM function calling or REST API.

    Enforces:
    1. Capability Registry Gating: rejects any function not in the strict registry.
    2. Parameter Sanitization: strips any LLM attempts to override permissions, URLs, or identity.
    3. Application-Level Python Permission Check: verifies authoritative permissions before calling downstream APIs.
    4. Downstream Response Sanitization: purges secrets, tokens, and internal metadata.
    """
    fn_name = (function_name or "").strip()
    capability = get_capability(fn_name)
    if not capability or fn_name not in PREDEFINED_FUNCTIONS:
        raise ValueError(
            f"Unauthorized function '{fn_name}'. The LLM can only invoke predefined functions: "
            f"{list(CAPABILITY_REGISTRY.keys())}. Arbitrary database queries are strictly prohibited."
        )

    clean_args = sanitize_tool_arguments(arguments)

    # ── APPLICATION-LEVEL PERMISSION CHECK IN PYTHON ──────────────────────────
    # The application decides WHETHER the user is allowed to perform the operation.
    try:
        verify_resource_permission(permissions, capability.required_resource)
    except PermissionDeniedError as exc:
        return {
            "function": fn_name,
            "arguments": clean_args,
            "error": exc.message,
            "status": "permission_denied",
        }

    # Inject verified server-side security context (never from user/LLM input)
    args_to_pass = dict(clean_args)
    args_to_pass["permissions"] = permissions
    if token:
        args_to_pass["token"] = token

    try:
        raw_result = capability.handler(**args_to_pass)
        if inspect.isawaitable(raw_result):
            raw_result = await raw_result

        # Sanitize downstream response before exposing to LLM or client
        sanitized_result = sanitize_for_llm(raw_result)

        return {
            "function": fn_name,
            "arguments": clean_args,
            "result": sanitized_result,
            "status": "success",
        }
    except PermissionDeniedError as exc:
        return {
            "function": fn_name,
            "arguments": clean_args,
            "error": exc.message,
            "status": "permission_denied",
        }
    except Exception as exc:
        logger.error(f"Error executing predefined function {fn_name}: {exc}", exc_info=True)
        return {
            "function": fn_name,
            "arguments": clean_args,
            "error": str(exc),
            "status": "error",
        }
