"""Chat generation for CPG AI Assistant using OpenAI (GPT-4o-mini).

Wires LLM function calling to predefined, read-only CPG functions.
The LLM never constructs or executes raw database queries.
Enforces SRS Section 10 evidence standards and captures audit metadata.
Uses the real API key strictly loaded from the environment (.env).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from openai import APIError, AsyncAzureOpenAI, AsyncOpenAI, AuthenticationError, RateLimitError

from app.agent.prompts import SYSTEM_PROMPT
from app.core.config import settings
from app.services.capabilities import ChatPermissions
from app.services.facility_api_service import (
    PREDEFINED_TOOL_SCHEMAS,
    execute_predefined_tool,
)

logger = logging.getLogger(__name__)


class AiConfigError(Exception):
    """Raised when provider keys/config are missing."""


class AiAuthError(Exception):
    """Raised when the provider rejects the API key."""


class AiRateLimitError(Exception):
    """Raised when the provider rate-limits the request."""


class AiProviderError(Exception):
    """Raised on other provider failures."""


class GenerationResult:
    """Carries the final response text along with audit metadata."""

    def __init__(
        self,
        text: str,
        functions_called: list[dict[str, Any]] | None = None,
        retrieved_record_ids: list[str] | None = None,
        permission_denied: bool = False,
        denial_reason: str | None = None,
        entity_type: str = "",
        traceability_chain: str = "",
        risk_config_version: str = "",
    ) -> None:
        self.text = text
        self.functions_called = functions_called or []
        self.retrieved_record_ids = retrieved_record_ids or []
        self.permission_denied = permission_denied
        self.denial_reason = denial_reason
        self.entity_type = entity_type
        self.traceability_chain = traceability_chain
        self.risk_config_version = risk_config_version

    def __str__(self) -> str:
        return self.text


def _extract_result_metadata(
    fn_name: str,
    args: dict[str, Any],
    tool_result: dict[str, Any],
) -> tuple[str, str, list[str], str]:
    """Extract entity_type, traceability_chain, record IDs, and risk_config_version from function execution."""
    result = tool_result.get("result") or {}
    chain = ""
    entity_type = ""
    risk_config_ver = ""
    if isinstance(result, dict):
        chain = result.get("traceability_chain") or ""
        entity_type = result.get("entity_type") or ""
        risk_config_ver = result.get("config_version") or ""

    if not entity_type:
        if "material" in fn_name or ("lot_id" in args and ("rm-" in str(args).lower() or "chem" in str(args).lower())):
            entity_type = "material_lot"
        elif "component" in fn_name or ("lot_id" in args and "comp" in str(args).lower()):
            entity_type = "component"
        elif "operator" in fn_name:
            entity_type = "operator"
        elif "equipment" in fn_name:
            entity_type = "equipment"
        elif "finished_drug" in fn_name or "drug" in fn_name:
            entity_type = "finished_drug"
        elif "deviation" in fn_name:
            entity_type = "deviation"
        elif "recommend" in fn_name:
            entity_type = "batch_recommendation"
        elif "batch" in fn_name or "similar" in fn_name or "risk" in fn_name:
            entity_type = "batch"
        elif "trend" in fn_name or "product" in fn_name:
            entity_type = "product"
        elif "entity_type" in args:
            entity_type = str(args["entity_type"])

    record_ids: list[str] = []
    for key in ("batch_id", "operator_id", "equipment_id", "lot_number", "deviation_id", "lot_id", "drug_id_or_lot", "entity_id", "product_id"):
        if key in args and args[key]:
            val = str(args[key])
            if val not in record_ids:
                record_ids.append(val)
    if isinstance(result, dict):
        for key in ("record_id", "lot_id", "finished_drug_id", "deviation_id", "operator_id", "equipment_id", "target_batch_id", "batch_id", "product_id"):
            val = result.get(key)
            if val and str(val) not in record_ids:
                record_ids.append(str(val))

    return entity_type, chain, record_ids, risk_config_ver


def _compose_system(*, username: str | None, extra_context: str | None) -> str:
    user_line = f"The signed-in user is named {username}." if username else ""
    parts = [SYSTEM_PROMPT.strip(), user_line, (extra_context or "").strip()]
    return "\n\n".join(part for part in parts if part)


def _build_openai_client() -> AsyncOpenAI | AsyncAzureOpenAI:
    key = settings.openai_api_key.strip()
    endpoint = (settings.azure_openai_endpoint or "").strip().rstrip("/")
    base_url = (settings.openai_base_url or "").strip().rstrip("/")

    if endpoint:
        if not key:
            raise AiConfigError("OPENAI_API_KEY is not configured in .env.")
        return AsyncAzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=(settings.azure_openai_api_version or "2024-08-01-preview").strip(),
        )

    if base_url:
        return AsyncOpenAI(
            api_key=key or "ollama",
            base_url=base_url,
            default_headers={"ngrok-skip-browser-warning": "true"},
        )

    if not key:
        raise AiConfigError("OPENAI_API_KEY is not configured in .env.")
    return AsyncOpenAI(api_key=key)


def _to_openai_messages(
    *,
    system: str,
    history: list[dict[str, str]] | None,
    message: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for turn in history or []:
        role = (turn.get("role") or "user").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        openai_role = "assistant" if role in {"assistant", "model"} else "user"
        messages.append({"role": openai_role, "content": content})
    messages.append({"role": "user", "content": message.strip()})
    return messages


async def _generate_openai_with_tools(
    message: str,
    *,
    username: str | None = None,
    history: list[dict[str, str]] | None = None,
    extra_context: str | None = None,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> GenerationResult:
    """Execute chat completion strictly using the live OpenAI tool-calling API."""
    model = (settings.ai_model or "gpt-4o-mini").strip()
    system = _compose_system(username=username, extra_context=extra_context)
    client = _build_openai_client()

    messages = _to_openai_messages(system=system, history=history, message=message)

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=PREDEFINED_TOOL_SCHEMAS,
            temperature=0.2,
            max_tokens=1024,
        )
    except AuthenticationError as exc:
        raise AiAuthError(f"OpenAI authentication failed: {exc}") from exc
    except RateLimitError as exc:
        raise AiRateLimitError(f"OpenAI rate limit / quota exceeded: {exc}") from exc
    except Exception as exc:
        raise AiProviderError(f"OpenAI request failed: {exc}") from exc
    finally:
        await client.close()

    choice = completion.choices[0]
    msg = choice.message

    # Check if the LLM requested a function call
    if not msg.tool_calls:
        return GenerationResult(text=(msg.content or "").strip())

    functions_called: list[dict[str, Any]] = []
    record_ids: list[str] = []
    is_any_denied = False
    denial_reason = None

    messages.append(msg.to_dict() if hasattr(msg, "to_dict") else dict(msg))

    entity_type = ""
    traceability_chain = ""
    risk_config_version = ""

    for tool_call in msg.tool_calls:
        fn_name = tool_call.function.name
        try:
            fn_args = json.loads(tool_call.function.arguments)
        except Exception:
            fn_args = {}

        functions_called.append({"name": fn_name, "parameters": fn_args})

        # Execute safe predefined function with permissions and token forwarded
        tool_result = execute_predefined_tool(fn_name, fn_args, permissions=permissions, token=token)

        if tool_result.get("status") == "permission_denied":
            is_any_denied = True
            denial_reason = tool_result.get("error")

        t_entity, t_chain, t_records, t_config_ver = _extract_result_metadata(fn_name, fn_args, tool_result)
        if t_entity and not entity_type:
            entity_type = t_entity
        if t_chain and not traceability_chain:
            traceability_chain = t_chain
        if t_config_ver and not risk_config_version:
            risk_config_version = t_config_ver
        for rid in t_records:
            if rid not in record_ids:
                record_ids.append(rid)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result),
            }
        )

    # Second call to let the LLM generate the final response with the retrieved data
    client2 = _build_openai_client()
    try:
        final_completion = await client2.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        final_text = (final_completion.choices[0].message.content or "").strip()
    except AuthenticationError as exc:
        raise AiAuthError(f"OpenAI authentication failed: {exc}") from exc
    except RateLimitError as exc:
        raise AiRateLimitError(f"OpenAI rate limit / quota exceeded: {exc}") from exc
    except Exception as exc:
        raise AiProviderError(f"OpenAI request failed: {exc}") from exc
    finally:
        await client2.close()

    return GenerationResult(
        text=final_text,
        functions_called=functions_called,
        retrieved_record_ids=record_ids,
        permission_denied=is_any_denied,
        denial_reason=denial_reason,
        entity_type=entity_type,
        traceability_chain=traceability_chain,
        risk_config_version=risk_config_version,
    )


# ── Primary Entry Point ───────────────────────────────────────────────────────


async def generate_response(
    message: str,
    *,
    username: str | None = None,
    history: list[dict[str, str]] | None = None,
    extra_context: str | None = None,
    permissions: ChatPermissions | None = None,
    token: str | None = None,
) -> GenerationResult:
    """
    Generate assistant reply using the OpenAI provider and API key from .env.
    Uses OpenAI tool-calling to execute predefined CPG functions; never allows raw queries.
    Strictly executes against OpenAI with NO fallback to heuristic mocks.
    """
    # Reject raw database queries immediately
    if re.search(r"\b(select\s+.*from|drop\s+table|delete\s+from|insert\s+into|update\s+.*set)\b", message, re.I):
        return GenerationResult(
            text="I cannot construct or execute raw database queries. Direct database querying is strictly prohibited. All data access is restricted to predefined, authorized CPG functions.",
            permission_denied=False,
        )

    if not (settings.openai_base_url or "").strip() and not (settings.openai_api_key or "").strip():
        raise AiConfigError("OPENAI_API_KEY is not configured in .env.")

    return await _generate_openai_with_tools(
        message,
        username=username,
        history=history,
        extra_context=extra_context,
        permissions=permissions,
        token=token,
    )
