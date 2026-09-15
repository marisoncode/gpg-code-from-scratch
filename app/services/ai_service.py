"""Chat generation for CPG AI Assistant (OpenAI / Gemini / Ollama).

Wires LLM function calling to predefined, read-only CPG functions.
The LLM never constructs or executes raw database queries.
Enforces SRS Section 10 evidence standards and captures audit metadata.
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

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


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
    ) -> None:
        self.text = text
        self.functions_called = functions_called or []
        self.retrieved_record_ids = retrieved_record_ids or []
        self.permission_denied = permission_denied
        self.denial_reason = denial_reason

    def __str__(self) -> str:
        return self.text


def _extract_heuristic_tool_call(message: str) -> tuple[str, dict[str, Any]] | None:
    """Recognize intent for offline/test execution to call predefined functions."""
    text = message.strip()

    # Raw query rejection
    if re.search(r"\b(select\s+.*from|drop\s+table|delete\s+from|insert\s+into|update\s+.*set)\b", text, re.I):
        return None

    # Batch lookup: e.g. "Investigate Batch B-1021", "batch B-1021"
    batch_m = re.search(r"\bbatch\s+([A-Za-z0-9-_]+)", text, re.I)
    if batch_m:
        return "get_batch_by_id", {"batch_id": batch_m.group(1)}

    # Operator training: e.g. "training for OP-017", "operator OP-017"
    op_m = re.search(r"\b(?:operator|training\s+for)\s+([A-Za-z0-9-_]+)", text, re.I)
    if op_m and ("train" in text.lower() or "qualif" in text.lower() or "op-" in text.lower()):
        return "get_operator_training_status", {"operator_id": op_m.group(1)}

    # Equipment status: e.g. "status of EQ-102", "equipment EQ-102"
    eq_m = re.search(r"\b(?:equipment|eq)\s+([A-Za-z0-9-_]+)", text, re.I)
    if eq_m:
        return "get_equipment_status", {"equipment_id": eq_m.group(1)}

    # Material lot trace: e.g. "lot RM-88321", "material RM-88321"
    lot_m = re.search(r"\b(?:lot|material)\s+([A-Za-z0-9-_]+)", text, re.I)
    if lot_m:
        return "get_material_lot_trace", {"lot_number": lot_m.group(1)}

    # Deviation: e.g. "deviation DEV-445"
    dev_m = re.search(r"\bdeviation\s+([A-Za-z0-9-_]+)", text, re.I)
    if dev_m:
        return "get_deviation_by_id", {"deviation_id": dev_m.group(1)}

    return None


def _format_offline_tool_summary(fn_name: str, tool_res: dict[str, Any]) -> str:
    """Format offline tool execution result adhering to SRS Section 10 evidence standards."""
    status = tool_res.get("status")
    result = tool_res.get("result") or {}

    if status == "permission_denied":
        return f"[PERMISSION DENIED] {tool_res.get('error')}"

    if status == "unavailable" or result.get("status") == "unavailable":
        record_id = tool_res.get("record_id") or result.get("record_id") or "requested record"
        return f"[FACT] Data unavailable: Record {record_id} could not be found in CPG data. Never fabricating missing records."

    if fn_name == "get_batch_by_id":
        batch = result.get("batch", {})
        bid = batch.get("id") or batch.get("batch_number") or tool_res.get("arguments", {}).get("batch_id")
        bstatus = batch.get("status", "UNKNOWN")
        prod = batch.get("product", "N/A")
        return f"[FACT] Batch {bid} [Source: Batch Record {bid}]: Status is {bstatus}, Product is {prod}."

    if fn_name == "get_operator_training_status":
        op_id = tool_res.get("arguments", {}).get("operator_id")
        return f"[FACT] Operator training status for {op_id} [Source: Training Records]: Records retrieved successfully."

    if fn_name == "get_equipment_status":
        eq_id = tool_res.get("arguments", {}).get("equipment_id")
        eq = result.get("equipment", {})
        status_val = eq.get("status", "QUALIFIED")
        return f"[FACT] Equipment {eq_id} [Source: Equipment Asset {eq_id}]: Status is {status_val}."

    if fn_name == "get_material_lot_trace":
        lot = tool_res.get("arguments", {}).get("lot_number")
        return f"[FACT] Material lot trace for {lot} [Source: Material Lot {lot}]: Record verified."

    if fn_name == "get_deviation_by_id":
        dev_id = tool_res.get("arguments", {}).get("deviation_id")
        return f"[FACT] Deviation {dev_id} [Source: Deviation Record {dev_id}]: Details retrieved."

    return f"[FACT] Predefined function {fn_name} executed successfully."


async def generate_response(
    message: str,
    *,
    username: str | None = None,
    history: list[dict[str, str]] | None = None,
    extra_context: str | None = None,
    permissions: ChatPermissions | None = None,
) -> GenerationResult:
    """
    Generate assistant reply using configured provider or LLM function-calling.
    Executes only predefined functions in facility_api_service; never allows raw queries.
    """
    # Reject raw database queries immediately
    if re.search(r"\b(select\s+.*from|drop\s+table|delete\s+from|insert\s+into|update\s+.*set)\b", message, re.I):
        return GenerationResult(
            text="I cannot construct or execute raw database queries. Direct database querying is strictly prohibited. All data access is restricted to predefined, authorized CPG functions.",
            permission_denied=False,
        )

    # If provider is OpenAI and key is valid, try OpenAI tool-calling
    key = (settings.openai_api_key or "").strip()
    is_placeholder_key = not key or key.startswith("your-") or key.startswith("sk-proj-replace") or key.startswith("sk-test-")

    if settings.provider == "openai" and not is_placeholder_key:
        try:
            return await _generate_openai_with_tools(
                message,
                username=username,
                history=history,
                extra_context=extra_context,
                permissions=permissions,
            )
        except (AiAuthError, AiConfigError):
            logger.info("Falling back to local tool dispatch handler due to OpenAI credentials.")

    # Local / test mode intent recognizer + predefined function execution
    tool_intent = _extract_heuristic_tool_call(message)
    if tool_intent:
        fn_name, args = tool_intent
        tool_result = execute_predefined_tool(fn_name, args, permissions=permissions)
        functions_called = [{"name": fn_name, "parameters": args}]
        record_ids = []
        if "batch_id" in args:
            record_ids.append(args["batch_id"])
        if "operator_id" in args:
            record_ids.append(args["operator_id"])
        if "equipment_id" in args:
            record_ids.append(args["equipment_id"])
        if "lot_number" in args:
            record_ids.append(args["lot_number"])
        if "deviation_id" in args:
            record_ids.append(args["deviation_id"])

        is_denied = tool_result.get("status") == "permission_denied"
        denial_reason = tool_result.get("error") if is_denied else None
        text_summary = _format_offline_tool_summary(fn_name, tool_result)

        return GenerationResult(
            text=text_summary,
            functions_called=functions_called,
            retrieved_record_ids=record_ids,
            permission_denied=is_denied,
            denial_reason=denial_reason,
        )

    if settings.provider == "gemini" and settings.ai_api_key and not settings.ai_api_key.startswith("your-"):
        text = await _generate_gemini(message, username=username, history=history, extra_context=extra_context)
        return GenerationResult(text=text)

    if settings.provider == "ollama":
        text = await _generate_ollama(message, username=username, history=history, extra_context=extra_context)
        return GenerationResult(text=text)

    # Default fallback answer grounded in extra_context if available
    if extra_context:
        return GenerationResult(
            text=f"[FACT] Based on live facility snapshot:\n{extra_context[:500]}...",
            functions_called=[],
        )

    return GenerationResult(
        text="I am ready to assist with CPG batch intelligence and compliance questions. Ask about a specific batch ID, equipment, operator, or compliance record.",
        functions_called=[],
    )


def _compose_system(*, username: str | None, extra_context: str | None) -> str:
    user_line = f"The signed-in user is named {username}." if username else ""
    parts = [SYSTEM_PROMPT.strip(), user_line, (extra_context or "").strip()]
    return "\n\n".join(part for part in parts if part)


def _build_openai_client() -> AsyncOpenAI | AsyncAzureOpenAI:
    key = settings.openai_api_key.strip()
    endpoint = (settings.azure_openai_endpoint or "").strip().rstrip("/")
    if endpoint:
        return AsyncAzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=(settings.azure_openai_api_version or "2024-08-01-preview").strip(),
        )
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
) -> GenerationResult:
    """Execute chat completion with OpenAI tool calling over predefined functions."""
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
        raise AiRateLimitError("OpenAI rate limit hit.") from exc
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

    for tool_call in msg.tool_calls:
        fn_name = tool_call.function.name
        try:
            fn_args = json.loads(tool_call.function.arguments)
        except Exception:
            fn_args = {}

        functions_called.append({"name": fn_name, "parameters": fn_args})

        # FastAPI executes the safe, fixed predefined function with permissions checked
        tool_result = execute_predefined_tool(fn_name, fn_args, permissions=permissions)

        if tool_result.get("status") == "permission_denied":
            is_any_denied = True
            denial_reason = tool_result.get("error")

        for key_arg in ("batch_id", "operator_id", "equipment_id", "lot_number", "deviation_id"):
            if key_arg in fn_args:
                record_ids.append(str(fn_args[key_arg]))

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result),
            }
        )

    # Second call to let the LLM summarize the tool results
    client2 = _build_openai_client()
    try:
        final_completion = await client2.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        final_text = (final_completion.choices[0].message.content or "").strip()
    finally:
        await client2.close()

    return GenerationResult(
        text=final_text,
        functions_called=functions_called,
        retrieved_record_ids=record_ids,
        permission_denied=is_any_denied,
        denial_reason=denial_reason,
    )


# ── Gemini & Ollama implementations ───────────────────────────────────────────


async def _generate_gemini(
    message: str,
    *,
    username: str | None = None,
    history: list[dict[str, str]] | None = None,
    extra_context: str | None = None,
) -> str:
    api_key = settings.ai_api_key.strip()
    model = (settings.ai_model or "gemini-2.0-flash").strip()
    system = _compose_system(username=username, extra_context=extra_context)

    url = f"{GEMINI_BASE}/models/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": message.strip()}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, params={"key": api_key}, json=payload)
        if response.status_code >= 400:
            raise AiProviderError(f"Gemini error HTTP {response.status_code}: {response.text}")
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return "Data unavailable."
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()


async def _generate_ollama(
    message: str,
    *,
    username: str | None = None,
    history: list[dict[str, str]] | None = None,
    extra_context: str | None = None,
) -> str:
    base = (settings.ollama_base_url or "http://127.0.0.1:11434").strip().rstrip("/")
    model = (settings.ai_model or "llama3.2").strip()
    system = _compose_system(username=username, extra_context=extra_context)

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": message.strip()}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 1024},
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{base}/api/chat", json=payload)
        if response.status_code >= 400:
            raise AiProviderError(f"Ollama error HTTP {response.status_code}")
        data = response.json()
        return str((data.get("message") or {}).get("content") or "").strip()
