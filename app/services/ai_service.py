"""Chat generation for CPG AI Assistant using OpenAI (GPT-4o-mini).

Wires LLM function calling to predefined, read-only CPG functions.
The LLM never constructs or executes raw database queries.
Enforces SRS Section 10 evidence standards and captures audit metadata.
Uses the real API key strictly loaded from the environment (.env).
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any

import httpx
from openai import APIError, AsyncAzureOpenAI, AsyncOpenAI, AuthenticationError, RateLimitError

from app.agent.prompts import SYSTEM_PROMPT
from app.ai import (
    PREDEFINED_TOOL_SCHEMAS,
    execute_predefined_tool,
)
from app.core.config import settings
from app.services.capabilities import ChatPermissions

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
        data_source: str = "",
    ) -> None:
        self.text = text
        self.functions_called = functions_called or []
        self.retrieved_record_ids = retrieved_record_ids or []
        self.permission_denied = permission_denied
        self.denial_reason = denial_reason
        self.entity_type = entity_type
        self.traceability_chain = traceability_chain
        self.risk_config_version = risk_config_version
        self.data_source = data_source

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


def _is_conversational_greeting(message: str) -> bool:
    """Detect if a user input is a pure greeting or capability check rather than a domain query."""
    cleaned = message.strip().lower().rstrip("!.,? ")
    greetings = {
        "hi", "hello", "hey", "hiya", "howdy", "good morning", "good afternoon",
        "good evening", "greetings", "who are you", "what can you do", "help",
    }
    if cleaned in greetings:
        return True
    if any(cleaned.startswith(f"{g} ") for g in ("hi", "hello", "hey", "good morning", "good afternoon", "good evening")):
        domain_keywords = ("batch", "lot", "deviation", "equipment", "material", "kpi", "dashboard", "training", "operator", "oos", "oot", "investigat", "recommend", "trace")
        if not any(k in cleaned for k in domain_keywords):
            return True
    return False


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

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    # Pass tools only when query is not a pure greeting / introductory query
    if not _is_conversational_greeting(message):
        create_kwargs["tools"] = PREDEFINED_TOOL_SCHEMAS

    functions_called: list[dict[str, Any]] = []
    record_ids: list[str] = []
    is_any_denied = False
    denial_reason = None
    data_source = ""
    last_tool_data = None
    completion = None

    try:
        completion = await client.chat.completions.create(**create_kwargs)
    except AuthenticationError as exc:
        raise AiAuthError(f"OpenAI authentication failed: {exc}") from exc
    except RateLimitError as exc:
        STOP_WORDS = {"record", "records", "details", "status", "info", "information", "summary", "dossier", "data", "es", "the", "a", "an", "is", "for", "number", "no", "id", "batch", "batches", "active"}
        batch_candidates = re.findall(r'(?:batch|lot|record)(?:\s+(?:number|no|id|record|details))*[\s#:]*([A-Za-z0-9_-]+)', message, re.IGNORECASE)
        b_id = None
        for cand in batch_candidates:
            cand_clean = cand.strip().strip("#:.,")
            if cand_clean.lower() not in STOP_WORDS and len(cand_clean) >= 3 and any(c.isdigit() for c in cand_clean):
                b_id = cand_clean
                break

        if not b_id:
            standalone = re.findall(r'\b(\d{5,12}|[A-Za-z]{1,4}-\d{3,8}|[a-z_]+_\d{3,8})\b', message, re.IGNORECASE)
            for cand in standalone:
                cand_clean = cand.strip()
                if cand_clean.lower() not in STOP_WORDS and any(c.isdigit() for c in cand_clean):
                    b_id = cand_clean
                    break

        if b_id:
            tool_res = await execute_predefined_tool("get_batch_by_id", {"batch_id": b_id}, permissions=permissions, token=token)
            functions_called.append({"name": "get_batch_by_id", "parameters": {"batch_id": b_id}})
            last_tool_data = tool_res
            data_source = "production-api.cpguardian.com"
        elif any(w in message.lower() for w in ("production", "throughput", "dashboard", "line", "batch", "batches")):
            tool_res = await execute_predefined_tool("get_production_dashboard_data", {}, permissions=permissions, token=token)
            functions_called.append({"name": "get_production_dashboard_data", "parameters": {}})
            last_tool_data = tool_res
            data_source = "facility-user-api.cpguardian.com"
        elif any(w in message.lower() for w in ("compliance", "deviation", "qa", "calibration")):
            tool_res = await execute_predefined_tool("get_compliance_dashboard_data", {}, permissions=permissions, token=token)
            functions_called.append({"name": "get_compliance_dashboard_data", "parameters": {}})
            last_tool_data = tool_res
            data_source = "compliance-api.cpguardian.com"
        else:
            raise AiRateLimitError(f"OpenAI rate limit / quota exceeded: {exc}") from exc
    except Exception as exc:
        raise AiProviderError(f"OpenAI request failed: {exc}") from exc
    finally:
        await client.close()

    entity_type = ""
    traceability_chain = ""
    risk_config_version = ""
    final_text = ""

    if completion is not None:
        choice = completion.choices[0]
        msg = choice.message

        # Check if the LLM requested a function call
        if not msg.tool_calls:
            return GenerationResult(text=(msg.content or "").strip())

        # Build assistant message for OpenAI / Gemini compatibility
        tool_calls_payload = []
        for tc in msg.tool_calls:
            tc_dict: dict[str, Any] = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            if hasattr(tc, "extra_content") and tc.extra_content:
                tc_dict["extra_content"] = tc.extra_content
            tool_calls_payload.append(tc_dict)

        assistant_msg = {
            "role": "assistant",
            "content": msg.content or None,
            "tool_calls": tool_calls_payload,
        }
        messages.append(assistant_msg)

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except Exception:
                fn_args = {}

            functions_called.append({"name": fn_name, "parameters": fn_args})

            # Execute safe predefined function with permissions and token forwarded
            res = execute_predefined_tool(fn_name, fn_args, permissions=permissions, token=token)
            if inspect.isawaitable(res):
                tool_result = await res
            else:
                tool_result = res

            last_tool_data = tool_result
            res_data = tool_result.get("result") or {}
            if isinstance(res_data, dict) and res_data.get("data_source"):
                data_source = str(res_data.get("data_source"))
                functions_called[-1]["data_source"] = data_source

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
            logger.warning("LLM second pass hit rate limit / quota, falling back to deterministic formatter: %s", exc)
        except Exception as exc:
            logger.warning("LLM second pass call failed: %s", exc)
        finally:
            await client2.close()

    # Guarantee: If LLM returned empty string, hit rate limit, or had a parsing glitch, format the real data directly
    if not final_text and last_tool_data:
        # Unpack executor envelope if present: {"function": ..., "result": ...}
        tool_payload = last_tool_data
        if isinstance(tool_payload, dict) and "result" in tool_payload and isinstance(tool_payload["result"], dict):
            tool_payload = tool_payload["result"]

        # 1. Check if tool returned not found or error
        if isinstance(tool_payload, dict) and (
            tool_payload.get("found") is False
            or tool_payload.get("status") == "not_found"
            or "error" in tool_payload
        ):
            target = tool_payload.get("record_id") or "requested"
            err_detail = tool_payload.get("error") or f"Batch '{target}' was not found in the production database."
            final_text = (
                f"### Batch Record Not Found\n\n"
                f"{err_detail}\n\n"
                f"- **Status:** `Not Found`\n"
                f"- **Queried Microservice:** `https://production-api.cpguardian.com/api/BatchRecord`\n\n"
                f"Please verify the Batch Number or Lot Number and confirm it exists in the active tenant."
            )
        else:
            # 2. Extract actual batch dictionary
            raw_info = tool_payload.get("batch") if isinstance(tool_payload, dict) and tool_payload.get("batch") is not None else tool_payload
            if isinstance(raw_info, dict) and "data" in raw_info and isinstance(raw_info["data"], (dict, list)):
                raw_info = raw_info["data"]
            if isinstance(raw_info, list) and len(raw_info) > 0 and isinstance(raw_info[0], dict):
                raw_info = raw_info[0]

            # Check if this is a list of batches or records (e.g. get_batches)
            records_list = None
            if isinstance(tool_payload, list):
                records_list = tool_payload
            elif isinstance(tool_payload, dict):
                for lk in ("batches", "batch_list", "items", "records", "training_records", "deviations"):
                    if isinstance(tool_payload.get(lk), list):
                        records_list = tool_payload[lk]
                        break

            if isinstance(raw_info, dict) and any(k in raw_info for k in ("batch_number", "lot_number", "product", "batch_name", "Lot_number", "Batch_name", "id")):
                b_num = raw_info.get("batch_number") or raw_info.get("lot_number") or raw_info.get("Lot_number") or raw_info.get("id") or "N/A"
                prod = raw_info.get("product") or raw_info.get("product_name") or raw_info.get("batch_name") or raw_info.get("Batch_name") or "N/A"
                stat = raw_info.get("status") or raw_info.get("Status") or "Active"
                b_size = raw_info.get("batch_size") or raw_info.get("Units_container") or raw_info.get("units_container") or raw_info.get("size") or "N/A"
                exp = raw_info.get("expiration_date") or raw_info.get("expiry_date") or raw_info.get("Expiry_date") or "N/A"
                mfr = raw_info.get("mfr_name") or raw_info.get("master_formula") or raw_info.get("Master_formula") or "N/A"
                operator = raw_info.get("operator_name") or raw_info.get("Requestor_name") or raw_info.get("requestor_name") or raw_info.get("operator_id") or "N/A"
                equip = raw_info.get("equipment_id") or raw_info.get("Equipment_id") or "N/A"
                sched = raw_info.get("scheduled_date") or raw_info.get("batch_date") or raw_info.get("Batch_date") or raw_info.get("created_date") or "N/A"
                guid = raw_info.get("id") or raw_info.get("record_id") or "N/A"

                final_text = (
                    f"### Batch Record Dossier: **{b_num}**\n\n"
                    f"| Specification / Parameter | Value |\n"
                    f"| :--- | :--- |\n"
                    f"| **Batch / Lot #** | `{b_num}` |\n"
                    f"| **Product / Batch Name** | **{prod}** |\n"
                    f"| **Operational Status** | `{stat}` |\n"
                    f"| **Batch Size / Units** | {b_size} units |\n"
                    f"| **Master Formula (MFR)** | `{mfr}` |\n"
                    f"| **Requestor / Operator** | {operator} |\n"
                    f"| **Production Date** | {sched} |\n"
                    f"| **Expiration Date** | {exp} |\n"
                    f"| **Record Identifier (GUID)** | `{guid}` |\n"
                )
                if equip != "N/A":
                    final_text += f"| **Assigned Equipment** | `{equip}` |\n"

                if raw_info.get("chemical_components") and isinstance(raw_info["chemical_components"], list):
                    final_text += "\n#### Dispensed Chemical Components\n\n"
                    final_text += "| Component | Lot Number | Quantity | Unit |\n| :--- | :--- | :--- | :--- |\n"
                    for comp in raw_info["chemical_components"]:
                        if isinstance(comp, dict):
                            c_name = comp.get("chemical_name") or comp.get("name") or "-"
                            c_lot = comp.get("lot_number") or comp.get("lot") or "-"
                            c_qty = comp.get("quantity_dispensed") or comp.get("quantity") or "-"
                            c_unit = comp.get("unit") or "-"
                            final_text += f"| {c_name} | `{c_lot}` | {c_qty} | {c_unit} |\n"

                final_text += (
                    f"\n**Suggested Actions:**\n"
                    f"- Trace raw materials & genealogy: `Trace genealogy for batch {b_num}`\n"
                    f"- Verify QA deviations: `Check open deviations for batch {b_num}`\n"
                    f"- Check equipment status: `What equipment was used for {b_num}?`\n"
                )
            elif isinstance(raw_info, dict) and any(k in raw_info for k in ("totalBatch", "productionBatch", "releasedBatch", "holdBatch")):
                tot = raw_info.get("totalBatch", 0)
                prod = raw_info.get("productionBatch", 0)
                rel = raw_info.get("releasedBatch", 0)
                hld = raw_info.get("holdBatch", 0)
                pend = raw_info.get("pendingBatch", 0)
                final_text = (
                    f"### Production Operations Dashboard\n\n"
                    f"| Manufacturing Metric / KPI | Current Count | Operational Status |\n"
                    f"| :--- | :--- | :--- |\n"
                    f"| **Total Facility Batches** | **{tot}** | `Active Monitoring` |\n"
                    f"| **Active in Production** | **{prod}** | `Production` |\n"
                    f"| **Released (QA Passed)** | **{rel}** | `Approved` |\n"
                    f"| **On QA Hold** | **{hld}** | `Quarantine` |\n"
                    f"| **Pending Initiation** | **{pend}** | `Draft` |\n\n"
                    f"**Suggested Inquiries:**\n"
                    f"- *\"What are the {hld} batches currently on QA hold?\"*\n"
                    f"- *\"Give me details on the active production batches\"*\n"
                    f"- *\"Summarize QA compliance and open deviations\"*\n"
                )
            elif records_list is not None and len(records_list) > 1:
                final_text = f"### Retrieved Records ({len(records_list)} found)\n\n"
                final_text += "| Batch / Identifier | Name / Description | Status | Additional Details |\n"
                final_text += "| :--- | :--- | :--- | :--- |\n"
                for r in records_list[:25]:
                    if isinstance(r, dict):
                        b_id = r.get("lot_number") or r.get("Lot_number") or r.get("batch_number") or r.get("id") or "-"
                        b_name = r.get("batch_name") or r.get("Batch_name") or r.get("product") or r.get("name") or "-"
                        b_stat = r.get("status") or r.get("Status") or "Active"
                        b_extra = r.get("mfr_name") or r.get("Master_formula") or r.get("date") or r.get("Units_container") or "-"
                        final_text += f"| `{b_id}` | **{b_name}** | `{b_stat}` | {b_extra} |\n"
                    else:
                        final_text += f"| - | {str(r)} | - | - |\n"
            elif isinstance(raw_info, dict):
                # Clean key-value bullet points
                final_text = "### Record Details\n\n"
                for k, v in raw_info.items():
                    if k.lower() in ("id", "raw_token", "_rid", "_self", "_etag"):
                        continue
                    clean_k = k.replace("_", " ").title()
                    final_text += f"- **{clean_k}:** {v}\n"
            else:
                final_text = "No batch record details could be retrieved."

    return GenerationResult(
        text=final_text or "No details could be retrieved for this query.",
        functions_called=functions_called,
        retrieved_record_ids=record_ids,
        permission_denied=is_any_denied,
        denial_reason=denial_reason,
        entity_type=entity_type,
        traceability_chain=traceability_chain,
        risk_config_version=risk_config_version,
        data_source=data_source,
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
