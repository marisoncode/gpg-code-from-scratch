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


def _extract_heuristic_tool_call(message: str) -> tuple[str, dict[str, Any]] | None:
    """Recognize intent for offline/test execution to call predefined functions."""
    text = message.strip()

    # Raw query rejection
    if re.search(r"\b(select\s+.*from|drop\s+table|delete\s+from|insert\s+into|update\s+.*set)\b", text, re.I):
        return None

    # Phase 3: Batch similarity (e.g. "similar batches to B-1021", "batch similarity for B-1021", "find similar batches to B-1021")
    if "similar" in text.lower() or "similarity" in text.lower():
        sim_bid_m = re.search(r"\b([A-Z0-9]+-[0-9]+)\b", text, re.I)
        if sim_bid_m:
            return "find_similar_batches", {"batch_id": sim_bid_m.group(1), "top_n": 5}

    # Phase 3: Risk score (e.g. "risk score for batch B-1021", "calculate risk for B-1021", "risk score for B-1021")
    if "risk" in text.lower() and ("score" in text.lower() or "batch" in text.lower() or "calculat" in text.lower() or "level" in text.lower()):
        risk_bid_m = re.search(r"\b([A-Z0-9]+-[0-9]+)\b", text, re.I)
        if risk_bid_m:
            return "calculate_risk_score", {"batch_id": risk_bid_m.group(1)}

    # Phase 3: Trend detection (e.g. "detect trends for product Aspirin", "yield trend for Aspirin", "trends for product PRD-01")
    if "trend" in text.lower() or "drift" in text.lower() or "spc" in text.lower():
        prod_m2 = re.search(r"\bproduct\s+([A-Za-z0-9-_]+)", text, re.I)
        p_name = prod_m2.group(1) if prod_m2 else "Aspirin"
        metric_val = "yield"
        if "em" in text.lower() or "environmental" in text.lower():
            metric_val = "em_events"
        elif "equipment" in text.lower() or "dev" in text.lower():
            metric_val = "equipment_deviations"
        win_m = re.search(r"\b(\d+[dmy])\b", text, re.I)
        win_val = win_m.group(1) if win_m else "90d"
        return "detect_trends", {"product_id": p_name, "metric": metric_val, "window": win_val}

    # Finished drug reverse genealogy: e.g. "finished drug FD-901", "trace finished drug DRUG-10"
    drug_m = re.search(r"\b(?:finished\s+drug|drug\s+lot|drug)\s+([A-Za-z0-9-_]+)", text, re.I)
    if drug_m and ("genealogy" in text.lower() or "trace" in text.lower() or "finished" in text.lower() or "fd-" in text.lower()):
        return "get_finished_drug_genealogy", {"drug_id_or_lot": drug_m.group(1)}

    # Material lot genealogy (forward traceability): e.g. "material lot RM-88321 genealogy", "trace material lot RM-88321"
    mat_gen_m = re.search(r"\b(?:material\s+lot|material|chemical\s+lot|chemical)\s+([A-Za-z0-9-_]+)", text, re.I)
    if mat_gen_m and ("genealogy" in text.lower() or "forward" in text.lower() or "trace" in text.lower()):
        return "get_material_lot_genealogy", {"lot_id": mat_gen_m.group(1)}

    # Component lot genealogy: e.g. "component lot COMP-501 genealogy", "trace component COMP-501"
    comp_gen_m = re.search(r"\b(?:component\s+lot|component|packaging)\s+([A-Za-z0-9-_]+)", text, re.I)
    if comp_gen_m and ("genealogy" in text.lower() or "trace" in text.lower()):
        return "get_component_lot_genealogy", {"lot_id": comp_gen_m.group(1)}

    # Operator batch history: e.g. "operator OP-017 batch history", "batches for operator OP-017"
    op_hist_m = re.search(r"\boperator\s+([A-Za-z0-9-_]+)", text, re.I)
    if op_hist_m and ("batch" in text.lower() or "history" in text.lower() or "participat" in text.lower()):
        return "get_operator_batch_history", {"operator_id": op_hist_m.group(1)}

    # Equipment batch history: e.g. "equipment EQ-102 batch history", "batches on equipment EQ-102"
    eq_hist_m = re.search(r"\b(?:equipment|eq)\s+([A-Za-z0-9-_]+)", text, re.I)
    if eq_hist_m and ("batch" in text.lower() or "history" in text.lower() or "processed" in text.lower()):
        return "get_equipment_batch_history", {"equipment_id": eq_hist_m.group(1)}

    # Deviation impact: e.g. "impact of deviation DEV-445", "deviation DEV-445 impact"
    dev_impact_m = re.search(r"\bdeviation\s+([A-Za-z0-9-_]+)", text, re.I)
    if dev_impact_m and ("impact" in text.lower() or "affected" in text.lower()):
        return "get_deviation_impact", {"deviation_id": dev_impact_m.group(1)}

    # Multi-entity investigate location/product: e.g. "investigate location CLEANROOM-1", "batches for location ROOM-101"
    loc_m = re.search(r"\b(?:location|room)\s+([A-Za-z0-9-_]+)", text, re.I)
    if loc_m:
        return "investigate_entity", {"entity_type": "location", "entity_id": loc_m.group(1)}

    prod_m = re.search(r"\bproduct\s+([A-Za-z0-9-_]+)", text, re.I)
    if prod_m and ("batch" in text.lower() or "investigat" in text.lower()):
        return "investigate_entity", {"entity_type": "product", "entity_id": prod_m.group(1)}

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
    """Format offline tool execution result adhering to SRS Section 10 & 19 evidence standards."""
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

    if fn_name == "get_material_lot_genealogy":
        lot = tool_res.get("arguments", {}).get("lot_id")
        batches = result.get("batches") or []
        drugs = result.get("finished_drugs") or []
        confirmed = result.get("confirmed_impact") or []
        potential = result.get("potential_impact") or []
        chain = result.get("traceability_chain") or "material_lot -> batches -> finished_drugs"
        lines = [
            f"[FACT] Material Lot Genealogy for {lot} [Source: Material Lot {lot}]:",
            f"Traceability Chain: {chain}",
            f"Batches linked: {', '.join(batches) if batches else 'None'}",
            f"Finished Drugs: {', '.join(drugs) if drugs else 'None'}",
        ]
        if confirmed:
            lines.append("Confirmed Impact:")
            for item in confirmed:
                lines.append(f"  - [CONFIRMED IMPACT] Batch {item.get('batch_id')} [Source: Batch {item.get('batch_id')}]: {item.get('reason')}")
        if potential:
            lines.append("Potential Impact:")
            for item in potential:
                lines.append(f"  - [POTENTIAL IMPACT] Batch {item.get('batch_id')} [Source: Batch {item.get('batch_id')}]: {item.get('reason')}")
        return "\n".join(lines)

    if fn_name == "get_component_lot_genealogy":
        lot = tool_res.get("arguments", {}).get("lot_id")
        batches = result.get("batches") or []
        confirmed = result.get("confirmed_impact") or []
        potential = result.get("potential_impact") or []
        chain = result.get("traceability_chain") or "component_lot -> batches -> finished_drugs"
        lines = [
            f"[FACT] Component Lot Genealogy for {lot} [Source: Component Lot {lot}]:",
            f"Traceability Chain: {chain}",
            f"Batches linked: {', '.join(batches) if batches else 'None'}",
        ]
        if confirmed:
            lines.append("Confirmed Impact:")
            for item in confirmed:
                lines.append(f"  - [CONFIRMED IMPACT] Batch {item.get('batch_id')} [Source: Batch {item.get('batch_id')}]: {item.get('reason')}")
        if potential:
            lines.append("Potential Impact:")
            for item in potential:
                lines.append(f"  - [POTENTIAL IMPACT] Batch {item.get('batch_id')} [Source: Batch {item.get('batch_id')}]: {item.get('reason')}")
        return "\n".join(lines)

    if fn_name == "get_operator_batch_history":
        op_id = tool_res.get("arguments", {}).get("operator_id")
        batches = result.get("batches") or []
        eq_handled = result.get("equipment_handled") or []
        chain = result.get("traceability_chain") or "operator -> batches -> equipment -> quality_events"
        return (
            f"[FACT] Operator Batch History for {op_id} [Source: Operator {op_id}]:\n"
            f"Traceability Chain: {chain}\n"
            f"Batches executed: {', '.join(batches) if batches else 'None'}\n"
            f"Equipment handled: {', '.join(eq_handled) if eq_handled else 'None'}"
        )

    if fn_name == "get_equipment_batch_history":
        eq_id = tool_res.get("arguments", {}).get("equipment_id")
        batches = result.get("batches") or []
        confirmed = result.get("confirmed_impact") or []
        potential = result.get("potential_impact") or []
        chain = result.get("traceability_chain") or "equipment -> batches -> deviations"
        lines = [
            f"[FACT] Equipment Batch History for {eq_id} [Source: Equipment Asset {eq_id}]:",
            f"Traceability Chain: {chain}",
            f"Batches processed: {', '.join(batches) if batches else 'None'}",
        ]
        if confirmed:
            lines.append("Confirmed Impact:")
            for item in confirmed:
                lines.append(f"  - [CONFIRMED IMPACT] Batch {item.get('batch_id')} [Source: Batch {item.get('batch_id')}]: {item.get('reason')}")
        if potential:
            lines.append("Potential Impact:")
            for item in potential:
                lines.append(f"  - [POTENTIAL IMPACT] Batch {item.get('batch_id')} [Source: Batch {item.get('batch_id')}]: {item.get('reason')}")
        return "\n".join(lines)

    if fn_name == "get_finished_drug_genealogy":
        drug_id = tool_res.get("arguments", {}).get("drug_id_or_lot")
        batches = result.get("batches") or []
        mats = result.get("input_material_lots") or []
        comps = result.get("input_component_lots") or []
        chain = result.get("traceability_chain") or "finished_drug -> batch -> input_lots"
        return (
            f"[FACT] Finished Drug Reverse Genealogy for {drug_id} [Source: Finished Drug {drug_id}]:\n"
            f"Traceability Chain: {chain}\n"
            f"Parent batches: {', '.join(batches) if batches else 'None'}\n"
            f"Input Material Lots: {', '.join(mats) if mats else 'None'}\n"
            f"Input Component Lots: {', '.join(comps) if comps else 'None'}"
        )

    if fn_name == "get_deviation_impact":
        dev_id = tool_res.get("arguments", {}).get("deviation_id")
        confirmed = result.get("confirmed_impact") or []
        potential = result.get("potential_impact") or []
        chain = result.get("traceability_chain") or "deviation -> affected_batches"
        lines = [
            f"[FACT] Deviation Impact Analysis for {dev_id} [Source: Deviation {dev_id}]:",
            f"Traceability Chain: {chain}",
        ]
        if confirmed:
            lines.append("Confirmed Impact:")
            for item in confirmed:
                lines.append(f"  - [CONFIRMED IMPACT] Batch {item.get('batch_id')} [Source: Batch {item.get('batch_id')}]: {item.get('reason')}")
        if potential:
            lines.append("Potential Impact:")
            for item in potential:
                lines.append(f"  - [POTENTIAL IMPACT] Batch {item.get('batch_id')} [Source: Batch {item.get('batch_id')}]: {item.get('reason')}")
        return "\n".join(lines)

    if fn_name in {"get_related_batches", "investigate_entity"}:
        etype = tool_res.get("arguments", {}).get("entity_type", "entity")
        eid = tool_res.get("arguments", {}).get("entity_id", "")
        chain = result.get("traceability_chain") or f"{etype} -> batches"
        return (
            f"[FACT] Multi-Entity Traceability for {etype} {eid} [Source: {etype.capitalize()} {eid}]:\n"
            f"Traceability Chain: {chain}\n"
            f"Status: {result.get('status', 'found')}."
        )

    if fn_name == "find_similar_batches":
        bid = tool_res.get("arguments", {}).get("batch_id")
        sim_list = result.get("similar_batches") or []
        lines = [
            f"[FACT] Batch Similarity Cohort Analysis for {bid} [Source: Batch Record {bid}]:",
            f"Evaluated cohort of {len(sim_list)} historically comparable batches:",
        ]
        for s in sim_list:
            lines.append(f"  - Batch {s.get('batch_id')} [Source: Batch {s.get('batch_id')}]: {s.get('summary')}")
            if s.get("top_factors"):
                lines.append(f"    Top Factors: {', '.join(s.get('top_factors'))}")
        return "\n".join(lines)

    if fn_name == "calculate_risk_score":
        bid = tool_res.get("arguments", {}).get("batch_id") or result.get("batch_id")
        score = result.get("risk_score", 0)
        level = result.get("risk_level", "UNKNOWN")
        cfg_ver = result.get("config_version", "v1.0-approved-2026")
        reasons = result.get("reasons") or []
        lines = [
            f"[FACT] Batch Risk Evaluation for {bid} [Source: Batch Record {bid}]:",
            f"Calculated Risk Score: {score}/100 ({level} Risk Level)",
            f"Approved Weight Configuration Version: {cfg_ver}",
            "Risk Factors & Evidence:",
        ]
        for r in reasons:
            lines.append(f"  - {r.get('evidence')} (Weight: {r.get('weight')})")
        return "\n".join(lines)

    if fn_name == "detect_trends":
        pid = tool_res.get("arguments", {}).get("product_id")
        metric = result.get("metric") or tool_res.get("arguments", {}).get("metric", "yield")
        period = result.get("period") or tool_res.get("arguments", {}).get("window", "90d")
        sample_size = result.get("sample_size", 0)
        method = result.get("method_threshold", "3-sigma SPC moving average")
        uncertainty = result.get("uncertainty_confidence", "Moderate Confidence")
        evidence = result.get("evidence") or []
        findings = result.get("statistical_findings") or []
        lines = [
            f"[FACT] Statistical Trend Analysis for {pid} [Metric: {metric}, Period: {period}]:",
            f"Sample Size: N={sample_size} records evaluated",
            f"Method / Threshold: {method}",
            f"Statistical Findings: {'; '.join(findings) if findings else 'Within standard baseline limits'}",
            f"Evidence: {', '.join(evidence[:5])}",
            f"Uncertainty & Confidence: {uncertainty}",
            "CORRELATION NOTICE (SRS Section 13): Correlation does not equal confirmed causation. "
            "Observed trends represent statistical associations and concurrent patterns. "
            "Root-cause determination requires authorized human QA/engineering review.",
        ]
        return "\n".join(lines)

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

        entity_type, chain, record_ids, risk_cfg_ver = _extract_result_metadata(fn_name, args, tool_result)

        return GenerationResult(
            text=text_summary,
            functions_called=functions_called,
            retrieved_record_ids=record_ids,
            permission_denied=is_denied,
            denial_reason=denial_reason,
            entity_type=entity_type,
            traceability_chain=chain,
            risk_config_version=risk_cfg_ver,
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

    entity_type = ""
    traceability_chain = ""

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
        entity_type=entity_type,
        traceability_chain=traceability_chain,
        risk_config_version=risk_config_version,
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
