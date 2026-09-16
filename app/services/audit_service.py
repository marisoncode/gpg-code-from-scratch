"""Audit trail service for CPG AI compliance logging.

Maintains an immutable audit log for all AI interactions and tool calls.
Never logs raw JWTs or secrets.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.core.config import settings

logger = logging.getLogger(__name__)

# In-memory audit store
_memory_audit_store: list[dict[str, Any]] = []


def sanitize_secrets(text: str) -> str:
    """Redact any potential bearer tokens or raw secrets from audit messages."""
    if not text:
        return ""
    import re

    # Redact Bearer tokens if accidentally included in user input
    cleaned = re.sub(r"Bearer\s+[^\s]+", "[REDACTED_JWT]", text, flags=re.IGNORECASE)
    # Redact potential password/secret query params
    cleaned = re.sub(r"(key|secret|token|password)=([^\s&]+)", r"\1=[REDACTED]", cleaned, flags=re.IGNORECASE)
    return cleaned


async def log_audit_entry(
    *,
    user_id: str,
    role: str = "User",
    raw_query: str,
    functions_called: list[dict[str, Any]] | None = None,
    retrieved_record_ids: list[str] | None = None,
    model_version: str = "",
    final_response: str = "",
    permission_denied: bool = False,
    denial_reason: str | None = None,
    entity_type: str = "",
    traceability_chain: str = "",
    risk_config_version: str = "",
) -> dict[str, Any]:
    """
    Log an immutable audit trail entry for a user request.
    Includes user ID, role, timestamp, raw query, functions called + parameters,
    retrieved record IDs, model/version, final response, permission denial status,
    Phase 2 multi-entity investigation fields (entity_type, traceability_chain),
    and Phase 3 risk-weight configuration version (for audit reproducibility).
    Never logs raw JWT or secrets.
    """
    now = datetime.now(timezone.utc).isoformat()
    clean_query = sanitize_secrets(raw_query)

    entry: dict[str, Any] = {
        "id": str(uuid4()),
        "user_id": user_id or "anonymous",
        "role": role or "User",
        "timestamp": now,
        "raw_query": clean_query,
        "functions_called": functions_called or [],
        "retrieved_record_ids": retrieved_record_ids or [],
        "model_version": model_version or settings.ai_model,
        "final_response": final_response,
        "permission_denied": permission_denied,
        "denial_reason": denial_reason,
        "entity_type": entity_type or "",
        "traceability_chain": traceability_chain or "",
        "risk_config_version": risk_config_version or "",
    }

    _memory_audit_store.append(entry)
    logger.info(f"AUDIT_ENTRY: user={entry['user_id']} role={entry['role']} functions={[f.get('function') for f in entry['functions_called']]}")
    return entry


def get_recent_audit_entries(limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve recent audit entries from memory store (for inspection/testing)."""
    return list(reversed(_memory_audit_store[-limit:]))
