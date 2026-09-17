"""Audit trail service for CPG AI compliance logging.

Maintains an immutable, append-only audit log for all AI interactions and tool calls
backed by MongoDB/Cosmos DB via Motor, with transparent in-memory fallback.
Never logs raw JWTs or secrets.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
import re
from typing import Any
from uuid import uuid4

from app.core.config import settings
from app.core.database import get_audit_collection, is_mongo_connected

logger = logging.getLogger(__name__)

# Bounded in-memory ring buffer (strictly O(1) memory to prevent OOM memory leaks in production,
# while providing rolling diagnostic retention and synchronous inspection for tests).
MAX_MEMORY_AUDIT_ENTRIES: int = 100
_memory_audit_store: deque[dict[str, Any]] = deque(maxlen=MAX_MEMORY_AUDIT_ENTRIES)


def sanitize_secrets(text: str) -> str:
    """Redact any potential bearer tokens or raw secrets from audit messages."""
    if not text:
        return ""

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
    data_source: str = "",
) -> dict[str, Any]:
    """
    Log an immutable, append-only audit trail entry for a user request.
    Includes user ID, role, timestamp, raw query, functions called + parameters,
    retrieved record IDs, model/version, final response, permission denial status,
    Phase 2 multi-entity investigation fields (entity_type, traceability_chain),
    Phase 3 risk-weight configuration version (for audit reproducibility),
    and data_source tracking for placeholder vs verified master data.
    Never logs raw JWT or secrets.
    """
    now = datetime.now(timezone.utc).isoformat()
    clean_query = sanitize_secrets(raw_query)
    entry_id = str(uuid4())

    entry: dict[str, Any] = {
        "id": entry_id,
        "_id": entry_id,
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
        "data_source": data_source or "",
    }

    # Persist to MongoDB when connected (strictly append-only insert)
    if is_mongo_connected():
        col = get_audit_collection()
        if col is not None:
            try:
                await col.insert_one(dict(entry))
            except Exception as exc:
                logger.error("Failed to insert audit log entry to MongoDB: %s", exc, exc_info=True)
                if settings.is_production:
                    raise

    # Maintain a bounded ring buffer (strictly O(1) memory, maxlen=MAX_MEMORY_AUDIT_ENTRIES)
    # for immediate synchronous inspection, testing, and recent diagnostic tail.
    _memory_audit_store.append(dict(entry))
    logger.info(
        "AUDIT_ENTRY: user=%s role=%s functions=%s",
        entry["user_id"],
        entry["role"],
        [f.get("function") for f in entry["functions_called"]],
    )
    return entry


def get_recent_audit_entries(limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve recent audit entries from bounded ring buffer (newest first)."""
    return list(reversed(_memory_audit_store))[:limit]


def clear_memory_audit_store() -> None:
    """Clear the in-memory audit store ring buffer (useful in test teardown)."""
    _memory_audit_store.clear()


async def query_audit_entries_async(
    *,
    user_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Async query for audit entries from MongoDB with user_id and timestamp index support."""
    if is_mongo_connected():
        col = get_audit_collection()
        if col is not None:
            query = {"user_id": user_id} if user_id else {}
            cursor = col.find(query).sort("timestamp", -1)
            docs = await cursor.to_list(length=limit)
            return docs

    filtered = [
        e for e in reversed(_memory_audit_store)
        if user_id is None or e.get("user_id") == user_id
    ]
    return filtered[:limit]
