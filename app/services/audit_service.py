"""Audit trail service for CPG AI compliance logging.

Uses Azure Cosmos DB and COSMOS_DB_WRITE_KEY to write to COSMOS_DB_AUDIT_CONTAINER.
The write-scoped credential is NEVER used against business data containers.
Never logs raw JWTs or secrets.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from azure.cosmos import CosmosClient

from app.core.config import settings

logger = logging.getLogger(__name__)

# Containers permitted for write-scoped credential
_ALLOWED_WRITE_CONTAINERS = {"audit_trail", "chat_history"}


class AuditSecurityError(Exception):
    """Raised when write-scoped operations violate container or credential constraints."""


_client_override: CosmosClient | None = None
# In-memory test store for when running tests or if Cosmos client is configured with mock
_memory_audit_store: list[dict[str, Any]] = []


def set_cosmos_write_client(client: CosmosClient | None) -> None:
    """Inject test CosmosClient for integration tests."""
    global _client_override
    _client_override = client


def get_write_cosmos_client() -> CosmosClient:
    """
    Returns CosmosClient configured strictly with COSMOS_DB_WRITE_KEY.
    Ensures that COSMOS_DB_READONLY_KEY is not used for audit logging.
    """
    global _client_override
    if _client_override is not None:
        return _client_override

    endpoint = (settings.cosmos_db_endpoint or "").strip()
    key = (settings.cosmos_db_write_key or "").strip()

    if not endpoint:
        raise ValueError("COSMOS_DB_ENDPOINT is not configured.")
    if not key:
        raise ValueError("COSMOS_DB_WRITE_KEY is not configured.")

    if settings.cosmos_db_readonly_key and key == settings.cosmos_db_readonly_key:
        raise AuditSecurityError(
            "Security violation: COSMOS_DB_READONLY_KEY cannot be used for audit trail / chat history writes."
        )

    return CosmosClient(endpoint, credential=key)


def get_write_container(container_name: str):
    """
    Get container proxy using write key.
    Strictly prohibits targeting business data containers (batches, equipment, etc.).
    """
    c_name = container_name.strip()
    allowed = {settings.cosmos_db_audit_container, settings.cosmos_db_chat_container, *_ALLOWED_WRITE_CONTAINERS}
    if c_name not in allowed:
        raise AuditSecurityError(
            f"Security violation: Write-scoped credential is not allowed to access container '{c_name}'. "
            f"Write access is strictly limited to audit and chat history containers."
        )

    client = get_write_cosmos_client()
    db = client.get_database_client(settings.cosmos_db_database)
    return db.get_container_client(c_name)


def sanitize_secrets(text: str) -> str:
    """Redact any potential bearer tokens or raw secrets from audit messages."""
    if not text:
        return ""
    import re

    # Redact Bearer tokens if accidentally included in user input
    cleaned = re.sub(r"Bearer\s+[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*", "[REDACTED_JWT]", text)
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
) -> dict[str, Any]:
    """
    Log an immutable audit trail entry for a user request.
    Includes user ID, role, timestamp, raw query, functions called + parameters,
    retrieved record IDs, model/version, final response, permission denial status,
    and Phase 2 multi-entity investigation fields (entity_type, traceability_chain).
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
    }

    # Store in memory for testing/fallback
    _memory_audit_store.append(entry)

    try:
        container = get_write_container(settings.cosmos_db_audit_container)
        container.create_item(body=entry)
    except Exception as exc:
        logger.warning(f"Failed writing audit entry to Cosmos DB container: {exc}")

    return entry


def get_recent_audit_entries(limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve audit entries from Cosmos DB or memory store (for inspection/testing)."""
    try:
        container = get_write_container(settings.cosmos_db_audit_container)
        query = f"SELECT TOP {limit} * FROM c ORDER BY c.timestamp DESC"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        if items:
            return items
    except Exception as exc:
        logger.warning(f"Could not read audit entries from Cosmos DB: {exc}")

    # Return from in-memory fallback
    return list(reversed(_memory_audit_store[-limit:]))

