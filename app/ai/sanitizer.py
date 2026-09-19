"""Data sanitization layer for downstream API responses.

Purges sensitive authentication tokens, database connection credentials,
internal database metadata (e.g., Cosmos DB system properties), and unnecessary
internal fields before data is exposed to the LLM or frontend clients.
"""

from __future__ import annotations

import re
from typing import Any

# Exact or substring key matches that must never be exposed
_SENSITIVE_KEY_SUBSTRINGS = {
    "password",
    "passwd",
    "pwd",
    "secret",
    "secret_key",
    "client_secret",
    "access_token",
    "refresh_token",
    "jwt",
    "bearer",
    "api_key",
    "apikey",
    "private_key",
    "connection_string",
    "conn_str",
    "db_password",
    "database_url",
    "mongo_uri",
    "sql_connection",
    "credentials",
}

# Internal Cosmos DB and database metadata fields that add no business value
_INTERNAL_METADATA_KEYS = {
    "_rid",
    "_self",
    "_etag",
    "_attachments",
    "_ts",
    "internal_id",
    "system_metadata",
    "stack_trace",
    "raw_token",
}

# Pattern for JWT tokens (3 base64 segments separated by dots)
_JWT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}$")

# Pattern for connection strings
_CONN_STR_PATTERN = re.compile(r"(mongodb(?:\+srv)?|postgres(?:ql)?|mysql|mssql|https?:\/\/[^:]+:[^@]+@)", re.IGNORECASE)


def _is_sensitive_key(key: str) -> bool:
    """Check if a dictionary key represents sensitive or internal data."""
    k_lower = key.strip().lower()
    if k_lower in _INTERNAL_METADATA_KEYS:
        return True
    return any(sub in k_lower for sub in _SENSITIVE_KEY_SUBSTRINGS)


def _sanitize_value(value: Any) -> Any:
    """Sanitize individual string or scalar values."""
    if isinstance(value, str):
        val_strip = value.strip()
        if _JWT_PATTERN.match(val_strip):
            return "[REDACTED_JWT]"
        if _CONN_STR_PATTERN.search(val_strip):
            return "[REDACTED_CONNECTION_STRING]"
        return value
    return value


def sanitize_for_llm(data: Any) -> Any:
    """
    Recursively sanitize data structures before passing to the LLM or API clients.

    - Removes keys matching sensitive patterns (passwords, secrets, tokens).
    - Removes internal Cosmos DB / database metadata (_rid, _etag, _self).
    - Redacts sensitive strings (JWTs, connection strings).
    - Preserves all legitimate business fields.
    """
    if isinstance(data, dict):
        sanitized_dict: dict[str, Any] = {}
        for k, v in data.items():
            if _is_sensitive_key(k):
                continue
            sanitized_dict[k] = sanitize_for_llm(v)
        return sanitized_dict

    if isinstance(data, list):
        return [sanitize_for_llm(item) for item in data]

    if isinstance(data, tuple):
        return tuple(sanitize_for_llm(item) for item in data)

    if isinstance(data, set):
        return {sanitize_for_llm(item) for item in data}

    return _sanitize_value(data)

