"""Async MongoDB database connection and collection management using Motor.

Provides thread-safe, async connection management, idempotent index initialization,
and production fail-fast / development in-memory fallback semantics.
"""

from __future__ import annotations

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase

from app.core.config import settings

logger = logging.getLogger(__name__)

_mongo_client: AsyncIOMotorClient | None = None
_mongo_db: Any = None
_mongo_connected: bool = False
_indexes_created: bool = False


async def init_mongo() -> None:
    """Initialize MongoDB connection and ensure required indexes exist.

    Fails fast in production if the connection cannot be established.
    Falls back to in-memory store with a loud warning in development.
    """
    global _mongo_client, _mongo_db, _mongo_connected, _indexes_created

    # If a database was already injected (e.g. via test fixture), initialize indexes on it
    if _mongo_db is not None:
        try:
            threads_col = get_threads_collection()
            if threads_col is not None:
                await threads_col.create_index([("user_id", 1), ("updated_at", -1)])
            audit_col = get_audit_collection()
            if audit_col is not None:
                await audit_col.create_index([("user_id", 1), ("timestamp", -1)])
            _indexes_created = True
            _mongo_connected = True
            return
        except Exception as exc:
            logger.warning("Failed to create indexes on injected test database: %s", exc)
            return

    conn_str = (settings.mongo_connection_string or "").strip()
    db_name = (settings.mongo_database_name or "cpg_ai_dev").strip()

    if not conn_str:
        if settings.is_production:
            raise RuntimeError(
                "MONGO_CONNECTION_STRING must be set in production. "
                "A production compliance system requires a persistent backing store."
            )
        _mongo_connected = False
        logger.warning(
            "LOUD WARNING: MONGO_CONNECTION_STRING is not configured. "
            "Falling back to in-memory store for development only. "
            "Chat history and compliance audit logs will NOT persist across restarts!"
        )
        return

    try:
        _mongo_client = AsyncIOMotorClient(conn_str, serverSelectionTimeoutMS=2000)
        # Verify server connectivity via ping command
        await _mongo_client.admin.command("ping")
        _mongo_db = _mongo_client[db_name]
        _mongo_connected = True
        logger.info("Connected successfully to MongoDB database '%s'.", db_name)

        # Idempotently create required indexes per specification:
        # 1. Threads collection indexed on (user_id, updated_at) for list_threads query pattern
        threads_col = _mongo_db[settings.mongo_threads_collection]
        await threads_col.create_index([("user_id", 1), ("updated_at", -1)])

        # 2. Audit collection indexed on (user_id, timestamp) for compliance queries
        audit_col = _mongo_db[settings.mongo_audit_collection]
        await audit_col.create_index([("user_id", 1), ("timestamp", -1)])

        _indexes_created = True
        logger.info(
            "MongoDB indexes verified for '%s' and '%s' collections.",
            settings.mongo_threads_collection,
            settings.mongo_audit_collection,
        )

    except Exception as exc:
        _mongo_connected = False
        _mongo_db = None
        if settings.is_production:
            raise RuntimeError(
                f"Failed to connect to MongoDB/Cosmos DB in production: {exc}. "
                "A production compliance system requires a verified persistent backing store."
            ) from exc

        logger.warning(
            "LOUD WARNING: MongoDB connection failed (%s). "
            "Falling back to in-memory store for development only. "
            "Chat history and compliance audit logs will NOT persist across restarts!",
            exc,
        )


async def close_mongo() -> None:
    """Gracefully close active MongoDB client connections."""
    global _mongo_client, _mongo_db, _mongo_connected, _indexes_created
    if _mongo_client is not None:
        try:
            _mongo_client.close()
        except Exception:
            pass
    _mongo_client = None
    _mongo_db = None
    _mongo_connected = False
    _indexes_created = False


def is_mongo_connected() -> bool:
    """Return True if a persistent MongoDB connection is active."""
    return _mongo_connected and _mongo_db is not None


def get_database() -> Any:
    """Return active MongoDB database instance or None."""
    return _mongo_db


def get_threads_collection() -> Any:
    """Return collection for chat threads."""
    if _mongo_db is not None:
        return _mongo_db[settings.mongo_threads_collection]
    return None


def get_audit_collection() -> Any:
    """Return collection for compliance audit logs."""
    if _mongo_db is not None:
        return _mongo_db[settings.mongo_audit_collection]
    return None


def set_database(db: Any) -> None:
    """Inject a test or mock database instance (used by test fixtures)."""
    global _mongo_db, _mongo_connected
    _mongo_db = db
    _mongo_connected = db is not None

