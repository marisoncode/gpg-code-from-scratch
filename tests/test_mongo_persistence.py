"""Comprehensive tests for Motor-backed MongoDB persistence layer.

Verifies:
1. Chat threads CRUD operations against MongoDB via Motor.
2. 404 Not Found handling for missing threads and cross-user isolation.
3. Turn appending, title auto-generation, and message flattening.
4. Audit trail append-only persistence to MongoDB and synchronous buffer inspection.
5. Idempotent index creation for (user_id, updated_at) and (user_id, timestamp).
6. Production fail-fast on unreachable MongoDB connection.
7. Development fallback to in-memory storage when MongoDB connection fails.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
import pytest

from app.core.config import settings
from app.core.database import (
    close_mongo,
    get_audit_collection,
    get_threads_collection,
    init_mongo,
    is_mongo_connected,
    set_database,
)
from app.services import audit_service, chat_store


# ── TEST 1: CHAT THREAD CRUD IN MONGODB ───────────────────────────────────────
@pytest.mark.asyncio
async def test_chat_thread_create_and_get() -> None:
    """create_thread persists document in MongoDB and get_thread retrieves it."""
    thread = await chat_store.create_thread(
        user_id="user-qa-1",
        username="QA User",
        title="Batch Investigation",
        greeting="Hello, how can I help?",
    )

    assert thread["id"] is not None
    assert thread["user_id"] == "user-qa-1"
    assert thread["title"] == "Batch Investigation"
    assert thread["greeting"] == "Hello, how can I help?"
    assert len(thread["messages"]) == 1
    assert thread["messages"][0]["role"] == "assistant"
    assert thread["messages"][0]["content"] == "Hello, how can I help?"

    # Retrieve from database
    retrieved = await chat_store.get_thread(thread_id=thread["id"], user_id="user-qa-1")
    assert retrieved["id"] == thread["id"]
    assert retrieved["user_id"] == "user-qa-1"


@pytest.mark.asyncio
async def test_chat_thread_user_isolation_and_404() -> None:
    """get_thread and delete_thread return 404 if thread not found or belongs to another user."""
    thread = await chat_store.create_thread(
        user_id="user-owner",
        username="Owner",
        title="Private Thread",
    )

    # Wrong user cannot access thread (returns 404)
    with pytest.raises(HTTPException) as exc_info:
        await chat_store.get_thread(thread_id=thread["id"], user_id="different-user")
    assert exc_info.value.status_code == 404

    # Non-existent thread returns 404
    with pytest.raises(HTTPException) as exc_info:
        await chat_store.get_thread(thread_id="non-existent-id", user_id="user-owner")
    assert exc_info.value.status_code == 404

    # Wrong user cannot delete thread
    with pytest.raises(HTTPException) as exc_info:
        await chat_store.delete_thread(thread_id=thread["id"], user_id="different-user")
    assert exc_info.value.status_code == 404

    # Owner can delete thread
    await chat_store.delete_thread(thread_id=thread["id"], user_id="user-owner")

    # After deletion, get_thread returns 404
    with pytest.raises(HTTPException) as exc_info:
        await chat_store.get_thread(thread_id=thread["id"], user_id="user-owner")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_chat_thread_append_turn_and_listing() -> None:
    """append_turn updates thread turns, updates title, and list_threads filters empty threads."""
    thread = await chat_store.create_thread(
        user_id="user-turn-test",
        username="Turn Tester",
    )

    # Empty thread is NOT returned in list_threads
    listed_before = await chat_store.list_threads(user_id="user-turn-test")
    assert len(listed_before) == 0

    # Append turn
    updated = await chat_store.append_turn(
        thread_id=thread["id"],
        user_id="user-turn-test",
        request="What is the yield of batch B-101?",
        response="Batch B-101 yield was 98.5%.",
    )

    assert len(updated["turns"]) == 1
    assert updated["title"] == "What is the yield of batch B-101?"
    assert len(updated["messages"]) == 2
    assert updated["messages"][0]["role"] == "user"
    assert updated["messages"][0]["content"] == "What is the yield of batch B-101?"
    assert updated["messages"][1]["role"] == "assistant"
    assert updated["messages"][1]["content"] == "Batch B-101 yield was 98.5%."

    # Now thread is returned in list_threads
    listed_after = await chat_store.list_threads(user_id="user-turn-test")
    assert len(listed_after) == 1
    assert listed_after[0]["id"] == thread["id"]

    # History for model extracts role/content pairs
    history = await chat_store.history_for_model(thread_id=thread["id"], user_id="user-turn-test")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


# ── TEST 2: AUDIT TRAIL PERSISTENCE (APPEND-ONLY) ─────────────────────────────
@pytest.mark.asyncio
async def test_audit_log_entry_persists_to_mongo() -> None:
    """log_audit_entry inserts into MongoDB collection without raw tokens."""
    entry = await audit_service.log_audit_entry(
        user_id="auditor-01",
        role="QA Specialist",
        raw_query="Check deviation DEV-202 with Bearer eyJsecretToken123",
        functions_called=[{"function": "get_deviation_by_id", "arguments": {"deviation_id": "DEV-202"}}],
        retrieved_record_ids=["DEV-202"],
        final_response="Deviation DEV-202 is under review.",
    )

    assert entry["user_id"] == "auditor-01"
    assert "[REDACTED_JWT]" in entry["raw_query"]
    assert "eyJsecretToken123" not in entry["raw_query"]

    # Synchronous inspection returns entry
    recent = audit_service.get_recent_audit_entries(limit=5)
    assert any(e["id"] == entry["id"] for e in recent)

    # Async query directly from MongoDB collection
    async_entries = await audit_service.query_audit_entries_async(user_id="auditor-01", limit=5)
    assert any(e["id"] == entry["id"] for e in async_entries)


# ── TEST 3: INDEX CREATION IDEMPOTENCE ─────────────────────────────────────────
@pytest.mark.asyncio
async def test_init_mongo_creates_indexes_idempotently() -> None:
    """init_mongo creates required compound indexes on threads and audit collections."""
    await init_mongo()

    threads_col = get_threads_collection()
    audit_col = get_audit_collection()

    assert len(threads_col._indexes) >= 1
    assert len(audit_col._indexes) >= 1

    # Running again is idempotent and doesn't fail
    await init_mongo()


# ── TEST 4: PRODUCTION FAIL-FAST ON CONNECTION FAILURE ────────────────────────
@pytest.mark.asyncio
async def test_production_fails_fast_when_mongo_unreachable() -> None:
    """In production mode, failing to connect to MongoDB must raise RuntimeError."""
    set_database(None)

    with patch.object(settings, "environment", "production"), \
         patch.object(settings, "mongo_connection_string", "mongodb://unreachable-host:27017"), \
         patch("app.core.database.AsyncIOMotorClient", side_effect=Exception("Connection refused")):

        with pytest.raises(RuntimeError) as exc_info:
            await init_mongo()

        assert "Failed to connect to MongoDB/Cosmos DB in production" in str(exc_info.value)
        assert not is_mongo_connected()


# ── TEST 5: DEVELOPMENT IN-MEMORY FALLBACK ────────────────────────────────────
@pytest.mark.asyncio
async def test_development_falls_back_to_in_memory_store(caplog: pytest.LogCaptureFixture) -> None:
    """In development mode, connection failure logs loud warning and falls back to in-memory store."""
    set_database(None)

    with patch.object(settings, "environment", "development"), \
         patch.object(settings, "mongo_connection_string", "mongodb://unreachable-host:27017"), \
         patch("app.core.database.AsyncIOMotorClient", side_effect=Exception("Connection timed out")):

        await init_mongo()

        assert not is_mongo_connected()
        assert any("LOUD WARNING" in record.message for record in caplog.records)

        # Chat store operations succeed using in-memory store
        t = await chat_store.create_thread(user_id="dev-user", username="Dev User", title="Dev Thread")
        assert t["id"] is not None
        fetched = await chat_store.get_thread(thread_id=t["id"], user_id="dev-user")
        assert fetched["title"] == "Dev Thread"


# ── TEST 6: AUDIT BUFFER BOUNDEDNESS (NO MEMORY LEAK) ─────────────────────────
@pytest.mark.asyncio
async def test_audit_store_memory_buffer_is_bounded_and_prevents_oom() -> None:
    """_memory_audit_store is a bounded ring buffer that never grows past MAX_MEMORY_AUDIT_ENTRIES."""
    from app.services.audit_service import (
        MAX_MEMORY_AUDIT_ENTRIES,
        _memory_audit_store,
        clear_memory_audit_store,
        get_recent_audit_entries,
        log_audit_entry,
    )

    clear_memory_audit_store()
    assert len(_memory_audit_store) == 0

    # Log more entries than MAX_MEMORY_AUDIT_ENTRIES (e.g., 100 + 25 = 125)
    total_entries = MAX_MEMORY_AUDIT_ENTRIES + 25
    for i in range(total_entries):
        await log_audit_entry(
            user_id=f"user-{i}",
            role="Operator",
            raw_query=f"Query number {i}",
        )

    # Buffer must be strictly capped at MAX_MEMORY_AUDIT_ENTRIES
    assert len(_memory_audit_store) == MAX_MEMORY_AUDIT_ENTRIES
    assert _memory_audit_store.maxlen == MAX_MEMORY_AUDIT_ENTRIES

    # Oldest 25 entries (0..24) must have been evicted FIFO
    stored_queries = [e["raw_query"] for e in _memory_audit_store]
    assert "Query number 0" not in stored_queries
    assert "Query number 24" not in stored_queries
    assert f"Query number {total_entries - 1}" in stored_queries

    # get_recent_audit_entries returns newest first up to limit
    recent_5 = get_recent_audit_entries(limit=5)
    assert len(recent_5) == 5
    assert recent_5[0]["raw_query"] == f"Query number {total_entries - 1}"

