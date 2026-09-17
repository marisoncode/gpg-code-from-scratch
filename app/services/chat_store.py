"""Thread and message storage for the chatbot.

Maintains thread document structure with embedded request/response turns
backed by MongoDB/Cosmos DB via Motor, with transparent in-memory fallback for local dev.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, status

from app.core.database import get_threads_collection, is_mongo_connected

logger = logging.getLogger(__name__)

# In-memory store for chat threads (fallback for local development)
_threads_memory_store: dict[str, dict[str, Any]] = {}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _flatten_messages(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand embedded turns into role/content rows for UI and history."""
    thread_id = str(doc.get("id") or doc.get("_id") or "")
    out: list[dict[str, Any]] = []

    greeting = (doc.get("greeting") or "").strip()
    if greeting:
        out.append(
            {
                "id": f"{thread_id}:greeting",
                "thread_id": thread_id,
                "role": "assistant",
                "content": greeting,
                "created_at": doc.get("created_at"),
            }
        )

    for turn in doc.get("turns") or []:
        turn_id = str(turn.get("id") or uuid4())
        created = turn.get("created_at")
        request = (turn.get("request") or "").strip()
        response = (turn.get("response") or "").strip()
        if request:
            out.append(
                {
                    "id": f"{turn_id}:request",
                    "thread_id": thread_id,
                    "role": "user",
                    "content": request,
                    "created_at": created,
                }
            )
        if response:
            out.append(
                {
                    "id": f"{turn_id}:response",
                    "thread_id": thread_id,
                    "role": "assistant",
                    "content": response,
                    "created_at": created,
                }
            )
    return out


def _serialize_thread(doc: dict[str, Any], *, include_messages: bool = False) -> dict[str, Any]:
    tid = str(doc.get("id") or doc.get("_id") or "")
    payload: dict[str, Any] = {
        "id": tid,
        "user_id": doc.get("user_id") or "",
        "username": doc.get("username") or "",
        "title": doc.get("title") or "Support chat",
        "greeting": doc.get("greeting") or "",
        "turns": doc.get("turns") or [],
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }
    if include_messages:
        payload["messages"] = _flatten_messages(doc)
    return payload


async def create_thread(
    *,
    user_id: str,
    username: str,
    title: str | None = None,
    greeting: str | None = None,
) -> dict[str, Any]:
    now = _utcnow()
    thread_id = str(uuid4())
    doc: dict[str, Any] = {
        "id": thread_id,
        "_id": thread_id,
        "user_id": user_id,
        "username": username,
        "title": (title or "Support chat").strip() or "Support chat",
        "greeting": (greeting or "").strip(),
        "turns": [],
        "created_at": now,
        "updated_at": now,
    }

    if is_mongo_connected():
        col = get_threads_collection()
        if col is not None:
            await col.insert_one(dict(doc))
    else:
        _threads_memory_store[thread_id] = doc

    return _serialize_thread(doc, include_messages=True)


async def list_threads(*, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    if is_mongo_connected():
        col = get_threads_collection()
        if col is not None:
            cursor = col.find(
                {"user_id": user_id, "turns.0": {"$exists": True}}
            ).sort("updated_at", -1)
            docs = await cursor.to_list(length=limit)
            return [_serialize_thread(t) for t in docs]

    user_threads = [
        t for t in _threads_memory_store.values()
        if t.get("user_id") == user_id and len(t.get("turns", [])) > 0
    ]
    user_threads.sort(key=lambda t: str(t.get("updated_at", "")), reverse=True)
    return [_serialize_thread(t) for t in user_threads[:limit]]


async def get_thread(*, thread_id: str, user_id: str) -> dict[str, Any]:
    if is_mongo_connected():
        col = get_threads_collection()
        if col is not None:
            doc = await col.find_one({"$or": [{"id": thread_id}, {"_id": thread_id}]})
            if doc:
                if doc.get("user_id") == user_id:
                    return _serialize_thread(doc, include_messages=True)
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")

    if thread_id in _threads_memory_store:
        doc = _threads_memory_store[thread_id]
        if doc.get("user_id") == user_id:
            return _serialize_thread(doc, include_messages=True)

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")


async def get_or_create_active_thread(*, user_id: str, username: str) -> dict[str, Any]:
    threads = await list_threads(user_id=user_id, limit=1)
    if threads:
        return await get_thread(thread_id=threads[0]["id"], user_id=user_id)
    return await create_thread(user_id=user_id, username=username)


async def delete_thread(*, thread_id: str, user_id: str) -> None:
    if is_mongo_connected():
        col = get_threads_collection()
        if col is not None:
            res = await col.delete_one({"$or": [{"id": thread_id}, {"_id": thread_id}], "user_id": user_id})
            if res.deleted_count > 0:
                return
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")

    if thread_id in _threads_memory_store and _threads_memory_store[thread_id].get("user_id") == user_id:
        del _threads_memory_store[thread_id]
        return

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")


async def set_greeting(*, thread_id: str, user_id: str, greeting: str) -> dict[str, Any]:
    now = _utcnow()
    clean_greeting = greeting.strip()

    if is_mongo_connected():
        col = get_threads_collection()
        if col is not None:
            await get_thread(thread_id=thread_id, user_id=user_id)
            await col.update_one(
                {"$or": [{"id": thread_id}, {"_id": thread_id}], "user_id": user_id},
                {"$set": {"greeting": clean_greeting, "updated_at": now}},
            )
            doc = await col.find_one({"$or": [{"id": thread_id}, {"_id": thread_id}]})
            if doc:
                return _serialize_thread(doc, include_messages=True)

    thread = await get_thread(thread_id=thread_id, user_id=user_id)
    doc = _threads_memory_store.get(thread_id, thread)
    doc["greeting"] = clean_greeting
    doc["updated_at"] = now
    _threads_memory_store[thread_id] = doc
    return _serialize_thread(doc, include_messages=True)


async def append_turn(
    *,
    thread_id: str,
    user_id: str,
    request: str,
    response: str,
) -> dict[str, Any]:
    """Persist one user request + AI response as an embedded turn."""
    now = _utcnow()
    req = request.strip()
    res = response.strip()
    turn = {
        "id": str(uuid4()),
        "request": req,
        "response": res,
        "created_at": now,
    }

    if is_mongo_connected():
        col = get_threads_collection()
        if col is not None:
            thread = await get_thread(thread_id=thread_id, user_id=user_id)
            update_fields: dict[str, Any] = {"updated_at": now}
            if req and (not thread.get("title") or thread.get("title") == "Support chat"):
                update_fields["title"] = req[:80]

            await col.update_one(
                {"$or": [{"id": thread_id}, {"_id": thread_id}], "user_id": user_id},
                {"$push": {"turns": turn}, "$set": update_fields},
            )
            doc = await col.find_one({"$or": [{"id": thread_id}, {"_id": thread_id}]})
            if doc:
                return _serialize_thread(doc, include_messages=True)

    await get_thread(thread_id=thread_id, user_id=user_id)
    doc = _threads_memory_store.get(thread_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")

    doc.setdefault("turns", []).append(turn)
    doc["updated_at"] = now
    if req and (not doc.get("title") or doc.get("title") == "Support chat"):
        doc["title"] = req[:80]

    _threads_memory_store[thread_id] = doc
    return _serialize_thread(doc, include_messages=True)


async def history_for_model(*, thread_id: str, user_id: str) -> list[dict[str, str]]:
    thread = await get_thread(thread_id=thread_id, user_id=user_id)
    return [
        {"role": m["role"], "content": m["content"]}
        for m in thread.get("messages") or []
        if m.get("content")
    ]


async def list_messages(*, thread_id: str, user_id: str) -> list[dict[str, Any]]:
    thread = await get_thread(thread_id=thread_id, user_id=user_id)
    return list(thread.get("messages") or [])
