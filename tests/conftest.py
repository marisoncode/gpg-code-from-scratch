"""
Pytest configuration and external API mocks for offline test isolation.
Ensures unit tests test the full application logic, serialization, and tool-calling loop
without requiring an active paid OpenAI quota or external network connection.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from app.services.recommendation_engine import ADVISORY_STATUS_LABEL


class MockToolCallFunction:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = json.dumps(arguments)


class MockToolCall:
    def __init__(self, tool_id: str, name: str, arguments: dict):
        self.id = tool_id
        self.function = MockToolCallFunction(name, arguments)


class MockMessage:
    def __init__(self, content: str | None = None, tool_calls: list[MockToolCall] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.role = "assistant"

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        return d


class MockChoice:
    def __init__(self, message: MockMessage):
        self.message = message


class MockCompletion:
    def __init__(self, message: MockMessage):
        self.choices = [MockChoice(message)]
        self.usage = MagicMock(total_tokens=100)


@pytest.fixture(autouse=True)
def mock_openai_client_for_tests():
    """
    Autouse fixture that intercepts AsyncOpenAI client in tests so test suite
    can run offline without requiring paid OpenAI credits, while testing
    the real tool-calling loop in _generate_openai_with_tools.
    """
    async def mock_create(**kwargs):
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools")

        # Check if this is the first turn (tools provided) or second turn (tool results provided)
        has_tool_response = any(m.get("role") == "tool" for m in messages if isinstance(m, dict))

        # Extract last user message
        user_msg = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                user_msg = m.get("content", "")

        user_lower = user_msg.lower()

        if tools and not has_tool_response:
            # First turn: model decides which tool to call based on user query
            if "recommend batch configuration" in user_lower:
                tc = MockToolCall(
                    "call_1",
                    "recommend_batch_configuration",
                    {"product_name": "Aspirin 500mg", "target_yield": 1000},
                )
                return MockCompletion(MockMessage(tool_calls=[tc]))

            elif "rm-88321" in user_lower or "material lot" in user_lower:
                tc = MockToolCall(
                    "call_2",
                    "get_material_lot_genealogy",
                    {"lot_id": "RM-88321"},
                )
                return MockCompletion(MockMessage(tool_calls=[tc]))

            elif "calculate risk score" in user_lower or "b-999" in user_lower:
                tc = MockToolCall(
                    "call_3",
                    "calculate_risk_score",
                    {"batch_id": "B-999"},
                )
                return MockCompletion(MockMessage(tool_calls=[tc]))

            elif "b-99999" in user_lower:
                tc = MockToolCall(
                    "call_4",
                    "get_batch_by_id",
                    {"batch_id": "B-99999"},
                )
                return MockCompletion(MockMessage(tool_calls=[tc]))

            elif "b-1021" in user_lower or "batch" in user_lower:
                tc = MockToolCall(
                    "call_5",
                    "get_batch_by_id",
                    {"batch_id": "B-1021"},
                )
                return MockCompletion(MockMessage(tool_calls=[tc]))

            # Default no-tool response
            return MockCompletion(MockMessage(content=f"Understood: {user_msg}"))

        # Second turn: synthesize tool output into final response text
        if "recommend batch configuration" in user_lower:
            ans = (
                f"{ADVISORY_STATUS_LABEL} NOTICE: This is an advisory-only recommendation.\n"
                "Recommended Configuration: Room Cleanroom-1, Yield 98.9%.\n"
                "Risk Indicator: Low.\n"
                "Required Human Approvals: QA and Production Supervisor sign-off."
            )
            return MockCompletion(MockMessage(content=ans))

        elif "rm-88321" in user_lower or "material lot" in user_lower:
            ans = (
                "[FACT] Material Lot RM-88321 Genealogy [Source: Material RM-88321]:\n"
                "Traceability Chain: material_lot -> batches -> finished_drugs\n"
                "[CONFIRMED IMPACT] Batch B-101 has confirmed deviation link.\n"
                "[POTENTIAL IMPACT] Batch B-102 consumed material on shared line."
            )
            return MockCompletion(MockMessage(content=ans))

        elif "b-99999" in user_lower:
            ans = "[FACT] Batch B-99999 is unavailable or not found in system records."
            return MockCompletion(MockMessage(content=ans))

        elif "calculate risk score" in user_lower or "b-999" in user_lower:
            ans = "[CALCULATION] Risk score calculated with config version v1.0. Overall Risk Score: 18.0."
            return MockCompletion(MockMessage(content=ans))

        elif "b-1021" in user_lower:
            # Check if permission was denied in tool messages
            tool_denied = any(
                "permission_denied" in str(m.get("content", ""))
                for m in messages
                if isinstance(m, dict) and m.get("role") == "tool"
            )
            if tool_denied:
                ans = "Permission denied: You do not have permission to view batch records under the Sales role."
            else:
                ans = "[FACT] Batch B-1021 details retrieved successfully. Status: IN_PROGRESS."
            return MockCompletion(MockMessage(content=ans))

        return MockCompletion(MockMessage(content="Response generated."))

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
    mock_client.close = AsyncMock()

    with patch("app.services.ai_service._build_openai_client", return_value=mock_client):
        yield mock_client


@pytest.fixture(autouse=True)
def cleanup_dependency_overrides():
    """Ensure FastAPI dependency overrides are cleared after each test to prevent test leakage."""
    from app.main import app
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def cleanup_user_context():
    """Ensure user identity and token context variables are cleared after each test."""
    yield
    from app.services.facility_api_service import set_current_token, set_current_user
    set_current_user(user_id="", username="", collection_id="")
    set_current_token("")


import copy
from uuid import uuid4


class AsyncFakeCursor:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = [copy.deepcopy(d) for d in docs]

    def sort(self, key_or_list: Any, direction: int = 1) -> AsyncFakeCursor:
        if isinstance(key_or_list, list) and len(key_or_list) > 0:
            sort_key, direction = key_or_list[0]
        else:
            sort_key = key_or_list
        reverse = (direction == -1)
        self._docs.sort(key=lambda d: str(d.get(sort_key, "")), reverse=reverse)
        return self

    def limit(self, n: int) -> AsyncFakeCursor:
        self._docs = self._docs[:n]
        return self

    async def to_list(self, length: int | None = None) -> list[dict[str, Any]]:
        if length is not None:
            return [copy.deepcopy(d) for d in self._docs[:length]]
        return [copy.deepcopy(d) for d in self._docs]


class AsyncFakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self._docs: list[dict[str, Any]] = []
        self._indexes: list[Any] = []

    def _matches(self, doc: dict[str, Any], query: dict[str, Any]) -> bool:
        for k, v in query.items():
            if k == "$or":
                if not any(self._matches(doc, sub) for sub in v):
                    return False
            elif k == "turns.0":
                if isinstance(v, dict) and "$exists" in v:
                    turns = doc.get("turns") or []
                    if (len(turns) > 0) != bool(v["$exists"]):
                        return False
            else:
                if doc.get(k) != v:
                    return False
        return True

    async def insert_one(self, doc: dict[str, Any]) -> Any:
        d = copy.deepcopy(doc)
        if "_id" not in d:
            d["_id"] = d.get("id") or str(uuid4())
        self._docs.append(d)
        res = MagicMock()
        res.inserted_id = d["_id"]
        return res

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        for d in self._docs:
            if self._matches(d, query):
                return copy.deepcopy(d)
        return None

    def find(self, query: dict[str, Any] | None = None) -> AsyncFakeCursor:
        q = query or {}
        matching = [d for d in self._docs if self._matches(d, q)]
        return AsyncFakeCursor(matching)

    async def update_one(self, query: dict[str, Any], update: dict[str, Any]) -> Any:
        for d in self._docs:
            if self._matches(d, query):
                for field, item in update.get("$push", {}).items():
                    d.setdefault(field, []).append(copy.deepcopy(item))
                for field, val in update.get("$set", {}).items():
                    d[field] = copy.deepcopy(val)
                res = MagicMock()
                res.modified_count = 1
                return res
        res = MagicMock()
        res.modified_count = 0
        return res

    async def delete_one(self, query: dict[str, Any]) -> Any:
        for i, d in enumerate(self._docs):
            if self._matches(d, query):
                self._docs.pop(i)
                res = MagicMock()
                res.deleted_count = 1
                return res
        res = MagicMock()
        res.deleted_count = 0
        return res

    async def create_index(self, keys: Any, **kwargs: Any) -> str:
        self._indexes.append((keys, kwargs))
        return "idx_" + str(len(self._indexes))


class AsyncFakeAdmin:
    async def command(self, cmd: str, **kwargs: Any) -> dict[str, Any]:
        if cmd == "ping":
            return {"ok": 1.0}
        return {"ok": 1.0}


class AsyncFakeMongoDatabase:
    def __init__(self) -> None:
        self._collections: dict[str, AsyncFakeCollection] = {}
        self.admin = AsyncFakeAdmin()

    def __getitem__(self, name: str) -> AsyncFakeCollection:
        if name not in self._collections:
            self._collections[name] = AsyncFakeCollection(name)
        return self._collections[name]

    def clear(self) -> None:
        self._collections.clear()


@pytest.fixture(autouse=True)
def setup_test_mongo_database():
    """Ensure tests run with an in-memory async Motor-compatible database."""
    from app.core.database import set_database
    fake_db = AsyncFakeMongoDatabase()
    set_database(fake_db)
    yield fake_db
    fake_db.clear()
    set_database(None)

