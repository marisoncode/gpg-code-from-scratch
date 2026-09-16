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
