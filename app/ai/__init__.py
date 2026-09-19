"""AI tool and schema layer for CPG AI Assistant."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "PREDEFINED_FUNCTIONS",
    "PREDEFINED_TOOL_SCHEMAS",
    "execute_predefined_tool",
    "AICapability",
    "CAPABILITY_REGISTRY",
    "get_capability",
    "is_capability_registered",
    "list_capabilities",
    "sanitize_for_llm",
    "tools",
]


def __getattr__(name: str) -> Any:
    if name in {"PREDEFINED_FUNCTIONS", "execute_predefined_tool"}:
        mod = importlib.import_module("app.ai.executor")
        return getattr(mod, name)
    if name == "PREDEFINED_TOOL_SCHEMAS":
        mod = importlib.import_module("app.ai.schemas")
        return getattr(mod, name)
    if name in {"AICapability", "CAPABILITY_REGISTRY", "get_capability", "is_capability_registered", "list_capabilities"}:
        mod = importlib.import_module("app.ai.registry")
        return getattr(mod, name)
    if name == "sanitize_for_llm":
        mod = importlib.import_module("app.ai.sanitizer")
        return getattr(mod, name)
    if name == "tools":
        return importlib.import_module("app.ai.tools")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
