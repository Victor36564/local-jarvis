from __future__ import annotations

# Typed state shared by the inference and tool graph nodes.
from typing import Any, TypedDict


class ToolCall(TypedDict):
    name: str
    arguments: dict[str, Any]


class JarvisState(TypedDict, total=False):
    transcript: str
    audio: Any
    screenshot: Any
    messages: list[dict[str, Any]]
    pending_tool_call: ToolCall | None
    last_tool_call: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    tool_calls_count: int
    final_response: str
