from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = (
    "You are Jarvis, a local Windows assistant. "
    "Be concise and safe. Use tools only when needed. "
    "For terminal commands, propose minimal and reversible actions."
)


def build_messages(transcript: str, tool_result: dict[str, Any] | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    if tool_result is not None:
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(tool_result, ensure_ascii=True),
            }
        )
    return messages
