from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


def _extract_json_object(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(0)


def parse_tool_call(text: str) -> ToolCall | None:
    candidate = _extract_json_object(text)
    if candidate is None:
        return None

    try:
        raw = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(raw, dict):
        return None

    try:
        return ToolCall.model_validate(raw)
    except ValidationError:
        return None
