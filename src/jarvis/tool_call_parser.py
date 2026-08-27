from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError
import yaml


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def parse_tool_call(text: str) -> ToolCall | None:
    raw = next(
        (
            candidate
            for candidate in reversed(_extract_json_objects(text))
            if "tool_name" in candidate and "arguments" in candidate
        ),
        None,
    )

    # Some local models emit the requested schema as YAML despite the prompt.
    if raw is None:
        yaml_match = re.search(r"(?ms)^\s*tool_name\s*:.*$", text)
        if yaml_match:
            try:
                raw = yaml.safe_load(yaml_match.group(0))
            except yaml.YAMLError:
                return None

    if not isinstance(raw, dict):
        return None

    try:
        return ToolCall.model_validate(raw)
    except ValidationError:
        return None
