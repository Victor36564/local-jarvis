from __future__ import annotations

import json
import logging
from typing import Any

from jarvis.prompting import build_messages
from jarvis.tool_call_parser import parse_tool_call

logger = logging.getLogger(__name__)


TOOL_CALL_SCHEMA_HELP = (
    "If you need a tool, return strict JSON with keys: "
    "tool_name (string), arguments (object). "
    "Otherwise return plain assistant text."
)


def infer_once(
    model: Any,
    processor: Any,
    transcript: str,
    screenshot: Any,
    audio: Any,
    max_new_tokens: int,
    tool_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages = build_messages(transcript, tool_result=tool_result)
    messages[0]["content"] = f"{messages[0]['content']} {TOOL_CALL_SCHEMA_HELP}"

    inputs = processor(
        text=[m["content"] for m in messages],
        images=[screenshot] if screenshot is not None else None,
        audio=[audio] if audio is not None else None,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    parsed = parse_tool_call(text)
    if parsed is not None:
        return {"type": "tool_call", "payload": parsed.model_dump()}

    try:
        # If plain JSON exists but is not a valid tool call, preserve it as final text.
        json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Model output is not JSON; returning text response")

    return {"type": "final", "payload": text}
