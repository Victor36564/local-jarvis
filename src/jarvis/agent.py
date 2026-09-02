from __future__ import annotations

# Model inference and tool-call interpretation live here.
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
    # Build one multimodal prompt and keep absent modalities out of the request.
    messages = build_messages(transcript, tool_result=tool_result)
    messages[0]["content"] = f"{messages[0]['content']} {TOOL_CALL_SCHEMA_HELP}"

    user_message = next(message for message in messages if message["role"] == "user")

    # Inject modality tokens into the user prompt. A tool-result message may be last.
    if screenshot is not None:
        user_message["content"] = f"<|image|>\n{user_message['content']}"
        
    if audio is not None:
        user_message["content"] = f"<|audio|>\n{user_message['content']}"

    # Flatten messages into a single prompt string using the processor's built-in template
    formatted_text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Only include image/audio keys if they are actually present, avoiding mismatched lists
    kwargs = {"text": formatted_text, "return_tensors": "pt", "padding": True}
    if screenshot is not None:
        kwargs["images"] = [screenshot]
    if audio is not None:
        kwargs["audio"] = [audio]

    inputs = processor(**kwargs)

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