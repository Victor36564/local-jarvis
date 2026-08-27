from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from jarvis.agent import infer_once
from jarvis.config import JarvisConfig
from jarvis.state import JarvisState
from jarvis.tools import (
    create_note,
    execute_terminal_command,
    format_tool_result,
    read_file_content,
    web_search,
)

logger = logging.getLogger(__name__)


TOOL_REGISTRY = {
    "execute_terminal_command": execute_terminal_command,
    "terminal": execute_terminal_command,
    "read_file_content": read_file_content,
    "create_note": create_note,
    "web_search": web_search,
}


def build_graph(model: Any, processor: Any, cfg: JarvisConfig):
    graph = StateGraph(JarvisState)

    def infer_node(state: JarvisState) -> JarvisState:
        tool_calls_count = state.get("tool_calls_count", 0)
        if tool_calls_count > cfg.runtime.max_tool_calls_per_turn:
            return {
                "final_response": "Tool call limit reached for this turn. Please refine the request.",
                "pending_tool_call": None,
            }

        result = infer_once(
            model=model,
            processor=processor,
            transcript=state.get("transcript", ""),
            screenshot=state.get("screenshot"),
            audio=state.get("audio"),
            max_new_tokens=cfg.model.max_new_tokens,
            tool_result=state.get("tool_result"),
        )
        if result["type"] == "tool_call":
            logger.info("Tool call requested: %s", result["payload"])
            if state.get("tool_result") is not None:
                logger.info("Follow-up tool call not executed; returning previous result")
                return {
                    "final_response": format_tool_result(
                        result["payload"]["tool_name"], state["tool_result"]
                    ),
                    "pending_tool_call": None,
                }
            return {
                "pending_tool_call": result["payload"],
                "tool_calls_count": tool_calls_count + 1,
            }
        return {"final_response": result["payload"], "pending_tool_call": None}

    def tool_node(state: JarvisState) -> JarvisState:
        call = state.get("pending_tool_call") or {}
        name = call.get("tool_name")
        arguments = call.get("arguments", {})

        tool = TOOL_REGISTRY.get(name)
        if tool is None:
            logger.info("Tool not executed: unknown tool '%s'", name)
            return {
                "tool_result": {"error": f"Unknown tool: {name}"},
                "pending_tool_call": None,
                "last_tool_call": call,
            }

        try:
            if name == "execute_terminal_command":
                result = tool(arguments.get("command", ""), cfg.tools)
            elif name == "terminal":
                result = tool(arguments.get("command", ""), cfg.tools)
            elif name == "web_search":
                result = tool(arguments.get("query", ""), arguments.get("open_in_browser", False))
            elif name == "create_note":
                result = tool(arguments.get("content", ""), arguments.get("title"))
            else:
                result = tool(arguments.get("file_path", ""))
        except Exception as exc:  # pragma: no cover
            logger.warning("Tool '%s' failed: %s", name, exc)
            result = {"error": str(exc), "tool": name}

        logger.info("Tool '%s' completed with result: %s", name, result)

        response = {"tool_result": result, "pending_tool_call": None, "last_tool_call": call}
        if "error" in result:
            response["final_response"] = format_tool_result(name or "tool", result)
        return response

    def should_run_tool(state: JarvisState) -> str:
        return "tool" if state.get("pending_tool_call") else "end"

    def should_continue_after_tool(state: JarvisState) -> str:
        return "end" if state.get("final_response") else "infer"

    graph.add_node("infer", infer_node)
    graph.add_node("tool", tool_node)
    graph.set_entry_point("infer")

    graph.add_conditional_edges(
        "infer",
        should_run_tool,
        {
            "tool": "tool",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "tool",
        should_continue_after_tool,
        {
            "infer": "infer",
            "end": END,
        },
    )

    return graph.compile()
