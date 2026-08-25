from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from jarvis.agent import infer_once
from jarvis.config import JarvisConfig
from jarvis.state import JarvisState
from jarvis.tools import create_note, execute_terminal_command, read_file_content, web_search


TOOL_REGISTRY = {
    "execute_terminal_command": execute_terminal_command,
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
            return {"tool_result": {"error": f"Unknown tool: {name}"}, "pending_tool_call": None}

        try:
            if name == "execute_terminal_command":
                result = tool(arguments.get("command", ""), cfg.tools)
            elif name == "web_search":
                result = tool(arguments.get("query", ""), arguments.get("open_in_browser", False))
            elif name == "create_note":
                result = tool(arguments.get("content", ""), arguments.get("title"))
            else:
                result = tool(arguments.get("file_path", ""))
        except Exception as exc:  # pragma: no cover
            result = {"error": str(exc), "tool": name}

        return {"tool_result": result, "pending_tool_call": None}

    def should_run_tool(state: JarvisState) -> str:
        return "tool" if state.get("pending_tool_call") else "end"

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
    graph.add_edge("tool", "infer")

    return graph.compile()
