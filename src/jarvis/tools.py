from __future__ import annotations

import os
import re
import subprocess
import webbrowser
from pathlib import Path
from typing import Any

from jarvis.config import ToolConfig
from jarvis.tool_safety import enforce_command_policy


def execute_terminal_command(command: str, cfg: ToolConfig) -> dict[str, Any]:
    command = _expand_home_paths(command)
    enforce_command_policy(command, cfg)

    invocation, use_shell = _command_invocation(command)
    proc = subprocess.run(  # noqa: S603
        invocation,
        capture_output=True,
        shell=use_shell,
        text=True,
        timeout=cfg.command_timeout_seconds,
    )
    stdout = proc.stdout[: cfg.max_output_chars]
    stderr = proc.stderr[: cfg.max_output_chars]
    return {
        "return_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _command_invocation(command: str) -> tuple[list[str] | str, bool]:
    """Use PowerShell for cmdlets; retain CMD semantics for ordinary commands."""
    if os.name == "nt" and re.match(r"^\s*(?:powershell(?:\.exe)?\b|get-|set-|select-|measure-|format-|convertto-)", command, re.IGNORECASE):
        return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], False
    return command, True


def _expand_home_paths(command: str) -> str:
    """Expand the Unix-style home shorthand commonly generated for Windows."""
    return re.sub(r"(?<![\w/~])~/", lambda _: f"{Path.home()}\\", command)


def read_file_content(file_path: str) -> dict[str, Any]:
    path = Path(os.path.expanduser(file_path)).resolve()
    text = path.read_text(encoding="utf-8", errors="replace")
    return {"file_path": str(path), "content": text}


def create_note(content: str, title: str | None = None) -> dict[str, Any]:
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    safe_title = (title or "jarvis_note").strip().replace(" ", "_")
    file_path = desktop / f"{safe_title}.txt"
    file_path.write_text(content, encoding="utf-8")
    return {"saved_to": str(file_path)}


def web_search(query: str, open_in_browser: bool = False) -> dict[str, Any]:
    if open_in_browser:
        webbrowser.open(f"https://duckduckgo.com/?q={query}")
        return {"opened": True, "query": query}

    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    response = {"opened": False, "query": query, "results": results}
    if not results:
        response["message"] = "No search results found"
    return response


def format_tool_result(tool_name: str, result: dict[str, Any]) -> str:
    """Turn common tool responses into concise text suitable for the user."""
    if tool_name in {"execute_terminal_command", "terminal"}:
        return _format_terminal_result(result)
    if tool_name == "web_search":
        return _format_search_result(result)
    if "error" in result:
        return f"The {tool_name} tool failed: {result['error']}"
    return "; ".join(f"{key}: {value}" for key, value in result.items())


def _format_terminal_result(result: dict[str, Any]) -> str:
    if result.get("error"):
        return f"The command failed: {result['error']}"

    if result.get("return_code") != 0:
        error = result.get("stderr") or result.get("stdout") or "unknown error"
        return f"The command failed (exit code {result.get('return_code')}): {error.strip()}"

    output = result.get("stdout", "").strip()
    gpu_match = re.search(
        r"\|\s*\d+\s+([^|]+?)\s+WDDM.*?\|\s*([\d,]+)MiB\s*/\s*([\d,]+)MiB\s*\|\s*([\d]+)%",
        output,
        flags=re.DOTALL,
    )
    if gpu_match:
        name, used, total, utilization = gpu_match.groups()
        free = int(total.replace(",", "")) - int(used.replace(",", ""))
        return (
            f"GPU: {name.strip()}. VRAM: {free} MiB free of {total} MiB total "
            f"({used} MiB used). GPU utilization: {utilization}%."
        )
    return output or "The command completed successfully but returned no output."


def _format_search_result(result: dict[str, Any]) -> str:
    results = result.get("results", [])
    if not results:
        return result.get("message", "No search results found")

    lines = [f"Search results for {result.get('query', 'your query')}:"]
    for item in results[:5]:
        title = item.get("title", "Untitled result")
        body = item.get("body", "").strip()
        href = item.get("href", "")
        lines.append(f"- {title}: {body} ({href})")
    return "\n".join(lines)
