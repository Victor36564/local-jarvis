from __future__ import annotations

import os
import subprocess
import webbrowser
from pathlib import Path
from typing import Any

from jarvis.config import ToolConfig
from jarvis.tool_safety import enforce_command_policy


def execute_terminal_command(command: str, cfg: ToolConfig) -> dict[str, Any]:
    enforce_command_policy(command, cfg)

    proc = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        shell=True,
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

    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    return {"opened": False, "query": query, "results": results}
