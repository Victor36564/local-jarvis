from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch
import subprocess

import pytest

from jarvis.config import ToolConfig
from jarvis.tools import _command_invocation, _expand_home_paths, format_tool_result, web_search
from jarvis.tool_safety import CommandSafetyError, command_is_allowlisted, enforce_command_policy


def test_allowlist_match_prefix():
    cfg = ToolConfig(allowlist=["echo", "dir"])
    assert command_is_allowlisted("echo hello", cfg)
    assert not command_is_allowlisted("del /s C:\\", cfg)


def test_enforce_blocks_non_allowlisted(caplog):
    cfg = ToolConfig(allowlist=["echo"])
    caplog.set_level("INFO")
    with pytest.raises(CommandSafetyError):
        enforce_command_policy("dir", cfg)
    assert "Command not executed: not in allowlist: dir" in caplog.text


def test_expand_home_paths_for_generated_windows_command():
    expanded = _expand_home_paths('echo hello > ~/Desktop/test.txt')

    assert "~/Desktop" not in expanded
    assert expanded.startswith(f"echo hello > {Path.home()}\\Desktop")
    assert expanded.endswith("Desktop/test.txt")


def test_nvidia_smi_is_allowlisted():
    cfg = ToolConfig(allowlist=["nvidia-smi"])

    enforce_command_policy("nvidia-smi", cfg)


def test_default_policy_does_not_prompt_for_allowlisted_command(monkeypatch):
    cfg = ToolConfig(allowlist=["echo"])
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("unexpected confirmation prompt"))

    enforce_command_policy("echo hello", cfg)


@pytest.mark.parametrize(
    "command",
    [
        "tasklist /fo table",
        "Get-Counter '\\Processor(_Total)\\% Processor Time'",
        "start spotify",
        "start ms-settings:display",
        'start "" notepad.exe',
        "explorer C:\\Users",
        "winget search vscode",
    ],
)
def test_common_safe_windows_commands_are_allowlisted(command):
    cfg = ToolConfig()

    enforce_command_policy(command, cfg)


def test_powershell_cmdlet_uses_powershell(monkeypatch):
    monkeypatch.setattr("jarvis.tools.os.name", "nt")

    invocation = _command_invocation("Get-Counter -Counter '_TotalProcessorTime'")

    assert invocation == (
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-Counter -Counter '_TotalProcessorTime'",
        ],
        False,
    )


def test_execute_terminal_command_runs_powershell_cmdlet(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, "cpu output", "")

    monkeypatch.setattr("jarvis.tools.os.name", "nt")
    monkeypatch.setattr("jarvis.tools.subprocess.run", fake_run)
    cfg = ToolConfig(allowlist=["Get-Counter"])

    from jarvis.tools import execute_terminal_command

    result = execute_terminal_command("Get-Counter -Counter '_TotalProcessorTime'", cfg)

    assert result["stdout"] == "cpu output"
    assert calls[0][0][0][0] == "powershell.exe"
    assert calls[0][1]["shell"] is False


def test_web_search_reports_empty_results():
    class EmptyDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def text(self, _query, max_results):
            assert max_results == 5
            return []

    fake_ddgs = ModuleType("ddgs")
    fake_ddgs.DDGS = EmptyDDGS
    with patch.dict(sys.modules, {"ddgs": fake_ddgs}):
        result = web_search("query with no matches")

    assert result["results"] == []
    assert result["message"] == "No search results found"


def test_format_gpu_terminal_result():
    result = {
        "return_code": 0,
        "stdout": "|   0  NVIDIA GeForce RTX 4060 Ti   WDDM  |\n|    7859MiB /   8188MiB |      3%      Default |",
        "stderr": "",
    }

    summary = format_tool_result("execute_terminal_command", result)

    assert "NVIDIA GeForce RTX 4060 Ti" in summary
    assert "329 MiB free" in summary
