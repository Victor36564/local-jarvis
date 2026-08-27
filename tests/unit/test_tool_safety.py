from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from jarvis.config import ToolConfig
from jarvis.tools import _expand_home_paths, format_tool_result, web_search
from jarvis.tool_safety import CommandSafetyError, command_is_allowlisted, enforce_command_policy


def test_allowlist_match_prefix():
    cfg = ToolConfig(require_confirmation=False, allowlist=["echo", "dir"])
    assert command_is_allowlisted("echo hello", cfg)
    assert not command_is_allowlisted("del /s C:\\", cfg)


def test_enforce_blocks_non_allowlisted(caplog):
    cfg = ToolConfig(require_confirmation=False, allowlist=["echo"])
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
    cfg = ToolConfig(require_confirmation=False, allowlist=["nvidia-smi"])

    enforce_command_policy("nvidia-smi", cfg)


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
