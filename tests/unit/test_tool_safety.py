import pytest

from jarvis.config import ToolConfig
from jarvis.tool_safety import CommandSafetyError, command_is_allowlisted, enforce_command_policy


def test_allowlist_match_prefix():
    cfg = ToolConfig(require_confirmation=False, allowlist=["echo", "dir"])
    assert command_is_allowlisted("echo hello", cfg)
    assert not command_is_allowlisted("del /s C:\\", cfg)


def test_enforce_blocks_non_allowlisted():
    cfg = ToolConfig(require_confirmation=False, allowlist=["echo"])
    with pytest.raises(CommandSafetyError):
        enforce_command_policy("dir", cfg)
