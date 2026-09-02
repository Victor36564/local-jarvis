from __future__ import annotations

# Allowlist policy for commands exposed through the terminal tool.
import logging

from jarvis.config import ToolConfig

logger = logging.getLogger(__name__)


class CommandSafetyError(ValueError):
    """Raised when a command is blocked by safety policy."""


def command_is_allowlisted(command: str, cfg: ToolConfig) -> bool:
    normalized = command.strip().lower()
    return any(normalized.startswith(pattern.lower()) for pattern in cfg.allowlist)


def enforce_command_policy(command: str, cfg: ToolConfig) -> None:
    # Reject before execution whenever the command lacks a safe prefix.
    if not command_is_allowlisted(command, cfg):
        logger.info("Command not executed: not in allowlist: %s", command)
        raise CommandSafetyError(
            "Command blocked by allowlist policy."
            " Add a safe prefix to config.tools.allowlist if needed."
        )
