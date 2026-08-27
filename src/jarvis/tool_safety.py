from __future__ import annotations

import logging

from jarvis.config import ToolConfig

logger = logging.getLogger(__name__)


class CommandSafetyError(ValueError):
    """Raised when a command is blocked by safety policy."""


def command_is_allowlisted(command: str, cfg: ToolConfig) -> bool:
    normalized = command.strip().lower()
    return any(normalized.startswith(pattern.lower()) for pattern in cfg.allowlist)


def require_user_confirmation(command: str, cfg: ToolConfig) -> bool:
    if not cfg.require_confirmation:
        return True
    response = input(f"Approve execution of '{command}'? [Y/N]: ").strip().lower()
    approved = response in {"y", "yes"}
    if not approved:
        logger.info("Command not executed: user denied confirmation: %s", command)
    return approved


def enforce_command_policy(command: str, cfg: ToolConfig) -> None:
    if not command_is_allowlisted(command, cfg):
        logger.info("Command not executed: not in allowlist: %s", command)
        raise CommandSafetyError(
            "Command blocked by allowlist policy."
            " Add a safe prefix to config.tools.allowlist if needed."
        )
    if not require_user_confirmation(command, cfg):
        raise CommandSafetyError("Command execution denied by user.")
