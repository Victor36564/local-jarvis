from __future__ import annotations

from jarvis.config import ToolConfig


class CommandSafetyError(ValueError):
    """Raised when a command is blocked by safety policy."""


def command_is_allowlisted(command: str, cfg: ToolConfig) -> bool:
    normalized = command.strip().lower()
    return any(normalized.startswith(pattern.lower()) for pattern in cfg.allowlist)


def require_user_confirmation(command: str, cfg: ToolConfig) -> bool:
    if not cfg.require_confirmation:
        return True
    response = input(f"Approve execution of '{command}'? [Y/N]: ").strip().lower()
    return response in {"y", "yes"}


def enforce_command_policy(command: str, cfg: ToolConfig) -> None:
    if not command_is_allowlisted(command, cfg):
        raise CommandSafetyError(
            "Command blocked by allowlist policy."
            " Add a safe prefix to config.tools.allowlist if needed."
        )
    if not require_user_confirmation(command, cfg):
        raise CommandSafetyError("Command execution denied by user.")
