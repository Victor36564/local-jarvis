from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    wake_chunk_size: int = 1280
    command_seconds: int = 20
    silence_seconds: float = 2.0
    silence_threshold: float = 0.01
    wake_threshold: float = 0.5
    debounce_seconds: float = 1.0
    device_index: int | None = None
    transcription_mode: str = "stt_fallback"
    stt_model_id: str = "openai/whisper-small.en"


class VisionConfig(BaseModel):
    max_width: int = 1280
    max_height: int = 720
    primary_monitor_only: bool = True


class ModelConfig(BaseModel):
    model_id: str = "google/gemma-4-E4B-it"
    device: str = "cuda:0"
    quantization: str = "4bit"
    max_new_tokens: int = 256
    max_context_tokens: int = 4096
    vram_budget_gb: float = 7.5


class ToolConfig(BaseModel):
    command_timeout_seconds: int = 25
    max_output_chars: int = 12000
    require_confirmation: bool = True
    allowlist: list[str] = Field(
        default_factory=lambda: [
            "dir",
            "echo",
            "type",
            "Get-ChildItem",
            "Get-Location",
            "git status",
            "git log",
            "python --version",
        ]
    )


class RuntimeConfig(BaseModel):
    listen_on_startup: bool = True
    recursion_limit: int = 4
    max_tool_calls_per_turn: int = 3
    log_level: str = "INFO"


class JarvisConfig(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> JarvisConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return JarvisConfig()

    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    return JarvisConfig.model_validate(data)
