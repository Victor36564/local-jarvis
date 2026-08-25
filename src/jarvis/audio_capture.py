from __future__ import annotations

import logging

import numpy as np

from jarvis.config import AudioConfig

logger = logging.getLogger(__name__)


def list_input_devices() -> list[dict]:
    import sounddevice as sd

    devices = sd.query_devices()
    return [d for d in devices if d.get("max_input_channels", 0) > 0]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


def record_command_audio(cfg: AudioConfig) -> np.ndarray:
    import sounddevice as sd

    frames = int(cfg.command_seconds * cfg.sample_rate)
    logger.info("Recording audio for up to %s seconds", cfg.command_seconds)
    audio = sd.rec(
        frames,
        samplerate=cfg.sample_rate,
        channels=cfg.channels,
        dtype="float32",
        device=cfg.device_index,
    )
    sd.wait()
    return normalize_audio(audio)
