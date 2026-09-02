from __future__ import annotations

import logging
import threading

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

    chunks: list[np.ndarray] = []
    recording_done = threading.Event()
    silence_frames = 0
    speech_detected = False
    silence_limit = int(cfg.silence_seconds * cfg.sample_rate)

    logger.info(
        "Listening until %.1f seconds of silence (maximum %s seconds)",
        cfg.silence_seconds,
        cfg.command_seconds,
    )
    def callback(indata, frames, _time_info, status):  # pragma: no cover
        nonlocal silence_frames, speech_detected
        if status:
            logger.warning("Audio stream status: %s", status)

        chunk = np.asarray(indata, dtype=np.float32).copy()
        chunks.append(chunk)
        level = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0

        if level > cfg.silence_threshold:
            speech_detected = True
            silence_frames = 0
        elif speech_detected:
            silence_frames += frames
            if silence_frames >= silence_limit:
                recording_done.set()

    with sd.InputStream(
        samplerate=cfg.sample_rate,
        channels=cfg.channels,
        dtype="float32",
        device=cfg.device_index,
        callback=callback,
    ):
        recording_done.wait(timeout=cfg.command_seconds)

    if not chunks:
        return np.empty(0, dtype=np.float32)
    return normalize_audio(np.concatenate(chunks, axis=0))
