from __future__ import annotations

# Local speech-to-text adapter with CPU/GPU selection.
import logging

import numpy as np

logger = logging.getLogger(__name__)


class STTUnavailableError(RuntimeError):
    """Raised when local STT cannot be initialized or run."""


def transcribe_audio(audio: np.ndarray, sample_rate: int, model_id: str) -> str:
    if audio.size == 0:
        return ""

    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover
        raise STTUnavailableError("transformers ASR pipeline is unavailable") from exc

    device = "cpu"
    try:
        # Prefer CUDA when available, but keep transcription usable on CPU.
        import torch

        if torch.cuda.is_available():
            device = "cuda:0"
    except Exception:
        device = "cpu"

    try:
        asr = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            device=device,
        )
        result = asr({"raw": audio, "sampling_rate": sample_rate})
    except Exception as exc:
        raise STTUnavailableError(f"STT inference failed: {exc}") from exc

    text = result.get("text", "") if isinstance(result, dict) else str(result)
    transcript = text.strip()
    logger.info("STT model %s produced transcript: %s", model_id, transcript)
    logger.info("Generated transcript with %d chars", len(transcript))
    return transcript
