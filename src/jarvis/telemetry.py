from __future__ import annotations

# Lightweight runtime metrics used during model loading and diagnostics.
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RuntimeTelemetry:
    vram_allocated_gb: float = 0.0
    vram_reserved_gb: float = 0.0


def snapshot_gpu_memory() -> RuntimeTelemetry:
    # Telemetry must never prevent the assistant from starting.
    try:
        import torch

        if not torch.cuda.is_available():
            return RuntimeTelemetry()
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return RuntimeTelemetry(vram_allocated_gb=allocated, vram_reserved_gb=reserved)
    except Exception as exc:  # pragma: no cover
        logger.debug("GPU memory snapshot failed: %s", exc)
        return RuntimeTelemetry()
