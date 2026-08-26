from __future__ import annotations

import logging
from dataclasses import dataclass
import torch

from jarvis.config import JarvisConfig
from jarvis.telemetry import snapshot_gpu_memory

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    model: object
    processor: object


def _bnb_config(mode: str):
    from transformers import BitsAndBytesConfig

    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Passed as a torch object instead of a string
        bnb_4bit_use_double_quant=True,       # Nested quantization saves an extra ~0.4 bits/weight
    )


def load_model_bundle(config: JarvisConfig) -> ModelBundle:
    from transformers import AutoModelForCausalLM, AutoProcessor

    quantization = config.model.quantization
    try_modes = [quantization, "8bit"] if quantization != "8bit" else ["8bit"]
    last_err: Exception | None = None

    for mode in try_modes:
        try:
            logger.info("Loading model %s with %s", config.model.model_id, mode)
            
            # Explicitly force everything onto your RTX 4060 (cuda:0)
            # This completely bypasses the broken auto-offload logic
            device_map = {"": "cuda:0"}

            model = AutoModelForCausalLM.from_pretrained(
                config.model.model_id,
                device_map=device_map,
                quantization_config=_bnb_config(mode),
                torch_dtype=torch.float16,   # Ensures full-precision weights don't spill to RAM/CPU
                low_cpu_mem_usage=True,      # Prevents memory allocation spikes during initiation
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(config.model.model_id, trust_remote_code=True)
            mem = snapshot_gpu_memory()
            logger.info(
                "GPU memory after model load: allocated=%.2fGB reserved=%.2fGB",
                mem.vram_allocated_gb,
                mem.vram_reserved_gb,
            )
            return ModelBundle(model=model, processor=processor)
        except Exception as exc:
            last_err = exc
            logger.warning("Model load failed for mode %s: %s", mode, exc)

    raise RuntimeError("Failed to load model in 4bit/8bit modes") from last_err
