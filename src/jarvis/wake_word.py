from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from jarvis.config import AudioConfig

logger = logging.getLogger(__name__)


class WakeWordListener:
    def __init__(self, cfg: AudioConfig, model_name: str = "hey_jarvis") -> None:
        self.cfg = cfg
        self.model_name = model_name
        self._last_trigger = 0.0
        self._model = None

    def initialize(self) -> None:
        from openwakeword.model import Model

        # 1. Initialize openwakeword clean without passing bad kwargs
        self._model = Model(
            wakeword_models=[self.model_name],
            inference_framework="onnx"
        )

        # 2. Force both the preprocessor and model sessions to release your full 4060 GPU
        cpu_provider = ["CPUExecutionProvider"]
        
        # Shift the embedding engine session to CPU
        if hasattr(self._model, 'preprocessor') and hasattr(self._model.preprocessor, 'melspec_model'):
            self._model.preprocessor.melspec_model.set_providers(cpu_provider)
            
        # Shift the actual hotword prediction classification layers to CPU
        for name in self._model.models:
            self._model.models[name].set_providers(cpu_provider)

    def listen(
        self, on_wake: Callable[[], None], on_status: Callable[[str], None] | None = None
    ) -> None:
        import sounddevice as sd

        if self._model is None:
            self.initialize()

        while True:
            wake_event = threading.Event()
            stats = {"callbacks": 0, "rms": 0.0, "peak": 0.0, "score": 0.0}

            def callback(
                indata,
                frames,
                time_info,
                status,
                *,
                wake_event=wake_event,
                stats=stats,
            ):  # pragma: no cover
                del frames, time_info
                if status:
                    logger.warning("Audio stream status: %s", status)

                samples = np.squeeze(indata).astype(np.float32)
                stats["callbacks"] += 1
                stats["rms"] = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
                stats["peak"] = float(np.max(np.abs(samples))) if samples.size else 0.0

                # openWakeWord expects signed 16-bit PCM, as used by the mic test.
                audio_int16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
                pred = self._model.predict(audio_int16)

                score = float(next(iter(pred.values()))) if pred else 0.0
                stats["score"] = score
                now = time.monotonic()

                if score >= self.cfg.wake_threshold and now - self._last_trigger > self.cfg.debounce_seconds:
                    self._last_trigger = now
                    logger.info("Wake word detected with score %.3f", score)
                    wake_event.set()

            with sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                blocksize=self.cfg.wake_chunk_size,
                dtype="float32",
                device=self.cfg.device_index,
                callback=callback,
            ):
                logger.info("Wake listener active. Say Jarvis.")
                last_health_log = time.monotonic()
                while not wake_event.wait(0.1):
                    now = time.monotonic()
                    if now - last_health_log >= 1.0:
                        logger.info(
                            "Audio input active: callbacks=%d rms=%.4f peak=%.4f wake_score=%.3f",
                            stats["callbacks"],
                            stats["rms"],
                            stats["peak"],
                            stats["score"],
                        )
                        last_health_log = now

            # Close the wake stream before opening the command recording stream.
            logger.info("Wake listener stopped; processing detected wake word.")
            on_wake()
            self._model.reset()
            logger.info("Command processing complete; resuming wake listener.")
            if on_status:
                on_status("Listening...")
