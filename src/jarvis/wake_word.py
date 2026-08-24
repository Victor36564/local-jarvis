from __future__ import annotations

import logging
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

        self._model = Model(wakeword_models=[self.model_name])

    def listen(self, on_wake: Callable[[], None]) -> None:
        import sounddevice as sd

        if self._model is None:
            self.initialize()

        def callback(indata, frames, time_info, status):  # pragma: no cover
            del frames, time_info
            if status:
                logger.warning("Audio stream status: %s", status)

            samples = np.squeeze(indata).astype(np.float32)
            pred = self._model.predict(samples)
            score = float(pred.get(self.model_name, 0.0))
            now = time.monotonic()

            if score >= self.cfg.wake_threshold and now - self._last_trigger > self.cfg.debounce_seconds:
                self._last_trigger = now
                logger.info("Wake word detected with score %.3f", score)
                on_wake()

        with sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            blocksize=self.cfg.wake_chunk_size,
            dtype="float32",
            device=self.cfg.device_index,
            callback=callback,
        ):
            logger.info("Wake listener active. Say Jarvis.")
            while True:
                time.sleep(0.1)
