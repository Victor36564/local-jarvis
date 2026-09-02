from __future__ import annotations

# Capture and downscale the configured monitor for multimodal inference.
from PIL import Image

from jarvis.config import VisionConfig


def capture_screen(cfg: VisionConfig) -> Image.Image:
    import mss

    with mss.mss() as sct:
        # mss index 1 is the primary monitor; index 0 represents all monitors.
        monitor = sct.monitors[1] if cfg.primary_monitor_only else sct.monitors[0]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)

    img.thumbnail((cfg.max_width, cfg.max_height), Image.Resampling.LANCZOS)
    return img
