from __future__ import annotations

from jarvis.config import load_config
from jarvis.model_loader import load_model_bundle
from jarvis.telemetry import snapshot_gpu_memory


def main() -> None:
    cfg = load_config()
    before = snapshot_gpu_memory()
    print(
        "before load: "
        f"allocated={before.vram_allocated_gb:.2f}GB "
        f"reserved={before.vram_reserved_gb:.2f}GB"
    )

    _ = load_model_bundle(cfg)

    after = snapshot_gpu_memory()
    print(
        "after load: "
        f"allocated={after.vram_allocated_gb:.2f}GB "
        f"reserved={after.vram_reserved_gb:.2f}GB"
    )

    if after.vram_reserved_gb > cfg.model.vram_budget_gb:
        raise SystemExit(
            "VRAM budget exceeded: "
            f"{after.vram_reserved_gb:.2f}GB > {cfg.model.vram_budget_gb:.2f}GB"
        )


if __name__ == "__main__":
    main()
