from __future__ import annotations


def main() -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"Torch import failed: {exc}")
        return

    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024**3)
        print(f"device index: {idx}")
        print(f"device name: {props.name}")
        print(f"compute capability: {props.major}.{props.minor}")
        print(f"total memory (GB): {total_gb:.2f}")


if __name__ == "__main__":
    main()
