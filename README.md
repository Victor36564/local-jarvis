# local-jarvis

Local multimodal Windows assistant.

## Current Status

Implemented baseline modules for:

- Config and runtime limits
- Quantized model loading (4-bit primary, 8-bit fallback)
- Wake-word listener skeleton (`hey_jarvis`)
- Command audio capture and normalization
- Primary monitor screen capture and downscaling
- Tool execution with allowlist + confirmation safety
- LangGraph inference/tool loop skeleton
- Strict tool-call parsing and per-turn tool-call budget guard
- Configurable transcription modes (`direct`, `stt`, `stt_fallback`, `manual`)
- CUDA and VRAM diagnostic scripts

This is an implementation starter. It is not production-ready yet.

## Prerequisites (Windows 10/11)

- Python 3.10+
- NVIDIA driver compatible with CUDA for RTX 4060
- A working microphone

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate
python -m pip install --upgrade pip
cd local-jarvis
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
```

If editable install with extras fails in your shell, use:

```powershell
pip install -r requirements.txt
pip install pytest pytest-timeout ruff
```

## Diagnostics

CUDA check:

```powershell
python scripts/check_cuda.py
```

VRAM profile check:

```powershell
python scripts/profile_vram.py
```

## Run

Default profile:

```powershell
jarvis
```

RTX 4060 tuned profile:

```powershell
jarvis --config configs/windows_rtx4060.yaml
```

On Windows, `jarvis` opens a separate status console while keeping the original
terminal attached to the application. Logging remains in the original terminal;
the popup contains only green status text for startup progress, `Listening...`,
wake-word detection, audio/transcription progress, tool activity, and the final
response. If Windows cannot create the popup, status text falls back to the
original terminal alongside the logging output.

## Safety Model

- `execute_terminal_command` is allowlist-only.
- Allowlisted commands execute automatically; commands outside the allowlist are blocked.
- Command output is truncated to avoid flooding context.
- Tool calls are capped per turn via runtime config.

## Transcription Modes

Set `audio.transcription_mode` in config:

- `direct`: pass audio context to model and use a generic transcript prompt.
- `stt`: require local STT and use the transcript.
- `stt_fallback`: try local STT first, then prompt for manual transcript.
- `manual`: always ask for typed transcript.

## Tests

```powershell
pytest -q
```