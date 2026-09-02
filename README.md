# local-jarvis

Local multimodal Windows OS assistant. This was created using full agentic coding (GitHub Copilot) starting with an inital PRD document (plan.md). Refer to plan.md for the full outline and implementation of this project.

## Demo

[![Watch the demo here](https://img.youtube.com/vi/yHPWhWqzMJA/maxresdefault.jpg)](https://youtu.be/yHPWhWqzMJA)


## Features

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

## Prerequisites (Windows 10/11)

- Python 3.10+
- NVIDIA driver compatible with CUDA
- A working microphone

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
cd local-jarvis
pip install -e .
python -c "import openwakeword; openwakeword.utils.download_models()"
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
