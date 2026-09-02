from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
from argparse import ArgumentParser

from jarvis.agent_graph import build_graph
from jarvis.audio_capture import record_command_audio
from jarvis.config import load_config
from jarvis.logging_setup import configure_logging
from jarvis.model_loader import load_model_bundle
from jarvis.screen_capture import capture_screen
from jarvis.stt import STTUnavailableError, transcribe_audio
from jarvis.terminal_ui import SocketTerminalUI, TerminalUI
from jarvis.wake_word import WakeWordListener

logger = logging.getLogger(__name__)


def _resolve_transcript(audio, cfg, terminal: TerminalUI | None = None) -> str:
    mode = cfg.audio.transcription_mode
    if mode == "direct":
        return "Please infer intent from provided audio and screenshot context."

    if mode == "stt":
        if terminal:
            terminal.status("Transcribing audio...")
        return transcribe_audio(audio, cfg.audio.sample_rate, cfg.audio.stt_model_id)

    if mode == "manual":
        return input("Command transcript: ").strip()

    # Default: STT fallback with manual backup for reliability.
    try:
        if terminal:
            terminal.status("Transcribing audio...")
        transcript = transcribe_audio(audio, cfg.audio.sample_rate, cfg.audio.stt_model_id)
        if transcript:
            return transcript
    except STTUnavailableError as exc:
        logger.warning("STT unavailable, falling back to manual transcript: %s", exc)

    return input("Command transcript (STT fallback): ").strip()


def _run_once(graph, cfg, terminal: TerminalUI | None = None):
    screenshot = capture_screen(cfg.vision)
    audio = record_command_audio(cfg.audio)
    transcript = _resolve_transcript(audio, cfg, terminal)
    if terminal:
        terminal.status(f'User Request: "{transcript}"')
    state = {
        "transcript": transcript,
        "audio": audio,
        "screenshot": screenshot,
        "messages": [],
        "tool_result": None,
        "tool_calls_count": 0,
    }
    result = graph.invoke(state, config={"recursion_limit": cfg.runtime.recursion_limit})
    final_response = result.get("final_response")
    if final_response:
        if terminal:
            terminal.response(final_response)
        else:
            print(final_response)
    elif result.get("tool_result") is not None:
        logger.info("No final model response; displaying tool result in CLI")
        response = json.dumps(result["tool_result"], indent=2, ensure_ascii=True)
        if terminal:
            terminal.response(response)
        else:
            print(response)
    else:
        logger.info("No final response or tool result produced")
        if terminal:
            terminal.response("[no response]")
        else:
            print("[no response]")


def _open_status_terminal() -> TerminalUI:
    if os.name != "nt":
        return TerminalUI()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(5.0)
    port = server.getsockname()[1]
    try:
        subprocess.Popen(
            [sys.executable, "-m", "jarvis.terminal_ui", "--port", str(port)],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        connection, _ = server.accept()
    except (OSError, TimeoutError) as exc:
        logger.warning("Could not open Jarvis status console: %s", exc)
        server.close()
        return TerminalUI()
    server.close()
    return SocketTerminalUI(connection)


def run() -> None:
    parser = ArgumentParser(description="Local Jarvis")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args()

    terminal = _open_status_terminal()
    terminal.startup("Booting systems...")

    cfg = load_config(args.config)
    configure_logging(cfg.runtime.log_level)

    terminal.startup("Loading model...")
    bundle = load_model_bundle(cfg)
    terminal.startup("Preparing agent...")
    graph = build_graph(bundle.model, bundle.processor, cfg, on_status=terminal.tool)

    listener = WakeWordListener(cfg.audio)
    terminal.startup("Systems ready.")

    def on_wake() -> None:
        logger.info("Wake event received")
        terminal.wake_triggered()
        _run_once(graph, cfg, terminal)

    if cfg.runtime.listen_on_startup:
        terminal.listening()
        listener.listen(on_wake, on_status=terminal.status)
    else:
        _run_once(graph, cfg, terminal)

    if isinstance(terminal, SocketTerminalUI):
        terminal.close()


if __name__ == "__main__":
    run()
