from __future__ import annotations

import logging
from argparse import ArgumentParser

from jarvis.audio_capture import record_command_audio
from jarvis.config import load_config
from jarvis.logging_setup import configure_logging
from jarvis.model_loader import load_model_bundle
from jarvis.screen_capture import capture_screen
from jarvis.stt import STTUnavailableError, transcribe_audio
from jarvis.wake_word import WakeWordListener
from jarvis.agent_graph import build_graph

logger = logging.getLogger(__name__)


def _resolve_transcript(audio, cfg) -> str:
    mode = cfg.audio.transcription_mode
    if mode == "direct":
        return "Please infer intent from provided audio and screenshot context."

    if mode == "stt":
        return transcribe_audio(audio, cfg.audio.sample_rate, cfg.audio.stt_model_id)

    if mode == "manual":
        return input("Command transcript: ").strip()

    # Default: STT fallback with manual backup for reliability.
    try:
        transcript = transcribe_audio(audio, cfg.audio.sample_rate, cfg.audio.stt_model_id)
        if transcript:
            return transcript
    except STTUnavailableError as exc:
        logger.warning("STT unavailable, falling back to manual transcript: %s", exc)

    return input("Command transcript (STT fallback): ").strip()


def _run_once(graph, cfg):
    screenshot = capture_screen(cfg.vision)
    audio = record_command_audio(cfg.audio)
    transcript = _resolve_transcript(audio, cfg)
    state = {
        "transcript": transcript,
        "audio": audio,
        "screenshot": screenshot,
        "messages": [],
        "tool_result": None,
        "tool_calls_count": 0,
    }
    result = graph.invoke(state, config={"recursion_limit": cfg.runtime.recursion_limit})
    print(result.get("final_response", "[no response]"))


def run() -> None:
    parser = ArgumentParser(description="Local Jarvis")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.runtime.log_level)

    bundle = load_model_bundle(cfg)
    graph = build_graph(bundle.model, bundle.processor, cfg)

    listener = WakeWordListener(cfg.audio)

    def on_wake() -> None:
        logger.info("Wake event received")
        _run_once(graph, cfg)

    if cfg.runtime.listen_on_startup:
        listener.listen(on_wake)
    else:
        _run_once(graph, cfg)


if __name__ == "__main__":
    run()
