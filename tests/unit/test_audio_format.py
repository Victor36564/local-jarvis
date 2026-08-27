import numpy as np

from jarvis.agent import infer_once
from jarvis.audio_capture import normalize_audio


def test_normalize_audio_clips_and_converts_to_mono():
    audio = np.array([[2.0, -2.0], [0.5, 0.5]], dtype=np.float64)
    out = normalize_audio(audio)

    assert out.dtype == np.float32
    assert out.ndim == 1
    assert np.max(np.abs(out)) <= 1.0


class _Processor:
    def __init__(self):
        self.prompt = ""

    def apply_chat_template(self, messages, **_kwargs):
        self.prompt = "\n".join(message["content"] for message in messages)
        return self.prompt

    def __call__(self, **_kwargs):
        class Value:
            def to(self, _device):
                return self

        return {"input_ids": Value()}

    def batch_decode(self, _output_ids, **_kwargs):
        return ["Done"]


class _Model:
    device = "cpu"

    def generate(self, **_kwargs):
        return [[1]]


def test_infer_keeps_modality_tokens_on_user_message_after_tool_result():
    processor = _Processor()

    result = infer_once(
        _Model(),
        processor,
        "check the screen",
        screenshot=object(),
        audio=None,
        max_new_tokens=8,
        tool_result={"stdout": "gpu result"},
    )

    assert result == {"type": "final", "payload": "Done"}
    assert "<|image|>" in processor.prompt
