import numpy as np

from jarvis.audio_capture import normalize_audio


def test_normalize_audio_clips_and_converts_to_mono():
    audio = np.array([[2.0, -2.0], [0.5, 0.5]], dtype=np.float64)
    out = normalize_audio(audio)

    assert out.dtype == np.float32
    assert out.ndim == 1
    assert np.max(np.abs(out)) <= 1.0
