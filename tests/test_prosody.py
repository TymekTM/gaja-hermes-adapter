import numpy as np
import pytest

from gaja.text.style import STYLE_PRESETS, STYLE_TAGS, Style
from gaja.tts.prosody import apply_post, engine_params, pitch_shift

SR = 22050


def _harmonic_speechlike(seconds: float = 3.0, f0: float = 140.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t) ** 2
    wave = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 12))
    return (0.6 * env * wave / np.max(np.abs(wave))).astype(np.float32)


def _f0_autocorr(x: np.ndarray, lo: float = 70.0, hi: float = 300.0) -> float:
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    min_lag, max_lag = int(SR / hi), int(SR / lo)
    lag = min_lag + int(np.argmax(ac[min_lag:max_lag]))
    return SR / lag


@pytest.mark.parametrize("semitones", [-2.5, -1.0, 1.0, 2.5])
def test_pitch_shift_preserves_duration_within_two_percent(semitones):
    x = _harmonic_speechlike()
    y = pitch_shift(x, SR, semitones)
    assert abs(len(y) - len(x)) / len(x) < 0.02


@pytest.mark.parametrize("semitones", [-2.0, 2.0])
def test_pitch_shift_moves_f0_by_requested_semitones(semitones):
    x = _harmonic_speechlike()
    y = pitch_shift(x, SR, semitones)
    measured = 12 * np.log2(_f0_autocorr(y[SR // 2:]) / _f0_autocorr(x[SR // 2:]))
    assert measured == pytest.approx(semitones, abs=0.3)


def test_pitch_shift_zero_is_identity_copy():
    x = _harmonic_speechlike(0.5)
    y = pitch_shift(x, SR, 0.0)
    np.testing.assert_array_equal(x, y)
    assert y is not x


def test_pitch_shift_very_short_input_is_returned_unchanged():
    x = np.zeros(100, dtype=np.float32)
    np.testing.assert_array_equal(pitch_shift(x, SR, 2.0), x)


@pytest.mark.parametrize("name", STYLE_TAGS)
def test_apply_post_every_preset_is_finite_and_bounded(name):
    x = _harmonic_speechlike(2.0)
    y = apply_post(x, SR, STYLE_PRESETS[name])
    assert y.dtype == np.float32
    assert np.isfinite(y).all()
    assert np.max(np.abs(y)) <= 1.0


def test_apply_post_soft_clips_loud_input_without_nans():
    x = np.clip(_harmonic_speechlike(1.0) * 3.0, -1.0, 1.0).astype(np.float32)
    loud = Style(energy=1.4, pitch_semitones=2.0)
    y = apply_post(x, SR, loud)
    assert np.isfinite(y).all()
    assert np.max(np.abs(y)) <= 1.0


def test_sad_adds_trailing_pause_and_keeps_speech_intact():
    x = _harmonic_speechlike(1.0)
    y = apply_post(x, SR, Style(emotion="sad"))
    pause = len(y) - len(x)
    assert pause == pytest.approx(0.15 * SR, abs=2)
    assert np.all(y[-pause:] == 0)


def test_neutral_style_is_passthrough():
    x = _harmonic_speechlike(1.0)
    np.testing.assert_allclose(apply_post(x, SR, Style()), x, atol=1e-6)


def test_laugh_and_sigh_are_pauses_not_audio():
    x = _harmonic_speechlike(1.0)
    y = apply_post(x, SR, Style(events=("laugh",)))
    assert len(y) > len(x)
    z = apply_post(x, SR, Style(events=("sigh",)))
    assert len(z) - len(x) == pytest.approx(0.45 * SR, abs=2)


def test_empty_audio_returns_empty():
    assert apply_post(np.zeros(0, dtype=np.float32), SR, Style(pitch_semitones=2.0)).size == 0


def test_engine_params_piper_maps_rate_to_length_scale():
    params = engine_params("piper", Style(emotion="excited", rate=1.25, energy=1.2))
    assert params["length_scale"] == pytest.approx((1 / 1.25) ** 1.5)
    assert 0.4 <= params["noise_scale"] <= 0.95
    assert 0.5 <= params["noise_w_scale"] <= 1.1


def test_engine_params_piper_neutral_defaults():
    params = engine_params("piper", Style())
    assert params["length_scale"] == pytest.approx(1.0)
    assert params["noise_scale"] == pytest.approx(0.667)
    assert params["noise_w_scale"] == pytest.approx(0.8)


def test_engine_params_paradee_speed():
    assert engine_params("paradee", Style(rate=0.85)) == {"speed": 0.85}


def test_engine_params_supertonic_speed():
    assert engine_params("supertonic", Style(rate=1.18)) == {"speed": 1.18}


def test_engine_params_unknown_engine_is_empty():
    assert engine_params("chatterbox-turbo", Style(rate=1.2)) == {}


def test_apply_post_skips_pitch_shift_unless_enabled():
    from gaja.tts import prosody

    x = _harmonic_speechlike(1.0) * 0.5
    style = Style(emotion="happy", pitch_semitones=2.0)
    prosody.configure(pitch_shift=False)
    np.testing.assert_allclose(apply_post(x, SR, style), x, atol=1e-6)
    prosody.configure(pitch_shift=True)
    try:
        assert not np.allclose(apply_post(x, SR, style), x, atol=1e-3)
    finally:
        prosody.configure(pitch_shift=False)


def test_apply_post_gain_never_clips_full_scale_speech():
    x = _harmonic_speechlike(1.0)
    x = (x / np.max(np.abs(x))).astype(np.float32)  # Piper output peaks at ~1.0
    y = apply_post(x, SR, Style(emotion="excited", energy=1.2))
    assert np.max(np.abs(y)) <= 0.95 + 1e-6
    # a uniform scale, not a squashed waveform: the shape is preserved
    np.testing.assert_allclose(y / np.max(np.abs(y)), x, atol=1e-5)
