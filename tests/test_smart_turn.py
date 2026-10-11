from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from gaja.audio.smart_turn import (
    MAX_SAMPLES,
    N_FFT,
    N_MELS,
    SmartTurn,
    compute_slaney_mel_filters,
    extract_features,
    resolve_smart_turn_model,
    truncate_or_pad_audio,
)

CLIP = Path("bench/out/input_pl_16k.wav")


def test_slaney_mel_filters_shape() -> None:
    filters = compute_slaney_mel_filters()
    assert filters.shape == (N_FFT // 2 + 1, N_MELS)
    assert filters.shape == (201, 80)
    assert not np.isnan(filters).any()
    assert (filters >= 0.0).all()


def test_truncate_or_pad_audio() -> None:
    # Shorter than 8s: pads at beginning
    short = np.ones(1000, dtype=np.float32)
    padded = truncate_or_pad_audio(short, max_samples=4000)
    assert len(padded) == 4000
    assert (padded[:3000] == 0.0).all()
    assert (padded[3000:] == 1.0).all()

    # Exact length: no change
    exact = np.ones(4000, dtype=np.float32)
    assert np.array_equal(truncate_or_pad_audio(exact, max_samples=4000), exact)

    # Longer: keeps last N samples
    longer = np.arange(5000, dtype=np.float32)
    truncated = truncate_or_pad_audio(longer, max_samples=4000)
    assert len(truncated) == 4000
    assert np.array_equal(truncated, longer[1000:])


def test_extract_features_shape() -> None:
    dummy = np.random.randn(32000).astype(np.float32)
    features = extract_features(dummy)
    assert features.shape == (1, 80, 800)
    assert features.dtype == np.float32
    assert not np.isnan(features).any()


def test_numerical_match_with_whisper_feature_extractor() -> None:
    try:
        from transformers import WhisperFeatureExtractor
    except ImportError:
        pytest.skip("transformers not installed")

    fe = WhisperFeatureExtractor(chunk_length=8)
    rng = np.random.default_rng(42)
    dummy = rng.standard_normal(48000).astype(np.float32)

    ref = fe(
        dummy[-MAX_SAMPLES:] if len(dummy) > MAX_SAMPLES else np.pad(dummy, (MAX_SAMPLES - len(dummy), 0)),
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=MAX_SAMPLES,
        truncation=True,
        do_normalize=True,
    ).input_features[0]

    mine = extract_features(dummy)[0]
    diff = float(np.max(np.abs(ref - mine)))
    assert diff < 1e-4, f"Max difference {diff} exceeds tolerance 1e-4"


def test_smart_turn_model_prediction() -> None:
    try:
        model_path = resolve_smart_turn_model()
    except FileNotFoundError:
        pytest.skip("smart-turn model file not found")

    st = SmartTurn(model_path, threads=1)
    silence = np.zeros(16000, dtype=np.float32)
    p_sil = st.predict_probability(silence)
    assert 0.0 <= p_sil <= 1.0

    # Test with real clip if available
    if CLIP.is_file():
        import soundfile as sf

        audio, _ = sf.read(CLIP, dtype="float32")
        # Full utterance should have high completion probability
        p_full = st.predict_probability(audio)
        assert p_full > 0.7

        # Cut utterance early (e.g. 35% through) should have lower completion probability
        p_cut = st.predict_probability(audio[: int(len(audio) * 0.35)])
        assert p_cut < p_full
