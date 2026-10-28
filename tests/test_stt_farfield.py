"""Unit tests for far-field degradation and gain helpers in bench/stt_farfield.py."""

from __future__ import annotations

import numpy as np
import pytest

from bench.stt_farfield import (
    apply_gain,
    calc_dbfs,
    calc_rms,
    degrade_quiet,
    degrade_room,
    degrade_room_noise,
    make_room_ir,
    peak_normalise,
    scale_to_rms_dbfs,
)


@pytest.fixture
def synthetic_speech_like() -> np.ndarray:
    """Deterministic synthetic test signal with varying amplitude and frequency."""
    rng = np.random.RandomState(1234)
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    carrier = np.sin(2 * np.pi * 300 * t) * 0.4
    modulator = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
    noise = rng.normal(0, 0.05, 16000).astype(np.float32)
    signal = (carrier * modulator + noise).astype(np.float32)
    return signal


# 1. Checks that the RMS after scaling is within 0.5 dB of the target


@pytest.mark.parametrize("target_dbfs", [-36.0, -26.0, -20.0, -15.0])
def test_rms_after_scaling_is_within_half_db(synthetic_speech_like: np.ndarray, target_dbfs: float) -> None:
    scaled = scale_to_rms_dbfs(synthetic_speech_like, target_dbfs)
    measured_dbfs = calc_dbfs(scaled)
    assert abs(measured_dbfs - target_dbfs) < 0.5


def test_degrade_quiet_rms_target(synthetic_speech_like: np.ndarray) -> None:
    quiet = degrade_quiet(synthetic_speech_like)
    measured_dbfs = calc_dbfs(quiet)
    assert abs(measured_dbfs - (-36.0)) < 0.5


def test_degrade_room_rms_target(synthetic_speech_like: np.ndarray) -> None:
    room = degrade_room(synthetic_speech_like, seed=42)
    measured_dbfs = calc_dbfs(room)
    assert abs(measured_dbfs - (-36.0)) < 0.5


def test_degrade_room_noise_rms_target(synthetic_speech_like: np.ndarray) -> None:
    # Adding 20 dB SNR noise increases total energy by only ~0.04 dB (10*log10(1 + 0.01))
    noisy = degrade_room_noise(synthetic_speech_like, snr_db=20.0, seed=42)
    measured_dbfs = calc_dbfs(noisy)
    assert abs(measured_dbfs - (-36.0)) < 0.5


def test_peak_normalise_target(synthetic_speech_like: np.ndarray) -> None:
    normalised = peak_normalise(synthetic_speech_like, target_peak=0.5)
    peak = float(np.max(np.abs(normalised)))
    assert peak == pytest.approx(0.5, abs=1e-5)


@pytest.mark.parametrize("gain_config, expected_dbfs", [
    ("-26dBFS", -26.0),
    ("-20dBFS", -20.0),
])
def test_apply_gain_rms_configs(synthetic_speech_like: np.ndarray, gain_config: str, expected_dbfs: float) -> None:
    quiet = degrade_quiet(synthetic_speech_like)
    boosted = apply_gain(quiet, gain_config)
    measured_dbfs = calc_dbfs(boosted)
    assert abs(measured_dbfs - expected_dbfs) < 0.5


# 2. Checks that the same seed gives the same output


def test_same_seed_gives_identical_room_ir() -> None:
    ir1 = make_room_ir(rt60=0.4, sample_rate=16000, seed=999)
    ir2 = make_room_ir(rt60=0.4, sample_rate=16000, seed=999)
    assert np.array_equal(ir1, ir2)

    ir_diff = make_room_ir(rt60=0.4, sample_rate=16000, seed=1000)
    assert not np.array_equal(ir1, ir_diff)


def test_same_seed_gives_identical_room_degradation(synthetic_speech_like: np.ndarray) -> None:
    run1 = degrade_room(synthetic_speech_like, seed=42)
    run2 = degrade_room(synthetic_speech_like, seed=42)
    assert np.array_equal(run1, run2)

    run_other_seed = degrade_room(synthetic_speech_like, seed=43)
    assert not np.array_equal(run1, run_other_seed)


def test_same_seed_gives_identical_room_noise_degradation(synthetic_speech_like: np.ndarray) -> None:
    run1 = degrade_room_noise(synthetic_speech_like, snr_db=20.0, seed=42)
    run2 = degrade_room_noise(synthetic_speech_like, snr_db=20.0, seed=42)
    assert np.array_equal(run1, run2)

    run_other_seed = degrade_room_noise(synthetic_speech_like, snr_db=20.0, seed=43)
    assert not np.array_equal(run1, run_other_seed)


# 3. Checks that silence is handled without NaN


def test_silence_handled_without_nan() -> None:
    silence = np.zeros(16000, dtype=np.float32)

    # RMS & dBFS of pure silence
    assert calc_rms(silence) == 0.0
    assert not np.isnan(calc_dbfs(silence))
    assert not np.isinf(calc_dbfs(silence))

    # Scaling pure silence
    scaled = scale_to_rms_dbfs(silence, -26.0)
    assert not np.isnan(scaled).any()
    assert not np.isinf(scaled).any()
    assert np.array_equal(scaled, silence)

    # Peak normalise pure silence
    peaked = peak_normalise(silence, 0.5)
    assert not np.isnan(peaked).any()
    assert not np.isinf(peaked).any()
    assert np.array_equal(peaked, silence)

    # Degrade quiet with silence
    quiet = degrade_quiet(silence)
    assert not np.isnan(quiet).any()
    assert not np.isinf(quiet).any()
    assert np.array_equal(quiet, silence)

    # Degrade room with silence
    room = degrade_room(silence, seed=42)
    assert not np.isnan(room).any()
    assert not np.isinf(room).any()
    assert np.array_equal(room, silence)

    # Degrade room_noise with silence
    room_noise = degrade_room_noise(silence, snr_db=20.0, seed=42)
    assert not np.isnan(room_noise).any()
    assert not np.isinf(room_noise).any()
    assert np.array_equal(room_noise, silence)

    # Apply gain on silence
    for cfg in ["none", "-26dBFS", "-20dBFS", "peak0.5"]:
        out = apply_gain(silence, cfg)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()
        assert np.array_equal(out, silence)


def test_empty_audio_handled_without_nan() -> None:
    empty = np.array([], dtype=np.float32)
    assert calc_rms(empty) == 0.0
    assert not np.isnan(calc_dbfs(empty))
    assert scale_to_rms_dbfs(empty, -20.0).size == 0
    assert peak_normalise(empty, 0.5).size == 0
    assert degrade_quiet(empty).size == 0
    assert degrade_room(empty).size == 0
    assert degrade_room_noise(empty).size == 0
