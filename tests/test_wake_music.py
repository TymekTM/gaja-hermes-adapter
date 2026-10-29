"""Unit tests for the mixing/level/counting helpers in bench/wake_music.py (model-free)."""

from __future__ import annotations

import numpy as np
import pytest

from bench.wake_music import (
    DEFAULT_CUTOFFS,
    active_rms,
    calc_rms,
    count_events,
    cutoff_key,
    device_gain_chain,
    music_segment,
    quantise_cutoff,
    scale_active_dbfs,
    synth_music_block,
)


@pytest.fixture
def speech_like() -> np.ndarray:
    """Speech-shaped signal: loud middle section, quiet room-silence head and tail."""
    sr = 16000
    t = np.arange(2 * sr) / sr
    envelope = np.zeros_like(t)
    envelope[int(0.5 * sr):int(1.5 * sr)] = 1.0  # 1 s of "speech"
    carrier = np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
    rng = np.random.RandomState(7)
    signal = 0.05 * carrier + 0.01 * rng.standard_normal(len(t))
    return (signal * envelope).astype(np.float32)  # silence outside the "speech" span


def test_active_rms_ignores_room_silence(speech_like: np.ndarray) -> None:
    # the reference must match the speech span's RMS, not the whole-take RMS the silence drags down
    t = np.arange(speech_like.size) / 16000
    span = speech_like[(t >= 0.5) & (t < 1.5)]
    assert abs(active_rms(speech_like) - calc_rms(span)) < 0.02 * calc_rms(span)
    assert active_rms(speech_like) > calc_rms(speech_like)


def test_scale_active_dbfs_hits_the_target(speech_like: np.ndarray) -> None:
    for target in (-36.0, -42.0):
        scaled = scale_active_dbfs(speech_like, target)
        measured = 20 * np.log10(active_rms(scaled))
        assert abs(measured - target) < 0.5


def test_device_gain_chain_clips_and_quantises() -> None:
    x = np.array([0.1, 0.3, -0.5, 0.9], np.float32)  # gain 4 -> one value saturates
    y = device_gain_chain(x, gain=4.0)
    assert y.max() == 1.0  # clipped like int16 saturation
    grid = np.round(y * 32767.0)
    assert np.allclose(y * 32767.0, grid)  # on the int16 grid the device feeds the model
    assert y[0] == pytest.approx(0.4, abs=1 / 32767)


def test_quantise_cutoff_matches_deployed_manifest() -> None:
    # hej_gaja.json's probability_cutoff: the 0.961 grid value quantised like ESPHome does
    assert quantise_cutoff(0.961) == 0.9607843137254902


def test_music_segment_is_deterministic_and_fills_length() -> None:
    src = np.random.RandomState(3).standard_normal(16000 * 5).astype(np.float32)
    rng1, rng2 = np.random.default_rng([0, 1]), np.random.default_rng([0, 1])
    a, b = music_segment(src, 16000 * 3, rng1), music_segment(src, 16000 * 3, rng2)
    assert a.shape == b.shape == (16000 * 3,)
    assert np.array_equal(a, b)  # same seed -> same window
    short = np.random.RandomState(4).standard_normal(16000).astype(np.float32)
    tiled = music_segment(short, 16000 * 3, np.random.default_rng(1))
    assert tiled.shape == (16000 * 3,)  # shorter sources tile with faded seams


def test_synth_music_block_deterministic_bounded() -> None:
    rng = np.random.default_rng([9, 9])
    a = synth_music_block(2.0, rng)
    b = synth_music_block(2.0, np.random.default_rng([9, 9]))
    assert a.shape == (32000,)
    assert np.array_equal(a, b)  # identical blocks across passes (levels/filters)
    assert np.all(np.isfinite(a))
    assert np.abs(a).max() <= 0.7 + 1e-6
    assert calc_rms(a) > 0.001  # audible content, not silence


def test_count_events_respects_refractory() -> None:
    # peaks at frames 10, 40, 300 with a 2 s refractory at 0.03 s/frame -> gap 66 frames
    avg = np.zeros(400, np.float32)
    for f in (10, 40, 300):
        avg[f] = 1.0
    ev = count_events(avg, [0.5], refractory_s=2.0, step_s=0.03)
    assert ev[cutoff_key(0.5)] == 2  # 10/40 collapse into one, 300 counts again


def test_count_events_uses_the_device_cutoff() -> None:
    # values exactly at the raw 0.961 grid point: firmware compares against int(c*255)/255
    avg = np.full(10, 0.961, np.float32)  # 0.961 stored as float32 rounds just above 0.960784
    ev = count_events(avg, [0.961], refractory_s=2.0, step_s=0.03)
    assert ev[cutoff_key(0.961)] == 1  # one detection (the whole run is one exceedance block)


def test_default_grid_matches_the_brief() -> None:
    assert DEFAULT_CUTOFFS == [0.961, 0.9, 0.8, 0.7, 0.6, 0.5]
