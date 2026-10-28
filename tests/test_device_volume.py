"""Per-device speaker volume: perceptual gain math and the persisted level store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gaja.devices.volume import (
    DEFAULT_LEVEL,
    DEFAULT_STEP,
    MAX_ATTENUATION_DB,
    STEP_FLOOR,
    VolumeStore,
    gain_for_level,
    get_volume_store,
    set_volume_store,
)


# ---- gain math ------------------------------------------------------------------------------------


def test_gain_at_100_is_unity_so_default_changes_nothing() -> None:
    assert gain_for_level(DEFAULT_LEVEL) == 1.0


def test_gain_at_zero_is_silence() -> None:
    assert gain_for_level(0) == 0.0


def test_gain_is_perceptual_db_linear() -> None:
    # 50 % sits exactly half-way through the attenuation range in dB.
    expected = 10 ** (-MAX_ATTENUATION_DB / 2 / 20)
    assert gain_for_level(50) == pytest.approx(expected, rel=1e-6)
    # A step of 12 % is the same dB change anywhere on the scale ...
    assert 20 * np.log10(gain_for_level(88) / gain_for_level(100)) == pytest.approx(
        20 * np.log10(gain_for_level(40) / gain_for_level(52)), rel=1e-6
    )


def test_gain_is_monotonic_and_steps_are_audible() -> None:
    gains = [gain_for_level(level) for level in range(0, 101)]
    assert all(b >= a for a, b in zip(gains, gains[1:]))
    # One default step must be clearly audible (>= 4 dB), never more than a jump scare (<= 7 dB).
    step_db = 20 * np.log10(gain_for_level(100 - DEFAULT_STEP) / gain_for_level(100))
    assert 4.0 <= -step_db <= 7.0


def test_one_default_step_is_the_documented_5_4_db() -> None:
    """45 dB over the 0-100 scale: 12 points are 12 x 0.45 dB = 5.4 dB (docs/DEVICE.md)."""
    step_db = 20 * np.log10(gain_for_level(100 - DEFAULT_STEP) / gain_for_level(100))
    assert -step_db == pytest.approx(MAX_ATTENUATION_DB / 100 * DEFAULT_STEP, rel=1e-6)
    assert -step_db == pytest.approx(5.4, abs=0.01)


def test_gain_stays_in_int16_safe_bounds_for_every_level() -> None:
    # Worst case input: full-scale square-ish wave.
    worst = np.full(64, 32767, dtype="<i2")
    for level in range(0, 101):
        out = (worst.astype(np.float32) * gain_for_level(level)).astype(np.float64)
        assert out.max() <= 32767.0 + 1e-6
        assert out.min() >= -32768.0 - 1e-6


# ---- store: clamping, steps, floor ----------------------------------------------------------------


def test_set_level_clamps_and_reports_mute(tmp_path: Path) -> None:
    store = VolumeStore(tmp_path / "volumes.json")
    assert store.set_level("atom-pokoj", 150) == 100
    assert store.set_level("atom-pokoj", -3) == 0
    assert store.level("atom-pokoj") == 0
    assert store.is_muted("atom-pokoj")


def test_explicit_zero_mutes_but_stepping_cannot_reach_zero(tmp_path: Path) -> None:
    store = VolumeStore(tmp_path / "volumes.json")
    store.set_level("atom-pokoj", 30)
    # Repeated "ciszej" stops at the floor, never silently at 0.
    level = 30
    for _ in range(20):
        level = store.step_level("atom-pokoj", -DEFAULT_STEP)
    assert level == STEP_FLOOR
    assert store.level("atom-pokoj") == STEP_FLOOR
    assert not store.is_muted("atom-pokoj")
    # Only an explicit set (or mute) may silence the device.
    store.set_level("atom-pokoj", 0)
    assert store.is_muted("atom-pokoj")


def test_step_up_from_mute_is_audible_again(tmp_path: Path) -> None:
    store = VolumeStore(tmp_path / "volumes.json")
    store.set_level("atom-pokoj", 0)
    assert store.step_level("atom-pokoj", DEFAULT_STEP) == DEFAULT_STEP


def test_step_clamps_at_max_and_floor(tmp_path: Path) -> None:
    store = VolumeStore(tmp_path / "volumes.json")
    assert store.step_level("atom-pokoj", 12) == DEFAULT_LEVEL  # unknown device starts at 100
    store.set_level("atom-pokoj", 95)
    assert store.step_level("atom-pokoj", 12) == 100
    store.set_level("atom-pokoj", 8)
    assert store.step_level("atom-pokoj", -12) == STEP_FLOOR


def test_unknown_device_defaults_to_100(tmp_path: Path) -> None:
    store = VolumeStore(tmp_path / "volumes.json")
    assert store.level("never-seen") == DEFAULT_LEVEL
    assert store.gain("never-seen") == 1.0


# ---- persistence ----------------------------------------------------------------------------------


def test_level_persists_across_store_instances(tmp_path: Path) -> None:
    path = tmp_path / "volumes.json"
    store = VolumeStore(path)
    store.set_level("atom-pokoj", 40)
    assert json.loads(path.read_text(encoding="utf-8"))["levels"]["atom-pokoj"] == 40

    reloaded = VolumeStore(path)
    assert reloaded.level("atom-pokoj") == 40
    assert reloaded.gain("atom-pokoj") == pytest.approx(gain_for_level(40), rel=1e-6)


def test_save_is_atomic_and_leaves_no_tmp_file(tmp_path: Path) -> None:
    path = tmp_path / "volumes.json"
    store = VolumeStore(path)
    store.set_level("atom-pokoj", 40)
    assert list(tmp_path.iterdir()) == [path]


def test_corrupt_file_keeps_defaults_instead_of_crashing(tmp_path: Path) -> None:
    path = tmp_path / "volumes.json"
    path.write_text("{not json", encoding="utf-8")
    store = VolumeStore(path)
    assert store.level("atom-pokoj") == DEFAULT_LEVEL
    # And saving afterwards repairs the file.
    store.set_level("atom-pokoj", 60)
    assert VolumeStore(path).level("atom-pokoj") == 60


def test_in_memory_store_without_path_never_writes(tmp_path: Path) -> None:
    store = VolumeStore(None)
    store.set_level("atom-pokoj", 30)
    assert store.level("atom-pokoj") == 30
    assert list(tmp_path.iterdir()) == []


# ---- module-level singleton (tools read it like get_timer_manager) --------------------------------


def test_module_level_getter_and_setter() -> None:
    store = VolumeStore(None)
    set_volume_store(store)
    assert get_volume_store() is store
    set_volume_store(None)
    assert get_volume_store() is None
