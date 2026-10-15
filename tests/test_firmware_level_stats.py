"""Host-side check of the firmware's speech-level tracker.

Compiles ``firmware/esphome/components/gaja_device/level_stats.h`` (pure C++) and asserts the
noise-gate semantics the adaptive reply volume and the follow-up listen_end logic rely on:
DC offset must not count as level, a silent window is "no speech", speech above the floor is
detected, and percentiles are taken over gated speech frames only.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "esphome" / "components" / "gaja_device" / "level_stats.h"

MAIN = r"""
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include "level_stats.h"
using esphome::gaja_device::LevelTracker;
static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}

static uint32_t seed = 12345;
static uint32_t rnd() {  // xorshift: deterministic "noise"
  seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
  return seed;
}

static constexpr float TWO_PI = 6.28318530718f;

// A 20 ms PCM16 mono frame: a sine whose RMS lands at `speech_dbfs` + white noise of amplitude
// `noise_amp` counts + a DC offset. speech_dbfs well below the noise makes an (almost) silent
// frame; the device's gain_factor is irrelevant here (both speech and noise carry it).
static std::vector<uint8_t> frame(float speech_dbfs, float noise_amp, float dc, float freq_hz = 300.0f) {
  const size_t samples = 320;
  std::vector<uint8_t> bytes(samples * 2);
  int16_t *pcm = (int16_t *) bytes.data();
  const float amp = std::pow(10.0f, speech_dbfs / 20.0f) * std::sqrt(2.0f) * 32767.0f;
  for (size_t i = 0; i < samples; i++) {
    const float t = (float) i / 16000.0f;
    float v = amp * std::sin(TWO_PI * freq_hz * t);
    v += (float) ((int32_t) (rnd() % 2001) - 1000) / 1000.0f * noise_amp;
    pcm[i] = (int16_t) (v + dc);
  }
  return bytes;
}

static void note(LevelTracker &lt, const std::vector<uint8_t> &f) { lt.note_frame(f.data(), f.size()); }

static void test_dc_offset_is_not_level() {
  LevelTracker a, b;
  for (int i = 0; i < 50; i++) {
    note(a, frame(-40.0f, 0.0f, 0.0f));    // -40 dBFS tone, no DC
    note(b, frame(-40.0f, 0.0f, 8000.0f)); // same tone on an 8000-count DC offset
  }
  check(std::fabs(a.max_db() - b.max_db()) < 1.0f, "DC offset does not add level");
  check(b.max_db() < -35.0f, "DC-dominated frame stays quiet");
}

static void test_silent_window_is_no_speech() {
  LevelTracker lt;
  for (int i = 0; i < 100; i++) {
    note(lt, frame(-70.0f, 60.0f, 0.0f));  // noise floor only
  }
  check(!lt.speech_seen(), "noise-only window is no speech");
  check(std::isnan(lt.speech_percentile_db(0.9f)), "no speech percentile is NAN");
}

static void test_speech_detected_and_gated() {
  LevelTracker lt;
  for (int i = 0; i < 30; i++) {
    note(lt, frame(-70.0f, 60.0f, 0.0f));  // silence before the utterance (floor)
  }
  for (int i = 0; i < 50; i++) {
    note(lt, frame(-18.0f, 60.0f, 0.0f));  // speech at -18 dBFS
  }
  for (int i = 0; i < 30; i++) {
    note(lt, frame(-70.0f, 60.0f, 0.0f));  // trailing silence
  }
  check(lt.speech_seen(), "speech above the gate is seen");
  check(lt.frames() == 110, "all frames counted");
  check(lt.speech_frames() >= 45 && lt.speech_frames() < 110, "gated frames are a subset");
  const float p90 = lt.speech_percentile_db(0.9f);
  check(!std::isnan(p90), "speech percentile exists");
  check(p90 > -26.0f && p90 < -12.0f, "p90 near the speech level, not the floor");
  check(lt.floor_db() < -50.0f, "floor is the noise floor");
}

static void test_clear_resets() {
  LevelTracker lt;
  for (int i = 0; i < 3; i++) {
    note(lt, frame(-70.0f, 60.0f, 0.0f));
  }
  for (int i = 0; i < 50; i++) {
    note(lt, frame(-18.0f, 60.0f, 0.0f));
  }
  check(lt.speech_seen(), "speech before clear");
  lt.clear();
  check(lt.frames() == 0 && lt.speech_frames() == 0, "clear resets counters");
  check(!lt.speech_seen(), "no speech after clear");
  // A mic window on this half-duplex device always starts with the mic's own silence (it was
  // off during playback), so speech right after the window opens must still be detected.
  for (int i = 0; i < 3; i++) {
    note(lt, frame(-70.0f, 60.0f, 0.0f));
  }
  for (int i = 0; i < 10; i++) {
    note(lt, frame(-24.0f, 60.0f, 0.0f));  // quiet speech just 4 dB above the floor
  }
  check(lt.speech_seen(), "quiet speech after window open is detected");
}

static void test_partial_and_empty() {
  LevelTracker lt;
  std::vector<uint8_t> f = frame(-18.0f, 60.0f, 0.0f);
  lt.note_frame(f.data(), 641);  // odd length: last byte ignored
  check(lt.frames() == 1, "odd-length frame counted once");
  lt.note_frame(f.data(), 1);
  check(lt.frames() == 1, "sub-sample frame ignored");
  lt.note_frame(f.data(), 0);
  check(lt.frames() == 1, "empty frame ignored");
}

int main() {
  test_dc_offset_is_not_level();
  test_silent_window_is_no_speech();
  test_speech_detected_and_gated();
  test_clear_resets();
  test_partial_and_empty();
  if (failures) { std::printf("%d level-stat failures\n", failures); return 1; }
  std::printf("level stats OK\n");
  return 0;
}
"""

pytestmark = pytest.mark.skipif(
    shutil.which("c++") is None and shutil.which("g++") is None and shutil.which("clang++") is None,
    reason="no host C++ compiler",
)


def _cxx() -> str:
    for candidate in ("c++", "g++", "clang++"):
        if shutil.which(candidate) is not None:
            return candidate
    raise AssertionError("unreachable: skipif above guarantees a compiler")


def test_level_stats() -> None:
    with tempfile.TemporaryDirectory(prefix="gaja-levels-") as tmp:
        main = Path(tmp) / "main.cpp"
        main.write_text(MAIN)
        exe = Path(tmp) / "levels"
        subprocess.run([_cxx(), "-std=c++17", "-I", str(HEADER.parent), str(main), "-o", str(exe)], check=True)
        result = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    assert "level stats OK" in result.stdout
    assert "FAIL" not in result.stdout
