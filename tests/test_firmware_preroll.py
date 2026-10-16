"""Host-side check of the firmware's pre-roll start computation (the on_wake cursor).

Compiles ``firmware/esphome/components/gaja_device/preroll.h`` (pure C++) and asserts the
semantics the wake path relies on (review B-1):

- the pre-roll never starts more than ``preroll_ms`` behind the ring head (the server verifies
  only the FIRST 2.5 s of what we send -- the whole retained window would push the wake phrase
  out of that window and every wake would be rejected),
- it stays one mic-chunk margin ahead of the ring tail (the producer must not lap -- and half
  overwrite -- the first frame while the burst is in flight),
- it never starts before the last mic (re)start (audio from a previous turn is not pre-roll),
- all of that survives the 32-bit position wrap (positions are modular counters).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "esphome" / "components" / "gaja_device" / "preroll.h"

MAIN = r"""
#include <cstdio>
#include <cstdint>
#include "preroll.h"
using esphome::gaja_device::preroll_start_pos;
using esphome::gaja_device::lap_recovery_pos;
using esphome::gaja_device::ring_pos_diff;
using esphome::gaja_device::ring_pos_t;
static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}

static constexpr uint32_t BPM = 32;  // bytes per ms at 16 kHz PCM16
static constexpr uint32_t MARGIN = 1024;

int main() {
  // A long-idle ring: 4 s retained (preroll 1500 ms + slack), tail far behind the head. The
  // start must be capped at the configured pre-roll, NOT at the oldest retained byte.
  {
    const ring_pos_t head = 40'000'000;
    const ring_pos_t tail = head - 4000 * BPM;   // 4 s window
    const ring_pos_t valid_from = 3'000'000;     // mic restarted ages ago
    uint32_t got = 9999;
    const ring_pos_t start = preroll_start_pos(head, tail, valid_from, 1500 * BPM, MARGIN, &got);
    check(start == head - 1500 * BPM, "start capped at preroll_ms behind head");
    check(got == 1500 * BPM, "announced preroll equals the configured cap");
  }

  // The margin: the start must never sit on (or within one mic chunk of) the oldest byte the
  // producer is about to overwrite -- a tail-exact cursor gets lapped by the next 16 ms chunk.
  {
    const ring_pos_t head = 1'000'000;
    const ring_pos_t tail = head - 2000 * BPM;
    const ring_pos_t start = preroll_start_pos(head, tail, 0, 5000 * BPM, MARGIN, nullptr);
    check(ring_pos_diff(start, tail) >= (int32_t) MARGIN, "start keeps the tail margin");
    check(ring_pos_diff(head, start) == 2000 * BPM - MARGIN, "window shorter than preroll: everything retained");
  }

  // A fresh ring (younger than the pre-roll): no underflow, the start is the tail plus the
  // margin (or the head when even the margin does not fit).
  {
    const ring_pos_t head = 500 * BPM;           // ring only ever saw 500 ms
    const ring_pos_t tail = 0;
    const ring_pos_t start = preroll_start_pos(head, tail, 0, 1500 * BPM, MARGIN, nullptr);
    check(start == tail + MARGIN, "young ring: start is the tail plus the margin");
  }
  {
    const ring_pos_t head = MARGIN / 2;          // barely anything buffered
    const ring_pos_t start = preroll_start_pos(head, 0, 0, 1500 * BPM, MARGIN, nullptr);
    check(start == head, "ring smaller than the margin: start at the head");
  }
  {
    const ring_pos_t head = 0;
    const ring_pos_t start = preroll_start_pos(0, 0, 0, 1500 * BPM, MARGIN, nullptr);
    check(start == 0, "empty ring: start at 0");
  }

  // The last mic restart wins over the configured cap: audio from a previous turn (which may
  // contain our own playback) must never be sent as pre-roll.
  {
    const ring_pos_t head = 9'000'000;
    const ring_pos_t tail = head - 4000 * BPM;
    const ring_pos_t valid_from = head - 700 * BPM;  // restarted 700 ms ago (a follow-up)
    uint32_t got = 9999;
    const ring_pos_t start = preroll_start_pos(head, tail, valid_from, 1500 * BPM, MARGIN, &got);
    check(start == valid_from, "valid_from is the newest bound");
    check(got == 700 * BPM, "announced preroll matches the real window");
  }
  {
    // valid_from exactly at the head (restart right now): zero pre-roll, no underflow.
    const ring_pos_t head = 500'000;
    const ring_pos_t start = preroll_start_pos(head, head - 4000 * BPM, head, 1500 * BPM, MARGIN, nullptr);
    check(start == head, "just restarted: pre-roll starts now");
  }
  {
    // valid_from slightly AHEAD of the head: only possible through the 32-bit wrap (the mic
    // restarted ~2^32 bytes ago), i.e. an ancient restart -- the bound is ignored.
    const ring_pos_t head = 500'000;
    const ring_pos_t start = preroll_start_pos(head, head - 4000 * BPM, head + 16, 1500 * BPM, MARGIN, nullptr);
    check(start == head - 1500 * BPM, "wrap-artifact valid_from is ignored");
  }

  // Positions wrap at 2^32 (~37 h of mic uptime): everything above must behave identically
  // with the head a few bytes below the wrap point.
  {
    const ring_pos_t head = 0xFFFFFFF0u;
    const ring_pos_t tail = head - 4000 * BPM;
    const ring_pos_t valid_from = head - 700 * BPM;
    uint32_t got = 9999;
    const ring_pos_t start = preroll_start_pos(head, tail, valid_from, 1500 * BPM, MARGIN, &got);
    check(start == valid_from, "wrap: valid_from bound");
    check(got == 700 * BPM, "wrap: announced preroll");
    const ring_pos_t start2 = preroll_start_pos(head, tail, head - 2'000'000'000, 1500 * BPM, MARGIN, nullptr);
    check(start2 == (ring_pos_t) (head - 1500 * BPM), "wrap: preroll cap");
    check(ring_pos_diff(start2, tail) > 0, "wrap: start stays ahead of the tail");
  }

  // N-6: lap recovery resumes a margin above the tail (not AT the tail, where the producer is
  // about to overwrite), and never past the head.
  {
    const ring_pos_t head = 70'000'000;
    const ring_pos_t tail = head - 65536;
    check(lap_recovery_pos(head, tail, 4096) == tail + 4096, "lap: tail plus the margin");
    check(ring_pos_diff(lap_recovery_pos(head, tail, 4096), tail) > 0, "lap: not at the tail");
    check(lap_recovery_pos(head, head - 100, 4096) == head, "lap: window below the margin -> head");
    check(lap_recovery_pos(head, head - 4096, 4096) == head, "lap: window == margin -> head");
    check(lap_recovery_pos(head, head, 4096) == head, "lap: empty window -> head");
    const ring_pos_t whead = 0x00000800u;  // 2048 past the 2^32 wrap
    const ring_pos_t wtail = whead - 65536;
    check(lap_recovery_pos(whead, wtail, 4096) == wtail + 4096, "lap: modular across the wrap");
  }

  if (failures) { std::printf("%d preroll failures\n", failures); return 1; }
  std::printf("preroll OK\n");
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


def test_preroll_start() -> None:
    with tempfile.TemporaryDirectory(prefix="gaja-preroll-") as tmp:
        main = Path(tmp) / "main.cpp"
        main.write_text(MAIN)
        exe = Path(tmp) / "preroll"
        subprocess.run([_cxx(), "-std=c++17", "-I", str(HEADER.parent), str(main), "-o", str(exe)], check=True)
        result = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    assert "preroll OK" in result.stdout
    assert "FAIL" not in result.stdout
