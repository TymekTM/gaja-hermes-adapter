"""Host-side check of the firmware's SPSC and rolling ring buffers.

Compiles ``firmware/esphome/components/gaja_device/rings.h`` (pure C++, no ESPHome includes)
with the system C++ compiler and asserts the semantics the gaja_device component relies on:
fragment ordering across the wrap, drop-on-full, consumer-side clear, and the absolute
positions of the rolling pre-roll window. Skipped when no C++ compiler is available.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "esphome" / "components" / "gaja_device" / "rings.h"

MAIN = r"""
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include "rings.h"
using esphome::gaja_device::SpscRing;
using esphome::gaja_device::RollingRing;
static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}

static void test_spsc() {
  SpscRing ring;
  std::vector<uint8_t> buf(16);
  ring.attach(buf.data(), 16);

  uint8_t src[20], dst[20];
  for (int i = 0; i < 20; i++) src[i] = (uint8_t) (0xA0 + i);

  // Fill completely; ordering must survive the wrap.
  check(ring.push(src, 10), "push 10 fits");
  check(ring.push(src + 10, 6), "push 6 fits");
  check(ring.available() == 16, "ring full");
  check(!ring.push(src + 16, 1), "push when full drops");
  check(ring.dropped() == 1, "drop counted");
  check(ring.space() == 0, "no space");

  memset(dst, 0, sizeof(dst));
  check(ring.peek(dst, 16) == 16, "peek all");
  check(memcmp(dst, src, 16) == 0, "peek data ordered across wrap candidate");
  ring.advance(9);
  check(ring.available() == 7 && ring.space() == 9, "advance frees space");

  // New data wraps around the physical buffer.
  check(ring.push(src + 16, 4), "push after advance");
  memset(dst, 0, sizeof(dst));
  check(ring.peek(dst, 11) == 11, "peek rest");
  check(memcmp(dst, src + 9, 11) == 0, "wrapped data intact");

  // Consumer-side clear: everything consumed at once; producer indices untouched.
  ring.clear();
  check(ring.available() == 0 && ring.space() == 16, "clear empties");
  uint8_t one = 0x42;
  check(ring.push(&one, 1), "push after clear");
  check(ring.available() == 1, "one byte after clear");

  // Partial peek never over-reads.
  SpscRing empty;
  std::vector<uint8_t> ebuf(8);
  empty.attach(ebuf.data(), 8);
  check(empty.peek(dst, 8) == 0, "peek empty returns 0");
}

static void test_rolling() {
  RollingRing ring;
  std::vector<uint8_t> buf(16);
  ring.attach(buf.data(), 16);

  uint8_t src[2048], dst[2048];
  for (int i = 0; i < 2048; i++) src[i] = (uint8_t) (0x30 + i);

  ring.push(src, 10);           // positions 0..9
  check(ring.head_pos() == 10, "head after first push");
  check(ring.tail_pos() == 0, "tail at 0 while under capacity");
  check(ring.peek_abs(0, dst, 10) == 10 && memcmp(dst, src, 10) == 0, "peek first window");

  ring.push(src + 10, 10);      // positions 10..19, window full (capacity 16)
  check(ring.head_pos() == 20, "head counts absolutely");
  check(ring.tail_pos() == 4, "oldest overwritten");
  check(ring.peek_abs(0, dst, 4) == 0, "peek below tail returns 0");
  check(ring.peek_abs(4, dst, 16) == 16, "peek retained window");
  check(memcmp(dst, src + 4, 16) == 0, "retained window data intact");

  // A push larger than the capacity keeps only the newest window.
  ring.push(src + 20, 40);      // would be positions 20..59, but len 40 > capacity 16
  check(ring.head_pos() == 60, "head still counts all bytes");
  check(ring.tail_pos() == 44, "tail is one capacity behind");
  check(ring.peek_abs(44, dst, 16) == 16, "newest window peekable");
  check(memcmp(dst, src + 44, 16) == 0, "newest window is the tail of the push");

  // A snapshot stays readable while the producer keeps running (the burst-send pattern).
  RollingRing live;
  std::vector<uint8_t> lbuf(2048);
  live.attach(lbuf.data(), 2048);
  live.push(src, 1024);
  const size_t snap_head = live.head_pos();
  const size_t snap_tail = live.tail_pos();
  live.push(src, 512);          // producer continues past the snapshot head
  size_t got = live.peek_abs(snap_tail, dst, snap_head - snap_tail);
  check(got == 1024, "snapshot still readable");
  check(memcmp(dst, src, 1024) == 0, "snapshot data intact");
}

int main() {
  test_spsc();
  test_rolling();
  if (failures) { std::printf("%d ring failures\n", failures); return 1; }
  std::printf("rings OK\n");
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


def test_rings() -> None:
    with tempfile.TemporaryDirectory(prefix="gaja-rings-") as tmp:
        main = Path(tmp) / "main.cpp"
        main.write_text(MAIN)
        exe = Path(tmp) / "rings"
        subprocess.run([_cxx(), "-std=c++17", "-I", str(HEADER.parent), str(main), "-o", str(exe)], check=True)
        result = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    assert "rings OK" in result.stdout
    assert "FAIL" not in result.stdout
