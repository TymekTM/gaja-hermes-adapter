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

// Absolute-position pattern used by the wrap tests: the byte at absolute position p is
// p & 0x3F, and peeked buffers are verified against it.
static void fill_pattern(uint8_t *p, uint32_t abs, size_t len) {
  for (size_t i = 0; i < len; i++) p[i] = (uint8_t) ((abs + (uint32_t) i) & 0x3Fu);
}
static bool pattern_matches(const uint8_t *p, uint32_t abs, size_t len) {
  for (size_t i = 0; i < len; i++) {
    if (p[i] != (uint8_t) ((abs + (uint32_t) i) & 0x3Fu)) return false;
  }
  return true;
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
  check(!ring.push(src + 16, 1), "push when full is refused");
  check(!ring.push(src + 16, 1), "a retried push is refused again");
  check(ring.dropped() == 0, "push itself never counts: the caller knows what is finally lost");
  ring.add_dropped(7);
  ring.add_dropped(3);
  check(ring.dropped() == 10, "add_dropped accumulates real bytes");
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
  const uint32_t snap_head = live.head_pos();
  const uint32_t snap_tail = live.tail_pos();
  live.push(src, 512);          // producer continues past the snapshot head
  size_t got = live.peek_abs(snap_tail, dst, snap_head - snap_tail);
  check(got == 1024, "snapshot still readable");
  check(memcmp(dst, src, 1024) == 0, "snapshot data intact");
}

// Positions are uint32_t counters (the ESP32-S3's size_t): after ~37 h of mic uptime they wrap.
// Push 2^32 bytes (+ a bit more, so the wrap is crossed twice-worth) through small rings and
// verify every property survives: byte-per-position data, tail tracking, snapshot peeks and the
// "lapped" detection. (Reproduces review M-1 on the host by driving the real wrap.)
static void test_rolling_wrap() {
  RollingRing ring;
  std::vector<uint8_t> buf(64);
  ring.attach(buf.data(), 64);

  // value at absolute position p is p & 0x3F (== p % capacity), filled through push()
  auto pattern = [](uint8_t *p, uint32_t abs, size_t len) { fill_pattern(p, abs, len); };

  uint8_t chunk[997];
  // Stop 64 bytes below the wrap, then cross it with explicit pushes.
  const uint32_t TOTAL = 0xFFFFFFC0u;
  uint32_t abs = 0;
  while (abs < TOTAL) {
    const size_t len = (TOTAL - abs < 997) ? (size_t) (TOTAL - abs) : 997;
    pattern(chunk, abs, len);
    ring.push(chunk, len);
    abs += (uint32_t) len;
  }
  check(ring.head_pos() == TOTAL, "head is the absolute total (mod 2^32)");

  // Data and tail right before the wrap point.
  const uint32_t head0 = ring.head_pos();
  check(ring.tail_pos() == head0 - 64, "tail is one capacity behind (modular)");
  check(ring.peek_abs(head0 - 64, chunk, 64) == 64, "peek whole window pre-wrap");
  check(pattern_matches(chunk, head0 - 64, 64), "byte-per-position data pre-wrap");

  // Cross the wrap: 100 bytes in two odd fragments.
  pattern(chunk, head0, 51);
  ring.push(chunk, 51);
  pattern(chunk, head0 + 51, 49);
  ring.push(chunk, 49);
  const uint32_t head1 = ring.head_pos();
  check(head1 == head0 + 100, "head advanced across the wrap (modular)");
  check(head1 < head0, "head wrapped past 2^32");
  check(ring.tail_pos() == head1 - 64, "tail across the wrap");
  check(ring.peek_abs(head1 - 64, chunk, 64) == 64, "peek whole window post-wrap");
  check(pattern_matches(chunk, head1 - 64, 64), "byte-per-position data post-wrap");
  // A 16-byte window that straddles the wrap itself (2^32 = 0x100000000: the retained window
  // [0xFFFFFFEC, 0x10000002C) covers it; -12 mod 2^32 = 0xFFFFFFF4 starts such a peek).
  const uint32_t straddle = (uint32_t) (0u - 12u);
  check(ring.peek_abs(straddle, chunk, 16) == 16, "peek straddling the wrap");
  check(pattern_matches(chunk, straddle, 16), "data straddling the wrap");

  // Snapshots pinned pre-wrap stay readable until lapped, and report 0 afterwards.
  const uint32_t snap = head1 - 32;
  check(ring.peek_abs(snap, chunk, 32) == 32, "snapshot readable post-wrap");
  ring.push(chunk, 4096);  // laps the snapshot (capacity 64)
  check(ring.peek_abs(snap, chunk, 32) == 0, "lapped snapshot returns 0 post-wrap");
  check(ring.peek_abs(head1 + 4096 - 64, chunk, 64) == 64, "newest window readable post-lap");

  // The consumer side (SpscRing) wraps identically: produce/consume across 2^32 with checks.
  SpscRing spsc;
  std::vector<uint8_t> sbuf(16);
  spsc.attach(sbuf.data(), 16);
  uint8_t out[16];
  uint32_t sabs = 0;                 // absolute position of the next byte to produce
  uint64_t remaining = (1ull << 32) + 4096;  // cross the wrap once, land 4 kB past it
  while (remaining > 0) {
    const size_t len = remaining > 8 ? 8 : (size_t) remaining;  // divides the 16 B capacity
    fill_pattern(chunk, sabs, len);
    // The producer is dropped-from only if the consumer lags; the consumer keeps up here.
    if (!spsc.push(chunk, len)) {
      const size_t n = spsc.peek(out, 16);
      check(pattern_matches(out, sabs - (uint32_t) spsc.available(), n), "spsc data before wrap");
      spsc.advance(n);
      check(spsc.push(chunk, len), "push succeeds after the consumer catches up");
    }
    sabs += (uint32_t) len;
    remaining -= len;
    if (spsc.available() >= 16) {
      const size_t n = spsc.peek(out, 16);
      check(pattern_matches(out, sabs - (uint32_t) spsc.available(), n), "spsc data across the wrap");
      spsc.advance(n);
    }
  }
  check(sabs == 4096, "spsc produced exactly 2^32 + 4096 bytes");
  check(spsc.dropped() == 0, "consumer keeping up: nothing dropped");
}

// N-3: the mic-restart "skip" window test must stay bounded on both sides, otherwise it flips to
// true for every frame once the ring position is >= 2^31 bytes (~18.6 h) past the restart.
static void test_within_before() {
  using esphome::gaja_device::ring_pos_within_before;
  const uint32_t SPAN = 6400;
  const uint32_t until = 1'000'000;
  check(ring_pos_within_before(until - 1, until, SPAN), "1 byte before: inside");
  check(ring_pos_within_before(until - SPAN, until, SPAN), "exactly span before: inside");
  check(!ring_pos_within_before(until - SPAN - 1, until, SPAN), "beyond span: outside");
  check(!ring_pos_within_before(until, until, SPAN), "at until: outside (first counted frame)");
  check(!ring_pos_within_before(until + 640, until, SPAN), "after until: outside");
  // Far past: 2^31 + x bytes after the restart (18.6 h of idle): the old signed ring_pos_before
  // test said "before" for every frame; the bounded check must say outside.
  const uint32_t far_after = until + 0x80000000u + 12345u;
  check(esphome::gaja_device::ring_pos_before(far_after, until), "premise: signed diff is negative");
  check(!ring_pos_within_before(far_after, until, SPAN), "18.6 h later: not in the skip window");
  check(!ring_pos_within_before(until - SPAN - 1000, until, SPAN), "a bit further back: outside");
  check(!ring_pos_within_before(until + 0x7FFFFFF0u, until, SPAN), "almost half the circle ahead: outside");
  // Across the 32-bit wrap.
  check(ring_pos_within_before(0xFFFFFF00u, 0x100u, SPAN), "window straddling the wrap");
  check(!ring_pos_within_before(0x200u, 0x100u, SPAN), "past the wrap: outside");
}

int main() {
  test_within_before();
  test_spsc();
  test_rolling();
  test_rolling_wrap();
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
