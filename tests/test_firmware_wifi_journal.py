"""Host-side check of the firmware's RTC-backed Wi-Fi event journal (wifi_journal.h).

Compiles ``firmware/esphome/components/gaja_device/wifi_journal.h`` (pure C++, no ESP-IDF
includes) with the system C++ compiler and asserts:
- compact trivial RTC storage layout (safe for ``RTC_NOINIT_ATTR`` without clearing on reboot);
- wraparound eviction and overwrite accounting;
- bounded snapshot + drain + ``mark_delivered`` (and failed send keeping entries);
- concurrent producer overwrite during an in-flight snapshot;
- RTC header validation (bad magic, bad checksum, version/capacity change, power-on -> empty);
- boot counter increment and preservation of previous-boot entries across software resets;
- 32-bit sequence counter wraparound.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "esphome" / "components" / "gaja_device" / "wifi_journal.h"

MAIN = r"""
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <type_traits>
#include <vector>
#include "wifi_journal.h"

using namespace esphome::gaja_device;

static_assert(std::is_trivially_default_constructible_v<WifiJournalStorage<64>>,
              "RTC storage must be trivially default-constructible so startup does not zero .rtc_noinit");
static_assert(std::is_standard_layout_v<WifiJournalStorage<64>>,
              "RTC storage must have standard layout");
static_assert(sizeof(WifiEventRecord) == 20, "WifiEventRecord must stay compact (20 B)");
static_assert(sizeof(WifiJournalStorage<64>) == 32 + 64 * 20,
              "WifiJournalStorage<64> must stay 1312 B with no hidden padding");

static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    failures++;
  }
}

static void test_wraparound_and_drain() {
  WifiJournalStorage<4> raw;
  std::memset(&raw, 0xA5, sizeof(raw));  // uninitialized RTC garbage

  WifiJournal<4> j;
  j.attach(&raw);
  check(!j.is_valid(), "garbage RTC memory is invalid");
  j.begin_boot(/*reset_reason=*/1, /*force_reset=*/false);
  check(j.is_valid(), "valid after begin_boot");
  check(j.boot() == 1, "first boot is 1");
  check(j.reset_reason() == 1, "reset reason recorded");
  check(j.pending() == 0, "starts empty");
  check(j.dropped() == 0, "zero dropped initially");

  const uint8_t bssid[6] = {0xD8, 0x32, 0x14, 0x47, 0xE7, 0xE2};
  // Append 7 events into capacity 4 -> oldest 3 overwritten, newest 4 kept.
  for (uint32_t i = 0; i < 7; i++) {
    j.append(1000 + i * 100, WifiEventKind::DISCONNECTED, (uint16_t)(200 + i), bssid, (uint8_t)(6 + i),
             (int8_t)(-50 - (int)i), WIFI_EVENT_FLAG_HAS_BSSID | WIFI_EVENT_FLAG_HAS_CHANNEL | WIFI_EVENT_FLAG_HAS_RSSI);
  }
  check(j.pending() == 4, "4 retained in capacity-4 ring");
  check(j.dropped() == 3, "3 overwrites counted");

  // Bounded snapshot (2 of 4), failed send -> nothing marked delivered.
  WifiEventRecord out[4];
  auto snap1 = j.snapshot(out, 2);
  check(snap1.count == 2, "chunked snapshot returns 2");
  check(snap1.dropped == 3, "snapshot reports 3 dropped");
  check(out[0].reason == 203 && out[0].ms == 1300, "oldest retained is #3");
  check(out[1].reason == 204 && out[1].ms == 1400, "second retained is #4");
  check(std::memcmp(out[0].bssid, bssid, 6) == 0, "bssid preserved");
  check(out[0].rssi == -53 && out[0].channel == 9, "rssi and channel preserved");

  // Failed send: do NOT call mark_delivered -> entries and dropped remain intact.
  WifiEventRecord retry[4];
  auto snap_retry = j.snapshot(retry, 2);
  check(snap_retry.count == 2 && snap_retry.dropped == 3, "failed send preserves entries and dropped");
  check(retry[0].reason == 203 && retry[1].reason == 204, "failed send returns same entries");

  // Successful send of chunk 1 -> advances tail, subtracts reported dropped.
  j.mark_delivered(snap_retry);
  check(j.pending() == 2, "2 entries remain after chunk 1");
  check(j.dropped() == 0, "dropped cleared after reported");

  // Chunk 2 drains the rest.
  auto snap2 = j.snapshot(out, 4);
  check(snap2.count == 2 && snap2.dropped == 0, "chunk 2 has 2 entries and 0 dropped");
  check(out[0].reason == 205 && out[1].reason == 206, "chunk 2 entries are #5 and #6");
  j.mark_delivered(snap2);
  check(j.pending() == 0, "empty after chunk 2");
}

static void test_overwrite_during_inflight_snapshot() {
  WifiJournalStorage<4> raw;
  WifiJournal<4> j;
  j.attach(&raw);
  j.begin_boot(1, true);

  for (uint32_t i = 0; i < 4; i++) {
    j.append(100 + i, WifiEventKind::SCAN_DONE, (uint16_t)i);
  }
  WifiEventRecord out[4];
  auto snap = j.snapshot(out, 2);  // snapshots seqs 0, 1 (ack_until = 2)
  check(snap.count == 2, "inflight snapshot took 2");

  // Producer appends 3 more events while send is in flight:
  // seq 4 overwrites seq 0 (tail->1, dropped=1)
  // seq 5 overwrites seq 1 (tail->2, dropped=2)
  // seq 6 overwrites seq 2 (tail->3, dropped=3)
  for (uint32_t i = 4; i < 7; i++) {
    j.append(100 + i, WifiEventKind::SCAN_DONE, (uint16_t)i);
  }
  // Now send completes: mark_delivered must NOT advance tail past seq 3!
  j.mark_delivered(snap);
  check(j.pending() == 4, "all 4 newer entries preserved when producer lapped ack_until");
  check(j.dropped() == 3, "3 new drops preserved");
  auto snap2 = j.snapshot(out, 4);
  check(snap2.count == 4, "4 entries drained");
  check(out[0].reason == 3 && out[1].reason == 4 && out[2].reason == 5 && out[3].reason == 6,
        "no unsent entry skipped");
  j.mark_delivered(snap2);
  check(j.pending() == 0 && j.dropped() == 0, "drained completely");

  // Partial overlap case: snapshot 0, 1 (ack_until=2); producer appends 1 event overwriting 0 (tail->1).
  j.begin_boot(1, true);
  for (uint32_t i = 0; i < 4; i++) {
    j.append(200 + i, WifiEventKind::CONNECTED, (uint16_t)i);
  }
  auto snap3 = j.snapshot(out, 2);  // ack_until = 2
  j.append(204, WifiEventKind::CONNECTED, 4);  // overwrites 0 -> tail=1, dropped=1
  j.mark_delivered(snap3);  // tail should move from 1 to 2, leaving 2, 3, 4 (3 entries)
  check(j.pending() == 3, "partial overlap leaves exactly 3 unsent entries");
  check(j.dropped() == 1, "new overwrite stays counted");
  auto snap4 = j.snapshot(out, 4);
  check(snap4.count == 3 && out[0].reason == 2 && out[1].reason == 3 && out[2].reason == 4,
        "partial overlap order intact");
}

static void test_rtc_validation_and_boot_counter() {
  WifiJournalStorage<8> raw;
  WifiJournal<8> j;
  j.attach(&raw);

  // Boot 1 (power-on).
  j.begin_boot(1 /*POWERON*/, true);
  j.append(50, WifiEventKind::BOOT, 1);
  j.append(500, WifiEventKind::DISCONNECTED, 2);
  check(j.boot() == 1 && j.pending() == 2, "boot 1 recorded 2 events");

  // Warm reset (e.g. panic=4): same RTC buffer survives -> boot increments, old events remain.
  j.begin_boot(4 /*PANIC*/, false);
  check(j.boot() == 2, "boot incremented to 2 across warm reset");
  check(j.reset_reason() == 4, "new reset reason stored");
  check(j.pending() == 2, "boot 1 undelivered events survived warm reset");
  j.append(10, WifiEventKind::BOOT, 4);
  j.append(800, WifiEventKind::CONNECTED, 0);
  check(j.pending() == 4, "4 events across 2 boots");

  WifiEventRecord out[8];
  auto snap = j.snapshot(out, 8);
  check(snap.count == 4, "drained 4");
  check(out[0].boot == 1 && out[1].boot == 1, "older events carry boot 1");
  check(out[2].boot == 2 && out[3].boot == 2, "newer events carry boot 2");

  // Bad magic -> resets to empty, boot=1.
  raw.magic ^= 1u;
  check(!j.is_valid(), "bad magic invalid");
  j.begin_boot(3, false);
  check(j.is_valid() && j.boot() == 1 && j.pending() == 0, "bad magic resets to empty");

  // Bad checksum (header mutation) -> resets to empty.
  j.append(100, WifiEventKind::GOT_IP, 0);
  check(j.pending() == 1, "1 event before corruption");
  raw.checksum ^= 0x80000000u;
  check(!j.is_valid(), "bad checksum invalid");
  j.begin_boot(3, false);
  check(j.pending() == 0 && j.boot() == 1, "bad checksum resets to empty");

  // Entry payload corruption (checksum covers entries too) -> resets to empty.
  j.append(100, WifiEventKind::GOT_IP, 0);
  raw.entries[0].reason ^= 0x0100u;
  check(!j.is_valid(), "entry bit flip invalidates checksum");
  j.begin_boot(3, false);
  check(j.pending() == 0 && j.boot() == 1, "corrupt entry resets to empty");

  // A bit flip after boot is not sealed under a fresh CRC by the next write: writes are refused and
  // the next boot still sees the corruption.
  j.append(100, WifiEventKind::GOT_IP, 0);
  raw.entries[0].rssi ^= 0x10;
  j.append(200, WifiEventKind::LOST_IP, 0);
  check(!j.is_valid(), "write after a runtime bit flip does not launder the checksum");
  j.begin_boot(3, false);
  check(j.pending() == 0 && j.boot() == 1, "runtime corruption still resets at boot");

  // Version change -> resets to empty.
  j.append(100, WifiEventKind::GOT_IP, 0);
  raw.version += 1;
  check(!j.is_valid(), "version mismatch invalid");
  j.begin_boot(3, false);
  check(j.pending() == 0 && j.boot() == 1, "version change resets to empty");

  // Power-on (force_reset = true) even with valid header -> resets to empty.
  j.append(100, WifiEventKind::GOT_IP, 0);
  check(j.is_valid(), "valid before power-on");
  j.begin_boot(1, /*force_reset=*/true);
  check(j.pending() == 0 && j.boot() == 1, "power-on resets to empty");
}

static void test_sequence_wraparound() {
  WifiJournalStorage<4> raw;
  WifiJournal<4> j;
  j.attach(&raw);
  j.begin_boot(1, true);

  // Move head and tail near 2^32 wrap boundary and recompute checksum via raw/reattach test helper.
  raw.head = 0xFFFFFFFEu;
  raw.tail = 0xFFFFFFFEu;
  raw.checksum = WifiJournal<4>::compute_checksum(raw);
  check(j.is_valid(), "valid near 32-bit wrap");

  for (uint32_t i = 0; i < 6; i++) {
    j.append(300 + i, WifiEventKind::DISCONNECTED, (uint16_t)(10 + i));
  }
  check(j.pending() == 4, "4 pending across 32-bit wrap");
  check(j.dropped() == 2, "2 dropped across 32-bit wrap");
  WifiEventRecord out[4];
  auto snap = j.snapshot(out, 4);
  check(snap.count == 4 && out[0].reason == 12 && out[3].reason == 15, "order intact across 32-bit wrap");
  j.mark_delivered(snap);
  check(j.pending() == 0 && j.dropped() == 0, "drained across 32-bit wrap");
}

int main() {
  test_wraparound_and_drain();
  test_overwrite_during_inflight_snapshot();
  test_rtc_validation_and_boot_counter();
  test_sequence_wraparound();
  if (failures == 0) {
    std::printf("wifi_journal OK\n");
  }
  return failures;
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


def test_wifi_journal() -> None:
    with tempfile.TemporaryDirectory(prefix="gaja-wifi-journal-") as tmp:
        main = Path(tmp) / "main.cpp"
        main.write_text(MAIN)
        exe = Path(tmp) / "wifi_journal"
        subprocess.run(
            [_cxx(), "-std=c++17", "-fsanitize=address,undefined", "-I", str(HEADER.parent), str(main), "-o", str(exe)],
            check=True,
        )
        result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert "wifi_journal OK" in result.stdout, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
