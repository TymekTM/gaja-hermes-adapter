"""Host-side check of the firmware's wake-word training recording helpers (protocol v1.10).

Compiles ``firmware/esphome/components/gaja_device/record.h`` (pure C++) and asserts:

- the recording id validation ([a-z0-9_-]{1,64}: charset, empty, null, 64 vs 65 chars),
- the record_start.max_ms bounds (1000..120000 inclusive on both ends),
- the sent-bytes -> audio-ms conversion used for record_mark.t_ms and record_end.ms
  (only audio really sent counts; a 20 ms uplink frame is exactly 640 B).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "esphome" / "components" / "gaja_device" / "record.h"

MAIN = r"""
#include <cstdio>
#include <cstring>
#include "record.h"
using esphome::gaja_device::RECORD_BYTES_PER_MS;
using esphome::gaja_device::record_bytes_to_ms;
using esphome::gaja_device::record_id_valid;
using esphome::gaja_device::record_max_ms_valid;
static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}

int main() {
  // id charset: [a-z0-9_-]
  check(record_id_valid("take-01"), "plain lowercase id");
  check(record_id_valid("a"), "single character");
  check(record_id_valid("hej_gaja_2026-10_hard_neg"), "digits, underscores, dashes");
  check(record_id_valid("---"), "only dashes");
  check(!record_id_valid(""), "empty id is invalid");
  check(!record_id_valid(nullptr), "null id is invalid");
  check(!record_id_valid("Take"), "uppercase is rejected");
  check(!record_id_valid("take.1"), "dot is rejected");
  check(!record_id_valid("take/1"), "slash is rejected");
  check(!record_id_valid("take 1"), "space is rejected");
  check(!record_id_valid("take\n1"), "newline is rejected");
  check(!record_id_valid("hej gaja"), "inner space is rejected");
  // id length: 64 ok, 65 rejected
  char id64[65];
  std::memset(id64, 'a', 64);
  id64[64] = '\0';
  check(record_id_valid(id64), "exactly 64 chars is valid");
  char id65[66];
  std::memset(id65, 'a', 65);
  id65[65] = '\0';
  check(!record_id_valid(id65), "65 chars is rejected");

  // max_ms bounds (inclusive)
  check(!record_max_ms_valid(0), "max_ms 0 is invalid");
  check(!record_max_ms_valid(999), "max_ms 999 is invalid");
  check(record_max_ms_valid(1000), "max_ms 1000 is valid");
  check(record_max_ms_valid(60000), "max_ms 60000 is valid");
  check(record_max_ms_valid(120000), "max_ms 120000 is valid");
  check(!record_max_ms_valid(120001), "max_ms 120001 is invalid");
  check(!record_max_ms_valid(4294967295u), "max_ms UINT32_MAX is invalid");

  // sent bytes -> audio ms: one 20 ms uplink frame is exactly 640 B at 16 kHz PCM16 mono
  check(RECORD_BYTES_PER_MS * 20 == 640, "a 20 ms frame is 640 B");
  check(record_bytes_to_ms(0) == 0, "no bytes -> 0 ms");
  check(record_bytes_to_ms(640) == 20, "one frame -> 20 ms");
  check(record_bytes_to_ms(32u * 6543) == 6543, "6543 ms of audio");
  check(record_bytes_to_ms(31) == 0, "sub-ms remainder truncates toward zero");
  check(record_bytes_to_ms(32u * 60000) == 60000, "a max-length take converts exactly");

  if (failures) return 1;
  std::printf("record OK\n");
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


def test_record_helpers() -> None:
    with tempfile.TemporaryDirectory(prefix="gaja-record-") as tmp:
        main = Path(tmp) / "main.cpp"
        main.write_text(MAIN)
        exe = Path(tmp) / "record"
        subprocess.run([_cxx(), "-std=c++17", "-Wall", "-Werror", "-I", str(HEADER.parent), str(main), "-o", str(exe)],
                       check=True)
        result = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
        assert "record OK" in result.stdout
        assert "FAIL" not in result.stdout
