"""Host-side check of the firmware's deferred log ring (log_defer.h).

FIFO order, wrap-around, truncation of long lines and eviction of the oldest lines when full.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "esphome" / "components" / "gaja_device" / "log_defer.h"

MAIN = r"""
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "log_defer.h"
using namespace esphome::gaja_device;
static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}
static void push(LogDefer &d, uint8_t lvl, const std::string &s) { d.push(lvl, s.data(), s.size()); }
static std::string pop(LogDefer &d, uint8_t *lvl = nullptr) {
  char out[LogDefer::MAX_LINE + 1];
  uint8_t level = 0;
  size_t len = 0;
  if (!d.pop(level, out, len)) return "<empty>";
  if (lvl) *lvl = level;
  check(std::strlen(out) == len, "nul-terminated at len");
  return std::string(out, len);
}

int main() {
  std::vector<uint8_t> store(600);
  LogDefer d;
  d.attach(store.data(), store.size());
  check(d.attached(), "attached");

  // FIFO with levels.
  push(d, 3, "one");
  push(d, 5, "two");
  uint8_t lvl = 0;
  check(d.lines() == 2, "two lines");
  check(pop(d, &lvl) == "one" && lvl == 3, "fifo one");
  check(pop(d, &lvl) == "two" && lvl == 5, "fifo two");
  check(pop(d) == "<empty>", "empty");

  // Wrap-around: many lines through a small ring keep their content.
  for (int i = 0; i < 200; i++) {
    std::string s = "line-" + std::to_string(i) + std::string(i % 37, 'x');
    push(d, 4, s);
    check(pop(d) == s, "wrap content");
  }

  // Truncation.
  push(d, 1, std::string(1000, 'a'));
  check(pop(d) == std::string(LogDefer::MAX_LINE, 'a'), "truncated to MAX_LINE");

  // Eviction of the oldest when full: 600 B ring, 103 B records -> 5 fit.
  for (int i = 0; i < 8; i++) push(d, 2, std::string(99, 'a') + std::to_string(i));
  check(d.lines() == 5, "five kept");
  check(d.take_dropped() == 3, "three dropped");
  check(d.take_dropped() == 0, "dropped reset");
  check(pop(d).back() == '3', "oldest kept is #3");
  for (int i = 4; i < 8; i++) check(pop(d).back() == (char) ('0' + i), "order after eviction");
  check(d.lines() == 0, "drained");

  if (failures == 0) std::printf("log_defer OK\n");
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


def test_log_defer() -> None:
    with tempfile.TemporaryDirectory(prefix="gaja-log-defer-") as tmp:
        main = Path(tmp) / "main.cpp"
        main.write_text(MAIN)
        exe = Path(tmp) / "log_defer"
        subprocess.run(
            [_cxx(), "-std=c++17", "-fsanitize=address,undefined", "-I", str(HEADER.parent), str(main), "-o", str(exe)],
            check=True,
        )
        result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert "log_defer OK" in result.stdout, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
