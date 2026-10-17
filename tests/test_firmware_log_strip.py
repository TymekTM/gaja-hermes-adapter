"""Host-side check of the firmware's ANSI colour and control byte stripping (log_strip.h).

Compiles ``firmware/esphome/components/gaja_device/log_strip.h`` (pure C++) and asserts:
- ANSI colour escapes (ESC [ ... m) are stripped,
- bytes < 0x20 (CR, LF, TAB, NUL, control bytes) are stripped,
- standard printable ASCII and UTF-8 multibyte characters are preserved,
- empty inputs, null pointers, and edge cases are handled safely.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "esphome" / "components" / "gaja_device" / "log_strip.h"

MAIN = r"""
#include <cstdio>
#include <cstring>
#include <string>
#include "log_strip.h"
using esphome::gaja_device::strip_log_line;

static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}

int main() {
  // Empty or null
  check(strip_log_line(nullptr, 0) == "", "null pointer returns empty");
  check(strip_log_line("", 0) == "", "empty string returns empty");

  // Plain printable ASCII
  {
    std::string s = "[I][gaja_device:12]: link up";
    check(strip_log_line(s.data(), s.size()) == s, "plain string unchanged");
  }

  // ANSI colour escape sequences
  {
    std::string in = "\x1b[0;32m[I][gaja_device:12]: link up\x1b[0m";
    check(strip_log_line(in.data(), in.size()) == "[I][gaja_device:12]: link up", "strip green ANSI");
  }
  {
    std::string in = "\x1b[33mwifi: disconnected\x1b[0m";
    check(strip_log_line(in.data(), in.size()) == "wifi: disconnected", "strip yellow ANSI");
  }
  {
    std::string in = "\x1b[1;31m[E][gaja_device:1]: boom\x1b[0m";
    check(strip_log_line(in.data(), in.size()) == "[E][gaja_device:1]: boom", "strip bold red ANSI");
  }
  {
    std::string in = "\x1b[mempty code\x1b[0m";
    check(strip_log_line(in.data(), in.size()) == "empty code", "strip bare ESC[m");
  }

  // Bytes < 0x20 (control characters)
  {
    std::string in = "\x1b[0;32m[I][gaja_device:12]: link up\x1b[0m\r\n";
    check(strip_log_line(in.data(), in.size()) == "[I][gaja_device:12]: link up", "strip CRLF");
  }
  {
    const char in[] = {'a', 0x00, 'b', 0x09, 'c', 0x1f};
    check(strip_log_line(in, sizeof(in)) == "abc", "strip NUL, TAB, unit separator");
  }

  // UTF-8 preservation (bytes >= 0x80)
  {
    std::string in = "\x1b[32mzażółć gęślą jaźń\x1b[0m";
    check(strip_log_line(in.data(), in.size()) == "zażółć gęślą jaźń", "preserve Polish UTF-8");
  }

  // Malformed or unterminated ESC sequences
  {
    std::string in = "\x1b[31unterminated";
    check(strip_log_line(in.data(), in.size()) == "[31unterminated", "drop ESC only on unterminated");
  }
  {
    std::string in = "\x1b";
    check(strip_log_line(in.data(), in.size()) == "", "isolated ESC is stripped");
  }

  if (failures == 0) std::printf("log_strip OK\n");
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


def test_log_strip() -> None:
    with tempfile.TemporaryDirectory(prefix="gaja-log-strip-") as tmp:
        main = Path(tmp) / "main.cpp"
        main.write_text(MAIN)
        exe = Path(tmp) / "log_strip"
        subprocess.run(
            [_cxx(), "-std=c++17", "-fsanitize=address,undefined", "-I", str(HEADER.parent), str(main), "-o", str(exe)],
            check=True,
        )
        res = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
        assert "log_strip OK" in res.stdout
