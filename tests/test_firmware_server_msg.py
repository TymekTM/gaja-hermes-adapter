"""Host-side check of the firmware's server-message classification (G-3 / N-8).

Compiles ``firmware/esphome/components/gaja_device/server_msg.h`` (pure C++) and asserts the
rules the WebSocket task applies before the main loop ever parses a message: the top-level
"type" is extracted without a JSON library, only log-only types are droppable, and every control
message (and anything unknown or unparseable) is not.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HEADER = Path(__file__).resolve().parents[1] / "firmware" / "esphome" / "components" / "gaja_device" / "server_msg.h"

MAIN = r"""
#include <cstdio>
#include <cstring>
#include <string>
#include "server_msg.h"
using namespace esphome::gaja_device;
static int failures = 0;
static void check(bool ok, const char *what) {
  if (!ok) { std::printf("FAIL %s\n", what); failures++; }
}
static std::string type_of(const std::string &json) {
  char out[SERVER_MSG_TYPE_MAX + 1];
  const size_t n = server_msg_type(json.data(), json.size(), out, sizeof(out));
  return n == 0 ? std::string() : std::string(out, n);
}
static bool droppable(const std::string &json) { return server_msg_text_is_droppable(json.data(), json.size()); }

int main() {
  // Compact and Python-style (", " / ": ") separators, type anywhere in the object.
  check(type_of("{\"type\":\"ready\",\"sample_rate\":48000}") == "ready", "compact, type first");
  check(type_of("{\"type\": \"listen_end\"}") == "listen_end", "spaces after the colon");
  check(type_of("{ \"a\": 1, \"type\" : \"audio_start\" , \"id\": 2}") == "audio_start", "type in the middle, loose ws");
  check(type_of("{\"sample_rate\":48000,\"type\":\"audio_end\"}") == "audio_end", "type last");
  check(type_of("\n {\"type\":\"error\",\"message\":\"x\"}") == "error", "leading whitespace");

  // Only a TOP-LEVEL type counts: nested objects/arrays and string contents are skipped.
  check(type_of("{\"device\":{\"type\":\"ready\"},\"type\":\"metrics\"}") == "metrics", "nested type ignored");
  check(type_of("{\"device\":{\"type\":\"ready\"}}") == "", "only a nested type: not found");
  check(type_of("{\"list\":[{\"type\":\"error\"}],\"type\":\"state\"}") == "state", "type inside array ignored");
  check(type_of("{\"text\":\"he said \\\"type\\\":\\\"ready\\\"\",\"type\":\"transcript\"}") == "transcript",
        "escaped quotes inside a string value");
  check(type_of("{\"text\":\"{\\\"type\\\":\\\"error\\\"}\",\"type\":\"utterance\"}") == "utterance",
        "JSON-in-a-string does not confuse the scanner");
  check(type_of("{\"x\":\"type\",\"type\":\"vad\"}") == "vad", "a VALUE equal to 'type' is not the key");

  // Malformed / unusual input never matches and never reads out of bounds.
  check(type_of("") == "", "empty");
  check(type_of("not json") == "", "not json");
  check(type_of("{\"type\":") == "", "truncated after the colon");
  check(type_of("{\"type\":\"read") == "", "truncated value");
  check(type_of("{\"type\":42}") == "", "non-string type");
  check(type_of("{\"type\":\"\"}") == "", "empty type");
  check(type_of("{\"type\":\"abcdefghijklmnopqrstuvwxyz0123\"}") == "", "overlong type");
  check(type_of("{\"ty\\u0070e\":\"ready\"}") == "", "escaped key is not 'type'");
  check(type_of("{\"text\":\"unterminated") == "", "unterminated string");
  char tiny[3];
  check(server_msg_type("{\"type\":\"ready\"}", 16, tiny, sizeof(tiny)) == 0, "does not fit the output buffer");
  // Respects the length, not the NUL: the type is cut off by len.
  check(type_of(std::string("{\"type\":\"ready\"}").substr(0, 12)) == "", "len cuts the message");

  // Droppable = log-only allow-list.
  const char *log_only[] = {"transcript", "utterance", "metrics", "state", "vad", "handoff", "tool", "notice", "pong"};
  for (const char *t : log_only) {
    check(server_msg_is_droppable(t), t);
    check(droppable(std::string("{\"type\":\"") + t + "\",\"x\":1}"), "log-only message is droppable");
  }
  // Control messages are never droppable.
  const char *control[] = {"ready", "wake_accepted", "wake_rejected", "listen_end", "follow_up", "audio_start",
                           "audio_end", "error", "ping", "record_start", "record_stop", "timer"};
  for (const char *t : control) {
    check(!server_msg_is_droppable(t), t);
    check(!droppable(std::string("{\"type\":\"") + t + "\"}"), "control message is not droppable");
  }
  // Unknown or unparseable: conservative -- not droppable.
  check(!server_msg_is_droppable("something_new"), "unknown type is not droppable");
  check(!server_msg_is_droppable(""), "empty type");
  check(!server_msg_is_droppable(nullptr), "null type");
  check(!droppable("garbage"), "unparseable message is not droppable");
  check(!droppable("{\"nope\":1}"), "no type is not droppable");
  // A transcript that merely MENTIONS a control type stays a transcript.
  check(droppable("{\"type\":\"transcript\",\"text\":\"{\\\"type\\\":\\\"listen_end\\\"}\"}"), "mention does not matter");
  // ...and a control message cannot be disguised by a droppable-looking string field.
  check(!droppable("{\"note\":\"\\\"type\\\":\\\"transcript\\\"\",\"type\":\"audio_end\"}"), "escaped decoy ignored");

  if (failures) { std::printf("%d server_msg failures\n", failures); return 1; }
  std::printf("server_msg OK\n");
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


def test_server_msg() -> None:
    with tempfile.TemporaryDirectory(prefix="gaja-server-msg-") as tmp:
        main = Path(tmp) / "main.cpp"
        main.write_text(MAIN)
        exe = Path(tmp) / "server_msg"
        subprocess.run(
            [_cxx(), "-std=c++17", "-fsanitize=address,undefined", "-I", str(HEADER.parent), str(main), "-o", str(exe)],
            check=True,
        )
        result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert "server_msg OK" in result.stdout, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
