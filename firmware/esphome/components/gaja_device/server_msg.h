#pragma once

// Classification of server text messages by their JSON "type", as pure functions so they can be
// unit-tested on the host (tests/test_firmware_server_msg.py).
//
// The WebSocket task has to decide, before the main loop ever parses a message, whether it may be
// dropped when the text queue is under pressure. The type is therefore extracted with a tiny
// allocation-free scanner (no JSON library, bounded work, a few bytes of stack).
//
// Only a short allow-list of purely informational ("log-only") types is droppable. EVERYTHING else
// -- ready, wake_accepted/rejected, listen_end, follow_up, audio_start/end, error, ping,
// record_start/stop, timer, and any type we do not know -- is a control message: the state machine
// may depend on it, so it is never evicted to make room and never discarded early.

#include <cstddef>
#include <cstring>

namespace esphome::gaja_device {

/// Longest "type" value the scanner reports (longer values are "not found", i.e. control).
static constexpr size_t SERVER_MSG_TYPE_MAX = 24;

/// Extract the top-level string member "type" of a JSON object into ``out`` (NUL-terminated).
/// Nested objects/arrays and escaped quotes inside strings are skipped correctly. Returns the type
/// length, or 0 when there is no top-level string "type" (or it does not fit / uses escapes).
static inline size_t server_msg_type(const char *json, size_t len, char *out, size_t out_size) {
  if (out_size > 0) {
    out[0] = '\0';
  }
  if (json == nullptr || out == nullptr || out_size == 0) {
    return 0;
  }
  size_t i = 0;
  auto skip_ws = [&]() {
    while (i < len && (json[i] == ' ' || json[i] == '\t' || json[i] == '\r' || json[i] == '\n')) {
      i++;
    }
  };
  // Scan one string token starting at the opening quote; leaves i after the closing quote.
  // Returns false when the string is unterminated. ``escaped`` reports a backslash inside.
  auto scan_string = [&](size_t *begin, size_t *end, bool *escaped) -> bool {
    i++;  // opening quote
    *begin = i;
    *escaped = false;
    while (i < len) {
      if (json[i] == '\\') {
        *escaped = true;
        i += 2;
        continue;
      }
      if (json[i] == '"') {
        *end = i;
        i++;
        return true;
      }
      i++;
    }
    return false;
  };

  skip_ws();
  if (i >= len || json[i] != '{') {
    return 0;
  }
  i++;
  int depth = 1;
  bool expect_key = true;  // at depth 1, right after '{' or ','
  while (i < len) {
    skip_ws();
    if (i >= len) {
      break;
    }
    const char c = json[i];
    if (c == '"') {
      size_t b = 0, e = 0;
      bool esc = false;
      if (!scan_string(&b, &e, &esc)) {
        return 0;
      }
      if (depth == 1 && expect_key) {
        expect_key = false;
        skip_ws();
        if (i < len && json[i] == ':') {
          i++;
          if (!esc && e - b == 4 && memcmp(json + b, "type", 4) == 0) {
            skip_ws();
            if (i >= len || json[i] != '"') {
              return 0;  // "type" is not a string
            }
            size_t vb = 0, ve = 0;
            bool vesc = false;
            if (!scan_string(&vb, &ve, &vesc) || vesc) {
              return 0;
            }
            const size_t n = ve - vb;
            if (n == 0 || n > SERVER_MSG_TYPE_MAX || n + 1 > out_size) {
              return 0;
            }
            memcpy(out, json + vb, n);
            out[n] = '\0';
            return n;
          }
        }
      }
      continue;
    }
    if (c == '{' || c == '[') {
      depth++;
    } else if (c == '}' || c == ']') {
      depth--;
      if (depth == 0) {
        break;
      }
    } else if (c == ',' && depth == 1) {
      expect_key = true;
    }
    i++;
  }
  return 0;
}

/// True for informational message types the device only logs: they may be discarded when the
/// queue is more than half full and evicted to make room for a control message. Control messages
/// and unknown types are NOT droppable.
static inline bool server_msg_is_droppable(const char *type) {
  if (type == nullptr || type[0] == '\0') {
    return false;
  }
  static const char *const DROPPABLE[] = {"transcript", "utterance", "metrics", "state", "vad",
                                          "handoff",    "tool",      "notice",  "pong"};
  for (const char *candidate : DROPPABLE) {
    if (strcmp(type, candidate) == 0) {
      return true;
    }
  }
  return false;
}

/// Convenience: classify a raw message (scan + allow-list). Unparseable -> not droppable.
static inline bool server_msg_text_is_droppable(const char *json, size_t len) {
  char type[SERVER_MSG_TYPE_MAX + 1];
  if (server_msg_type(json, len, type, sizeof(type)) == 0) {
    return false;
  }
  return server_msg_is_droppable(type);
}

}  // namespace esphome::gaja_device
