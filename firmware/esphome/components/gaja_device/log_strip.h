#pragma once

// Pure C++ helper for stripping ANSI colour escapes and control bytes from log lines
// before forwarding to the server (device protocol v1.9+).
// Unit-tested on the host: tests/test_firmware_log_strip.py.

#include <cstddef>
#include <cstdint>
#include <string>

namespace esphome::gaja_device {

/// Strip ANSI colour escape sequences (ESC [ ... m) and any other byte < 0x20
/// from a log line without extra allocations beyond the returned std::string.
inline std::string strip_log_line(const char *line, size_t len) {
  if (line == nullptr || len == 0) {
    return "";
  }
  std::string out;
  out.reserve(len);
  for (size_t i = 0; i < len;) {
    const uint8_t b = (uint8_t) line[i];
    if (b == 0x1B) {
      // Check for ESC [ [0-9;]* m sequence (ANSI SGR colour codes)
      if (i + 1 < len && line[i + 1] == '[') {
        size_t j = i + 2;
        while (j < len && (j - i) <= 16 && ((line[j] >= '0' && line[j] <= '9') || line[j] == ';')) {
          j++;
        }
        if (j < len && line[j] == 'm') {
          i = j + 1;
          continue;
        }
      }
      // Isolated ESC or non-colour sequence: skip ESC (since 0x1B < 0x20)
      i++;
      continue;
    }
    if (b < 0x20) {
      // Strip any byte < 0x20 (CR, LF, TAB, NUL, control bytes)
      i++;
      continue;
    }
    out.push_back(line[i]);
    i++;
  }
  return out;
}

}  // namespace esphome::gaja_device
