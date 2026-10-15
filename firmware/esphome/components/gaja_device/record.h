#pragma once

// Wake-word training recordings (device protocol v1.10): request validation and the
// audio-time conversion, as pure functions so they can be unit-tested on the host
// (tests/test_firmware_record.py).
//
// The server asks the IDLE device to record takes ("Hej Gaja", hard negatives, ambient)
// through the device microphone and measures on-device detection per take (record_mark).
// The id charset is deliberately narrow ([a-z0-9_-], 1..64): it travels back in every
// record_* message, so it must be printable, injection-free and cheap to validate.

#include <cstddef>
#include <cstdint>

namespace esphome::gaja_device {

// record_start.max_ms bounds (milliseconds).
static constexpr uint32_t RECORD_MIN_MS = 1000;
static constexpr uint32_t RECORD_MAX_MS = 120000;
// record_start.id: [a-z0-9_-]{1,64}.
static constexpr size_t RECORD_ID_MAX_LEN = 64;
// The uplink stream is PCM16 LE mono 16 kHz: 32 bytes per millisecond (640 B per 20 ms
// frame). t_ms (record_mark) and ms (record_end) are audio actually SENT, derived by
// dividing the sent byte count by this.
static constexpr uint32_t RECORD_BYTES_PER_MS = 32;

/// True when the recording id matches [a-z0-9_-]{1,64}.
static inline bool record_id_valid(const char *id) {
  if (id == nullptr) {
    return false;
  }
  size_t len = 0;
  for (const char *p = id; *p != '\0'; ++p) {
    const char c = *p;
    const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    if (!ok) {
      return false;
    }
    len++;
  }
  return len >= 1 && len <= RECORD_ID_MAX_LEN;
}

/// True when record_start.max_ms is inside the protocol range 1000..120000.
static inline bool record_max_ms_valid(uint32_t max_ms) {
  return max_ms >= RECORD_MIN_MS && max_ms <= RECORD_MAX_MS;
}

/// Sent bytes -> milliseconds of 16 kHz PCM16 mono audio (truncates sub-ms remainders).
static inline uint32_t record_bytes_to_ms(uint32_t bytes) { return bytes / RECORD_BYTES_PER_MS; }

}  // namespace esphome::gaja_device
