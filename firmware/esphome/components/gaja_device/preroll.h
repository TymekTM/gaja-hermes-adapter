#pragma once

// Where the pre-roll burst starts in the rolling ring (pure function, host-testable:
// tests/test_firmware_preroll.py).
//
// The start cursor is the NEWEST of three candidates, expressed as "distance behind head":
//   1. head - preroll_bytes:  the configured pre-roll cap. The server verifies only the FIRST
//      verify_ms (2500) of what we send; sending the whole retained window (~preroll+slack, e.g.
//      4 s) would push the wake phrase out of the verification window and every wake would be
//      rejected (docs/DEVICE.md: "the pre-roll ... up to preroll_ms").
//   2. tail + margin:  one mic chunk of slack above the oldest retained byte, so the cursor is
//      not invalidated (and the first frame not half-overwritten) while the burst is in flight.
//   3. valid_from:  audio before the last mic (re)start belongs to a previous turn (it may
//      contain our own playback) and must never be sent as pre-roll.

#include "rings.h"

#include <algorithm>
#include <cstdint>

namespace esphome::gaja_device {

/// Compute the pre-roll start cursor. All positions are absolute ring positions (modular).
/// Returns the number of bytes of pre-roll in *preroll_out when non-null.
static inline ring_pos_t preroll_start_pos(ring_pos_t head, ring_pos_t tail, ring_pos_t valid_from,
                                           uint32_t preroll_bytes, uint32_t margin_bytes,
                                           uint32_t *preroll_out = nullptr) {
  uint32_t behind = preroll_bytes;  // candidate 1: the configured cap

  // Candidate 2: stay one margin ahead of the tail (0 when the window is younger than that).
  const int32_t tail_behind = ring_pos_diff(head, tail);  // >= 0: tail is behind head
  if (tail_behind <= 0) {
    behind = 0;  // tail at/ahead of head (fresh ring): nothing older exists
  } else if ((uint32_t) tail_behind <= margin_bytes) {
    behind = 0;
  } else {
    behind = std::min(behind, (uint32_t) tail_behind - margin_bytes);
  }

  // Candidate 3: never send audio from before the last mic (re)start. valid_from is always
  // captured as head_pos() and head only grows, so valid_from "ahead of head" can only be the
  // 32-bit wrap (an ancient restart): then the bound is meaningless and is ignored.
  const int32_t vf_behind = ring_pos_diff(head, valid_from);
  if (vf_behind == 0) {
    behind = 0;  // the mic restarted right now: the pre-roll starts at the head
  } else if (vf_behind > 0) {
    behind = std::min(behind, (uint32_t) vf_behind);
  }

  if (preroll_out != nullptr) {
    *preroll_out = behind;
  }
  return (ring_pos_t) (head - behind);
}

/// Where the uplink send cursor resumes after the producer lapped it (the cursor fell behind the
/// ring tail, so the bytes in between are gone). Resuming exactly AT the tail would put the very
/// next read on the bytes the mic task is about to overwrite (torn first frame, and lapped again
/// one chunk later): resume ``margin_bytes`` above the tail instead, but never past the head.
static inline ring_pos_t lap_recovery_pos(ring_pos_t head, ring_pos_t tail, uint32_t margin_bytes) {
  const int32_t window = ring_pos_diff(head, tail);  // bytes retained
  if (window <= 0 || (uint32_t) window <= margin_bytes) {
    return head;  // less than a margin retained: start fresh at the head
  }
  return (ring_pos_t) (tail + margin_bytes);
}

}  // namespace esphome::gaja_device
