#pragma once

namespace esphome::gaja_device {

// Linear map of a measured speech level (dBFS) to a reply volume.
//
// `db` is a high percentile of 20 ms frame RMS values during the streamed utterance, so
// pauses between words do not drag it down. The mapping is linear from [quiet_db, loud_db]
// to [vol_min, vol_max] and clamped at both ends:
//   db <= quiet_db  -> vol_min   (quiet speech -> quiet answer)
//   db >= loud_db   -> vol_max   (loud speech -> full answer)
// Volume units are the media_player/speaker scale of the configured firmware, where 1.0
// means "loudest verified clean level" (on the Atom VoiceS3R: ES8311 DAC at 0 dB).
//
// Pure function on purpose: it is unit-tested on the host (tests/test_firmware_volume_map.py).
inline float map_db_to_volume(float db, float quiet_db, float loud_db, float vol_min, float vol_max) {
  if (loud_db <= quiet_db) {
    // Degenerate configuration: treat everything as loud rather than silently whispering.
    return vol_max;
  }
  float t = (db - quiet_db) / (loud_db - quiet_db);
  if (t < 0.0f) {
    t = 0.0f;
  } else if (t > 1.0f) {
    t = 1.0f;
  }
  return vol_min + t * (vol_max - vol_min);
}

}  // namespace esphome::gaja_device
