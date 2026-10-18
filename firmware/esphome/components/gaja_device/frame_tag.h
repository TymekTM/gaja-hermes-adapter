#pragma once

// Frame tags of the device protocol v1.11 (caps.ref): every binary frame the device sends
// carries a 1-byte tag before the PCM16 16 kHz payload. The server keeps matching constants in
// gaja/devices/aec.py; tests/test_device_aec.py asserts that both sides agree.
//
//   0x00: microphone audio (the old untagged frames are exactly these payloads)
//   0x01: playback reference -- a 16 kHz decimated copy (same decim3 filter as the mic path) of
//         the samples the device wrote to its speaker over the same 20 ms window

#ifdef USE_ESP32

#include <cstdint>

namespace esphome::gaja_device {

static constexpr uint8_t FRAME_TAG_MIC = 0x00;
static constexpr uint8_t FRAME_TAG_REF = 0x01;

}  // namespace esphome::gaja_device

#endif  // USE_ESP32
