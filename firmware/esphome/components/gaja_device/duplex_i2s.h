#pragma once

// One ESP-IDF full-duplex I2S controller for the Gaja Atom (gaja_device component).
//
// Half-duplex history: ESPHome's i2s_audio mic and speaker each grab an exclusive mutex on the
// shared I2SAudioComponent and allocate only one direction (RX or TX) of the port, so the mic
// must be stopped before every playback. This driver owns the port instead and allocates BOTH
// directions in one i2s_new_channel call, which per ESP-IDF is full-duplex mode:
//   "If both tx_handle and rx_handle are not NULL, it means this I2S controller will work at
//    full-duplex mode, the RX and TX channels will be allocated on a same I2S port in this
//    case." (esp-idf components/esp_driver_i2s/include/driver/i2s_common.h, i2s_new_channel note)
// Both channels are initialised in std mode with the SAME clock configuration, which per IDF
// "can constitude as full-duplex mode automatically"
//   (esp-idf components/esp_driver_i2s/include/driver/i2s_std.h, i2s_channel_init_std_mode note).
//
// Hardware facts kept from the verified half-duplex config (firmware yaml header): ES8311 at
// 48000 Hz (at 16000 the ADC returns white noise), mic ADC data on the LEFT slot, speaker on
// the LEFT slot, 16-bit, MCLK on (multiple 256). A shared bit-clock means ONE sample rate for
// both directions, so everything runs at 48 kHz and the mic path is decimated 3:1 to 16 kHz
// (decim3.h). The TX task pushes a decimated copy of everything played into the reference ring
// for the server-side echo canceller (protocol v1.11 caps.ref; docs/FULL_DUPLEX.md).

#ifdef USE_ESP32

#include "decim3.h"
#include "rings.h"

#include <driver/i2s_std.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <cstdint>

namespace esphome::gaja_device {

class DuplexBus {
 public:
  static constexpr uint32_t RATE = 48000;              // the ES8311-verified rate
  static constexpr uint32_t MIC_RATE = 16000;          // after decimation
  // 20 ms chunks: RX reads this much per iteration, TX writes exactly this much per iteration
  // (silence when idle), so both directions are paced by the I2S DMA itself.
  static constexpr size_t TX_CHUNK_BYTES = RATE / 50 * 2;  // 1920 B
  static constexpr size_t RX_CHUNK_BYTES = TX_CHUNK_BYTES;

  void set_pins(int lrclk, int bclk, int mclk, int din, int dout) {
    this->lrclk_ = lrclk;
    this->bclk_ = bclk;
    this->mclk_ = mclk;
    this->din_ = din;
    this->dout_ = dout;
  }

  /// Allocate both channels on one controller, init std mode (48 kHz / 16 bit / mono left /
  /// Philips) with identical clock configs, enable both. The TX channel must never be disabled
  /// afterwards: it clocks the shared bus the RX channel runs on.
  bool start();

  /// Reference ring (PSRAM, RollingRing): the TX task pushes a 16 kHz decimated copy of every
  /// chunk that was actually written to the DAC. The gaja_device uplink reads it with absolute
  /// positions, exactly like the pre-roll ring, so a wake during playback snapshots the matching
  /// reference window for the server-side echo canceller (protocol v1.11 caps.ref).
  void attach_ref_ring(uint8_t *buffer, size_t capacity) { this->ref_ring_.attach(buffer, capacity); }
  RollingRing &ref_ring() { return this->ref_ring_; }

  bool started() const { return this->started_; }
  i2s_chan_handle_t tx_handle() const { return this->tx_; }
  i2s_chan_handle_t rx_handle() const { return this->rx_; }

 protected:  i2s_chan_handle_t tx_{nullptr};
  i2s_chan_handle_t rx_{nullptr};
  int lrclk_{-1};
  int bclk_{-1};
  int mclk_{-1};
  int din_{-1};
  int dout_{-1};
  bool started_{false};
  RollingRing ref_ring_{};
};

}  // namespace esphome::gaja_device

#endif  // USE_ESP32
