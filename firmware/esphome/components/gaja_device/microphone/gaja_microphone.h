#pragma once

#ifdef USE_ESP32

#include "esphome/components/microphone/microphone.h"
#include "esphome/core/component.h"

#include "../duplex_i2s.h"

#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

#include <atomic>

namespace esphome::gaja_device {

/**
 * Microphone platform on the gaja_device duplex I2S bus (full duplex, 48 kHz shared clock).
 *
 * The bus is started once in setup() and never torn down: the RX task reads 20 ms chunks,
 * decimates them 48k -> 16 kHz (decim3.h) and dispatches them to the data callbacks (the same
 * 16 kHz PCM16 mono the old half-duplex i2s_audio microphone produced, so micro_wake_word,
 * sound_level and the gaja_device stream behave identically). start()/stop() only gate the
 * dispatch through the same counting-listener semaphore semantics as the stock microphone
 * (one take per source start, one give per source stop; the state is RUNNING while any
 * listener is active), so MicrophoneSource and the gaja_device mic power machinery work
 * unchanged. In full-duplex mode the caller never stops the mic anyway: the bus is shared
 * with the speaker and must not be released.
 *
 * The playback reference for the server-side echo canceller is produced on the speaker side
 * of the same bus (DuplexBus::ref_ring, fed by the TX task) -- see gaja_speaker.h.
 */
class GajaDuplexMicrophone : public microphone::Microphone, public Component {
 public:
  void setup() override;
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::DATA; }

  void start() override;
  void stop() override;

  void set_pins(int lrclk, int bclk, int mclk, int din) {
    this->bus_.set_pins(lrclk, bclk, mclk, din, -1);
  }
  /// The speaker platform attaches to the SAME bus object (speaker yaml `microphone_id`).
  DuplexBus &bus() { return this->bus_; }

  /// Absolute positions into the reference ring (gaja_device uplink, same snapshot pattern as
  /// the pre-roll ring: read [from, head) while the TX task keeps producing).
  ring_pos_t ref_head_pos() { return this->bus_.ref_ring().head_pos(); }
  ring_pos_t ref_tail_pos() { return this->bus_.ref_ring().tail_pos(); }
  size_t ref_peek_abs(ring_pos_t abs, uint8_t *dst, size_t len) {
    return this->bus_.ref_ring().peek_abs(abs, dst, len);
  }

 protected:
  static void rx_task_(void *arg);
  bool start_bus_();

  DuplexBus bus_{};
  SemaphoreHandle_t listeners_{nullptr};
  Decim3 decim_{};
  std::atomic<bool> task_running_{false};
};

}  // namespace esphome::gaja_device

#endif  // USE_ESP32
