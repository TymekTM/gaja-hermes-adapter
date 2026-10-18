#pragma once

#ifdef USE_ESP32

#include "esphome/components/audio/audio.h"
#include "esphome/components/speaker/speaker.h"
#include "esphome/core/component.h"
#include "esphome/core/helpers.h"

#include "../duplex_i2s.h"

#include <freertos/FreeRTOS.h>
#include <freertos/event_groups.h>
#include <freertos/task.h>

namespace esphome::gaja_device {

class GajaDuplexMicrophone;

/**
 * Speaker platform on the gaja_device duplex I2S bus (full duplex with the microphone).
 *
 * Same contract as the stock i2s_audio speaker for its callers: play() auto-starts, finish()
 * stops after draining the buffer, stop() breaks immediately, has_buffered_data() reports the
 * internal queue. Differences by construction:
 *  - The bus (and its clocks) NEVER stop: while no audio is queued the TX task writes silence,
 *    because the RX channel (microphone) on the same controller dies with the clock. "Stopped"
 *    is purely a data-path state; the amp switch around it stays the gaja_device component's job.
 *  - Audio plays at 48 kHz mono 16-bit on the left slot (the ES8311-verified configuration); a
 *    stream info with any other rate logs a warning and plays at the wrong pitch.
 *  - Every chunk written to the DAC is decimated 48k -> 16 kHz (the SAME decim3 filter as the
 *    mic path, so both carry the identical group delay) and pushed into the bus reference ring
 *    for the server-side echo canceller.
 *  - DAC volume/mute go through the linked audio_dac (es8311) via the Speaker base class,
 *    exactly like the stock speaker (volume 0.75 = ES8311 0 dB, the verified full scale).
 */
class GajaDuplexSpeaker : public speaker::Speaker, public Component {
 public:
  void setup() override;
  void loop() override;
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::DATA; }

  size_t play(const uint8_t *data, size_t length) override;
  void start() override;
  void stop() override;
  void finish() override;
  bool has_buffered_data() const override;
  // NOTE: Speaker::set_audio_stream_info is NOT virtual; callers hold the static type, so this
  // only shadows it. The stored stream info (base member) is what the callers read back.
  void set_audio_stream_info(const audio::AudioStreamInfo &info) {
    speaker::Speaker::set_audio_stream_info(info);
    this->warn_stream_rate_(info);
  }

  /// Attach to the microphone platform's bus (yaml `microphone_id`).
  void set_bus_mic(GajaDuplexMicrophone *mic);

 protected:
  static void tx_task_(void *arg);
  void write_chunk_(uint8_t *chunk, size_t len);   // len <= TX_CHUNK_BYTES, zero padded
  void push_ref_(const int16_t *samples48k, size_t count);
  void warn_stream_rate_(const audio::AudioStreamInfo &info);

  GajaDuplexMicrophone *mic_{nullptr};
  DuplexBus *bus_{nullptr};
  SpscRing tx_ring_{};
  EventGroupHandle_t events_{nullptr};
  Decim3 ref_decim_{};
  bool task_running_{false};
  int warned_rate_{0};

  // Scratch buffers (internal RAM, allocated in setup, TX-task only).
  uint8_t *chunk_{nullptr};
  int16_t *ref_in_{nullptr};
  int16_t *ref_out_{nullptr};

  enum EventBits : uint32_t {
    COMMAND_STOP = 1 << 0,           // stop(): break, drop the queue
    COMMAND_STOP_GRACEFUL = 1 << 1,  // finish(): stop after draining the queue
    TASK_DRAINED = 1 << 2,           // tx task: queue empty since a stop was requested
  };
};

}  // namespace esphome::gaja_device

#endif  // USE_ESP32
