#include "gaja_speaker.h"

#ifdef USE_ESP32

#include "../microphone/gaja_microphone.h"
#include "esphome/core/log.h"

#include <esp_heap_caps.h>

#include <cstring>

namespace esphome::gaja_device {

static const char *const TAG = "gaja_speaker";

static constexpr UBaseType_t TASK_PRIORITY = 10;
static constexpr uint32_t TASK_STACK = 3072;
// 48000 B = 500 ms of PCM16 mono 48 kHz: rides main-loop blocks the same way the verified
// buffer_duration: 1000ms of the stock speaker did, without starving at inter-sentence gaps.
static constexpr size_t TX_RING_BYTES = 48000;

void GajaDuplexSpeaker::set_bus_mic(GajaDuplexMicrophone *mic) {
  this->mic_ = mic;
  this->bus_ = mic != nullptr ? &mic->bus() : nullptr;
}

void GajaDuplexSpeaker::setup() {
  if (this->bus_ == nullptr) {
    ESP_LOGE(TAG, "No microphone_id: the speaker cannot share the duplex bus");
    this->mark_failed();
    return;
  }
  this->chunk_ = (uint8_t *) heap_caps_malloc(DuplexBus::TX_CHUNK_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  this->ref_in_ = (int16_t *) heap_caps_malloc(DuplexBus::TX_CHUNK_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  this->ref_out_ = (int16_t *) heap_caps_malloc(DuplexBus::TX_CHUNK_BYTES / 3, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  uint8_t *ring = (uint8_t *) heap_caps_malloc(TX_RING_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (ring == nullptr) {
    ring = (uint8_t *) heap_caps_malloc(TX_RING_BYTES, MALLOC_CAP_8BIT);
  }
  this->events_ = xEventGroupCreate();
  if (this->chunk_ == nullptr || this->ref_in_ == nullptr || this->ref_out_ == nullptr || ring == nullptr ||
      this->events_ == nullptr) {
    ESP_LOGE(TAG, "Could not allocate the TX buffers");
    this->mark_failed();
    return;
  }
  this->tx_ring_.attach(ring, TX_RING_BYTES);
  this->audio_stream_info_ = audio::AudioStreamInfo(16, 1, DuplexBus::RATE);
}

void GajaDuplexSpeaker::loop() {
  if (this->is_failed()) {
    return;
  }
  // The TX task is spawned lazily: setup order between same-priority components follows
  // registration order, but the bus is guaranteed started after every setup() has run.
  if (!this->task_running_) {
    if (this->bus_ == nullptr || !this->bus_->started()) {
      return;  // retry on the next pass
    }
    if (xTaskCreate(GajaDuplexSpeaker::tx_task_, "gaja_spk", TASK_STACK, this, TASK_PRIORITY,
                    nullptr) != pdPASS) {
      ESP_LOGE(TAG, "Could not create the TX task");
      this->mark_failed();
      return;
    }
    this->task_running_ = true;
  }
  const uint32_t bits = xEventGroupGetBits(this->events_);
  if ((bits & TASK_DRAINED) && this->state_ == speaker::STATE_STOPPING) {
    // The queue is empty (dropped or drained): the data path is stopped. New audio arriving
    // between the ack and this line re-armed play() (state STARTING) and wins.
    this->state_ = speaker::STATE_STOPPED;
    xEventGroupClearBits(this->events_, TASK_DRAINED | COMMAND_STOP | COMMAND_STOP_GRACEFUL);
  }
}

size_t GajaDuplexSpeaker::play(const uint8_t *data, size_t length) {
  if (this->is_failed()) {
    return 0;
  }
  if (this->state_ != speaker::STATE_RUNNING && this->state_ != speaker::STATE_STARTING) {
    // Auto-start like the stock speaker: callers only push and check the return value.
    this->start();
  }
  if (this->state_ != speaker::STATE_RUNNING) {
    return 0;  // still stopping: callers retry (gaja_device) or wait for is_running (pipeline)
  }
  // Whole samples only; the SPSC ring is all-or-nothing, so clip the request to the free space.
  size_t len = length & ~(size_t) 1;
  const size_t space = this->tx_ring_.space() & ~(size_t) 1;
  if (len > space) {
    len = space;
  }
  if (len == 0 || !this->tx_ring_.push(data, len)) {
    return 0;
  }
  return len;
}

void GajaDuplexSpeaker::start() {
  if (this->is_failed() || !this->task_running_) {
    return;  // before the first loop pass there is no task yet; the first play() retries
  }
  if (this->state_ == speaker::STATE_STOPPING) {
    return;  // a stop is in flight: callers retry once is_stopped() (never yank a pending stop)
  }
  if (this->state_ == speaker::STATE_STOPPED) {
    // A queued stop older than this play() must not kill the new stream.
    xEventGroupClearBits(this->events_, COMMAND_STOP | COMMAND_STOP_GRACEFUL | TASK_DRAINED);
    this->state_ = speaker::STATE_STARTING;
  }
  this->state_ = speaker::STATE_RUNNING;  // no hardware to wait for: the bus always runs
}

void GajaDuplexSpeaker::stop() {
  if (this->is_failed() || !this->task_running_ || this->state_ == speaker::STATE_STOPPED) {
    return;
  }
  xEventGroupSetBits(this->events_, COMMAND_STOP);
  this->state_ = speaker::STATE_STOPPING;
}

void GajaDuplexSpeaker::finish() {
  if (this->is_failed() || !this->task_running_ || this->state_ == speaker::STATE_STOPPED) {
    return;
  }
  xEventGroupSetBits(this->events_, COMMAND_STOP_GRACEFUL);
  this->state_ = speaker::STATE_STOPPING;
}

bool GajaDuplexSpeaker::has_buffered_data() const {
  // Between finish() and the task's drain ack the queue may already be empty while the DMA
  // still plays: report "buffered" for the whole stopping phase (the callers only use this to
  // detect the drained reply, which is about audio, and the amp-off delay covers the DMA tail).
  return this->tx_ring_.available() > 0 || this->state_ == speaker::STATE_STOPPING;
}

void GajaDuplexSpeaker::warn_stream_rate_(const audio::AudioStreamInfo &info) {
  if ((int) info.get_sample_rate() != this->warned_rate_ && info.get_sample_rate() != DuplexBus::RATE) {
    this->warned_rate_ = (int) info.get_sample_rate();
    ESP_LOGW(TAG, "stream info asks for %" PRIu32 " Hz but the duplex bus runs at %" PRIu32
                  " Hz: audio plays at the wrong pitch",
             info.get_sample_rate(), DuplexBus::RATE);
  }
}

void GajaDuplexSpeaker::dump_config() {
  ESP_LOGCONFIG(TAG, "Gaja duplex speaker:");
  ESP_LOGCONFIG(TAG, "  Bus: full-duplex I2S0 @ %" PRIu32 " Hz (shared with the microphone)",
                DuplexBus::RATE);
  ESP_LOGCONFIG(TAG, "  Queue: %u B (%u ms)", (unsigned) TX_RING_BYTES,
                (unsigned) (TX_RING_BYTES * 500 / (DuplexBus::RATE * 2)));
}

void GajaDuplexSpeaker::write_chunk_(uint8_t *chunk, size_t len) {
  // Pad to a full chunk with silence so the DAC pacing stays a constant 20 ms per iteration.
  if (len < DuplexBus::TX_CHUNK_BYTES) {
    memset(chunk + len, 0, DuplexBus::TX_CHUNK_BYTES - len);
  }
  size_t written = 0;
  esp_err_t err =
      i2s_channel_write(this->bus_->tx_handle(), chunk, DuplexBus::TX_CHUNK_BYTES, &written, pdMS_TO_TICKS(200));
  if (err != ESP_OK || written != DuplexBus::TX_CHUNK_BYTES) {
    ESP_LOGW(TAG, "I2S write error: %s (%u/%u B)", esp_err_to_name(err), (unsigned) written,
             (unsigned) DuplexBus::TX_CHUNK_BYTES);
    vTaskDelay(pdMS_TO_TICKS(20));
    return;
  }
  // The reference for the echo canceller: what REALLY left this chunk towards the DAC,
  // decimated with the same filter as the mic path (identical group delay).
  this->push_ref_((const int16_t *) chunk, DuplexBus::TX_CHUNK_BYTES / 2);
}

void GajaDuplexSpeaker::push_ref_(const int16_t *samples48k, size_t count) {
  if (this->bus_ == nullptr || !this->bus_->ref_ring().attached() || count < 3) {
    return;
  }
  count -= count % 3;
  const size_t out = this->ref_decim_.process(samples48k, count, this->ref_out_);
  if (out == 0) {
    return;
  }
  // A RollingRing never refuses: overflow simply overwrites the oldest window, which is the
  // correct behaviour while nobody consumes (device idle) — the ring is a rolling window.
  this->bus_->ref_ring().push((const uint8_t *) this->ref_out_, out * 2);
}

void GajaDuplexSpeaker::tx_task_(void *arg) {
  auto *self = static_cast<GajaDuplexSpeaker *>(arg);
  // 0.3.7: a partial chunk is held back AT MOST one iteration (this flag). Holding it until
  // finish() deadlocked any reply whose length was not a multiple of TX_CHUNK_BYTES (fw 0.3.6,
  // Crab Rave: 18 453 734 % 1920 = 614): gaja_device only calls finish() once the speaker
  // reports empty, and the held chunk kept it non-empty forever -> PLAY_STALL_TIMEOUT_MS cut
  // the reply's tail after 8 s of silence.
  bool held_partial = false;
  while (true) {
    const uint32_t bits = xEventGroupGetBits(self->events_);
    const size_t available = self->tx_ring_.available();

    if (available == 0) {
      held_partial = false;
      if (bits & (COMMAND_STOP | COMMAND_STOP_GRACEFUL)) {
        // Queue is empty (drained or dropped): ack the stop and idle on silence.
        xEventGroupSetBits(self->events_, TASK_DRAINED);
      }
      memset(self->chunk_, 0, DuplexBus::TX_CHUNK_BYTES);
      self->write_chunk_(self->chunk_, 0);  // silence keeps the shared bus clocked
      continue;
    }

    if (bits & COMMAND_STOP) {
      held_partial = false;
      // stop(): break immediately; the consumer side drops the queue (this task IS the
      // consumer, so this cannot race the producer).
      self->tx_ring_.advance(available);
      xEventGroupSetBits(self->events_, TASK_DRAINED);
      memset(self->chunk_, 0, DuplexBus::TX_CHUNK_BYTES);
      self->write_chunk_(self->chunk_, 0);
      continue;
    }

    const size_t n = self->tx_ring_.peek(self->chunk_, DuplexBus::TX_CHUNK_BYTES);
    if (n == 0) {
      continue;  // transient; retry
    }
    if (n < DuplexBus::TX_CHUNK_BYTES && !(bits & COMMAND_STOP_GRACEFUL) && !held_partial) {
      // A partial chunk mid-stream: hold it back ONE iteration (one 20 ms slice) so a batch
      // still in flight can complete it (constant pacing; gaja_device and the pipeline feed
      // multi-chunk batches). After that it plays zero-padded -- write_chunk_ pads.
      held_partial = true;
      memset(self->chunk_, 0, DuplexBus::TX_CHUNK_BYTES);
      self->write_chunk_(self->chunk_, 0);
      continue;
    }
    held_partial = false;
    // A full chunk, the one held back last iteration, or the graceful finish: plays the
    // remainder zero-padded, exactly once (the ring is consumed below), then drains to silence.
    self->tx_ring_.advance(n);
    self->write_chunk_(self->chunk_, n);
  }
}

}  // namespace esphome::gaja_device

#endif  // USE_ESP32
