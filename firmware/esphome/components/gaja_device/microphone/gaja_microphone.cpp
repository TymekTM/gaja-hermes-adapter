#include "gaja_microphone.h"

#ifdef USE_ESP32

#include "esphome/core/hal.h"
#include "esphome/core/log.h"

#include <esp_heap_caps.h>

#include <cstring>
#include <vector>

namespace esphome::gaja_device {

static const char *const TAG = "gaja_microphone";

static constexpr UBaseType_t MAX_LISTENERS = 16;
static constexpr UBaseType_t TASK_PRIORITY = 10;
static constexpr uint32_t TASK_STACK = 4096;

void GajaDuplexMicrophone::setup() {
  this->listeners_ = xSemaphoreCreateCounting(MAX_LISTENERS, MAX_LISTENERS);
  if (this->listeners_ == nullptr) {
    ESP_LOGE(TAG, "Creating the listener semaphore failed");
    this->mark_failed();
    return;
  }
  // The 16 kHz stream every consumer (wake word, gaja_device, sound_level) expects; the 48 kHz
  // bus rate is an implementation detail of the duplex driver (decim3.h).
  this->audio_stream_info_ = audio::AudioStreamInfo(16, 1, DuplexBus::MIC_RATE);

  if (!this->start_bus_()) {
    this->mark_failed();
    return;
  }

  BaseType_t ok = xTaskCreate(GajaDuplexMicrophone::rx_task_, "gaja_mic", TASK_STACK, this,
                              TASK_PRIORITY, nullptr);
  if (ok != pdPASS) {
    ESP_LOGE(TAG, "Could not create the RX task");
    this->mark_failed();
    return;
  }
  this->task_running_.store(true);
}

bool GajaDuplexMicrophone::start_bus_() {
  if (!this->bus_.start()) {
    return false;
  }
  if (!this->bus_.ref_ring().attached()) {
    // Attached by the main gaja_device component (its PSRAM allocation); without it the echo
    // canceller has no reference. The mic still works.
    ESP_LOGW(TAG, "no reference ring attached: playback reference frames will be missing");
  }
  return true;
}

void GajaDuplexMicrophone::dump_config() {
  ESP_LOGCONFIG(TAG,
                "Gaja duplex microphone:\n"
                "  Bus: full-duplex I2S0 @ %" PRIu32 " Hz (shared with the speaker)\n"
                "  Output: 16 bit mono 16 kHz (3:1 decimation)",
                DuplexBus::RATE);
}

void GajaDuplexMicrophone::start() {
  if (this->is_failed() || !this->task_running_.load()) {
    return;
  }
  // One take per active source, exactly like the stock microphone's listener counting.
  xSemaphoreTake(this->listeners_, 0);
  // The bus always runs: "running" only means the decimated audio reaches the callbacks.
  this->state_ = microphone::STATE_RUNNING;
}

void GajaDuplexMicrophone::stop() {
  if (this->is_failed() || !this->task_running_.load()) {
    return;
  }
  // Guarded give: a stop without a matching start (the "leaked listener count" the gaja_device
  // component works around on the stock mic) must not push the count past MAX, or the mic
  // could never report RUNNING again.
  if (uxSemaphoreGetCount(this->listeners_) < MAX_LISTENERS) {
    xSemaphoreGive(this->listeners_);
  }
  if (uxSemaphoreGetCount(this->listeners_) >= MAX_LISTENERS) {
    this->state_ = microphone::STATE_STOPPED;
  }
}

void GajaDuplexMicrophone::rx_task_(void *arg) {
  auto *self = static_cast<GajaDuplexMicrophone *>(arg);
  static_assert(DuplexBus::RX_CHUNK_BYTES % 6 == 0, "chunk must decimate to whole samples");

  uint8_t *raw = (uint8_t *) heap_caps_malloc(DuplexBus::RX_CHUNK_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  int16_t *decimated = (int16_t *) heap_caps_malloc(DuplexBus::RX_CHUNK_BYTES / 3, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (raw == nullptr || decimated == nullptr) {
    ESP_LOGE(TAG, "Could not allocate the RX chunk buffers");
    self->mark_failed();
    vTaskDelete(nullptr);
    return;
  }

  while (true) {
    size_t bytes_read = 0;
    // The RX channel stays enabled for the lifetime of the bus; the read is the pacing timer.
    esp_err_t err = i2s_channel_read(self->bus_.rx_handle(), raw, DuplexBus::RX_CHUNK_BYTES,
                                     &bytes_read, pdMS_TO_TICKS(200));
    if (err != ESP_OK || bytes_read < 6) {
      if (err != ESP_OK) {
        ESP_LOGW(TAG, "I2S read error: %s", esp_err_to_name(err));
        vTaskDelay(pdMS_TO_TICKS(20));
      }
      continue;
    }
    // 16-bit mono: whole samples only.
    bytes_read &= ~(size_t) 1;
    const size_t samples_in = bytes_read / 2 - (bytes_read / 2) % 3;
    if (samples_in == 0) {
      continue;
    }
    const size_t out_samples = self->decim_.process((const int16_t *) raw, samples_in, decimated);
    if (out_samples == 0) {
      continue;
    }
    if (self->state_ != microphone::STATE_RUNNING) {
      continue;  // nobody is listening right now (never the case in full-duplex mode)
    }
    std::vector<uint8_t> data((const uint8_t *) decimated,
                              (const uint8_t *) decimated + out_samples * 2);
    self->data_callbacks_.call(data);
  }
}

}  // namespace esphome::gaja_device

#endif  // USE_ESP32
