#include "gaja_device.h"

#ifdef USE_ESP32

#include "esphome/components/json/json_util.h"
#include "esphome/core/hal.h"
#include "esphome/core/log.h"
#include "volume_map.h"

#ifdef USE_NETWORK
#include "esphome/components/network/util.h"
#endif

#include <esp_heap_caps.h>
#include <esp_websocket_client.h>

#include <cmath>
#include <cstring>
#include <inttypes.h>

namespace esphome::gaja_device {

static const char *const TAG = "gaja_device";

// Uplink audio: PCM16 LE mono 16 kHz in 20 ms frames (docs/DEVICE.md recommends 20 ms / 640 B).
static constexpr uint32_t MIC_RATE = 16000;
static constexpr size_t FRAME_BYTES = MIC_RATE * 2 / 50;

// Buffer sizing (PSRAM unless noted). The Atom VoiceS3R has 8 MB octal PSRAM.
static constexpr size_t SEND_RING_BYTES = 64 * 1024;     // ~2 s of live mic audio
static constexpr size_t PLAYBACK_RING_BYTES = 1024 * 1024;  // ~10.9 s of 48 kHz reply PCM

// Timing constants, in milliseconds unless stated otherwise.
static constexpr uint32_t AMP_SETTLE_MS = 50;        // NS4150B enable -> first sample (verified)
static constexpr uint32_t AMP_OFF_DELAY_MS = 100;    // I2S buffer drains after the last sample
static constexpr uint32_t RECONNECT_BASE_MS = 2000;
static constexpr uint32_t RECONNECT_MAX_MS = 60000;
static constexpr uint32_t AUTH_FAIL_MIN_DELAY_MS = 30000;
static constexpr uint32_t HELLO_TIMEOUT_MS = 6000;   // server must answer ready within ~5 s
static constexpr uint32_t WAIT_REPLY_TIMEOUT_MS = 5000;  // listen_end without a following reply
static constexpr uint32_t MIC_STOP_TIMEOUT_MS = 5000;
static constexpr uint32_t SEND_TIMEOUT_TICKS = 100;  // FreeRTOS ticks for a single uplink frame
static constexpr uint32_t MIC_BYTES_PER_MS = MIC_RATE * 2 / 1000;

const char *gaja_state_to_string(GajaState state) {
  switch (state) {
    case GajaState::RECONNECTING:
      return "RECONNECTING";
    case GajaState::IDLE:
      return "IDLE";
    case GajaState::WAKING:
      return "WAKING";
    case GajaState::STREAMING:
      return "STREAMING";
    case GajaState::FOLLOWUP:
      return "FOLLOWUP";
    case GajaState::WAIT_REPLY:
      return "WAIT_REPLY";
    case GajaState::PREPARE_PLAY:
      return "PREPARE_PLAY";
    case GajaState::PLAYING:
      return "PLAYING";
    case GajaState::LINGER:
      return "LINGER";
    default:
      return "UNKNOWN";
  }
}

// ---- buffers -----------------------------------------------------------------------------

static uint8_t *alloc_bytes(size_t size, bool prefer_psram) {
  if (prefer_psram) {
    uint8_t *ptr = (uint8_t *) heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (ptr != nullptr) {
      return ptr;
    }
    ESP_LOGW(TAG, "%u bytes not available in PSRAM, using internal RAM", size);
  }
  return (uint8_t *) heap_caps_malloc(size, MALLOC_CAP_8BIT);
}

bool SpscRing::allocate(size_t capacity) {
  this->capacity_ = capacity;
  this->buffer_ = alloc_bytes(capacity, true);
  return this->buffer_ != nullptr;
}

bool SpscRing::push(const uint8_t *data, size_t len) {
  const size_t head = this->head_.load(std::memory_order_relaxed);
  const size_t tail = this->tail_.load(std::memory_order_acquire);
  const size_t used = head - tail;
  if (len > this->capacity_ - used) {
    this->dropped_++;
    return false;  // drop NEW data; the consumer keeps what it already has
  }
  const size_t start = head % this->capacity_;
  const size_t first = std::min(len, this->capacity_ - start);
  memcpy(this->buffer_ + start, data, first);
  if (len > first) {
    memcpy(this->buffer_, data + first, len - first);
  }
  this->head_.store(head + len, std::memory_order_release);
  return true;
}

size_t SpscRing::peek(uint8_t *dst, size_t len) const {
  const size_t head = this->head_.load(std::memory_order_acquire);
  const size_t tail = this->tail_.load(std::memory_order_relaxed);
  len = std::min(len, head - tail);
  const size_t start = tail % this->capacity_;
  const size_t first = std::min(len, this->capacity_ - start);
  memcpy(dst, this->buffer_ + start, first);
  if (len > first) {
    memcpy(dst + first, this->buffer_, len - first);
  }
  return len;
}

void SpscRing::advance(size_t len) {
  const size_t head = this->head_.load(std::memory_order_acquire);
  const size_t tail = this->tail_.load(std::memory_order_relaxed);
  len = std::min(len, head - tail);
  this->tail_.store(tail + len, std::memory_order_release);  // indices stay monotonic
}

bool RollingRing::allocate(size_t capacity) {
  this->capacity_ = capacity;
  if (capacity == 0) {
    return true;  // preroll disabled; push() is a no-op
  }
  this->buffer_ = alloc_bytes(capacity, true);
  return this->buffer_ != nullptr;
}

void RollingRing::push(const uint8_t *data, size_t len) {
  if (this->capacity_ == 0) {  // preroll_ms = 0: no ring at all
    return;
  }
  if (len >= this->capacity_) {  // keep only the newest window
    data += len - this->capacity_;
    len = this->capacity_;
  }
  const size_t head = this->head_.load(std::memory_order_relaxed);
  const size_t start = head % this->capacity_;
  const size_t first = std::min(len, this->capacity_ - start);
  memcpy(this->buffer_ + start, data, first);
  if (len > first) {
    memcpy(this->buffer_, data + first, len - first);
  }
  this->head_.store(head + len, std::memory_order_release);
}

size_t RollingRing::head_pos() const { return this->head_.load(std::memory_order_acquire); }

size_t RollingRing::tail_pos() const {
  const size_t head = this->head_.load(std::memory_order_acquire);
  return head > this->capacity_ ? head - this->capacity_ : 0;
}

size_t RollingRing::peek_abs(size_t abs, uint8_t *dst, size_t len) const {
  if (this->capacity_ == 0) {
    return 0;
  }
  const size_t head = this->head_.load(std::memory_order_acquire);
  const size_t tail = head > this->capacity_ ? head - this->capacity_ : 0;
  if (abs < tail) {
    return 0;  // overwritten (producer lapped us); caller pads with silence
  }
  len = std::min(len, head - abs);
  const size_t start = abs % this->capacity_;
  const size_t first = std::min(len, this->capacity_ - start);
  memcpy(dst, this->buffer_ + start, first);
  if (len > first) {
    memcpy(dst + first, this->buffer_, len - first);
  }
  return len;
}

// ---- setup / config ----------------------------------------------------------------------

void GajaDevice::setup() {
  this->out_rate_ = this->speaker_->get_audio_stream_info().get_sample_rate();
  if (this->out_rate_ < 8000 || this->out_rate_ > 48000) {
    ESP_LOGE(TAG, "speaker sample rate %" PRIu32 " Hz is outside the protocol range 8000..48000", this->out_rate_);
    this->mark_failed();
    return;
  }

  if (!this->preroll_ring_.allocate(this->preroll_ms_ * MIC_BYTES_PER_MS) ||
      !this->send_ring_.allocate(SEND_RING_BYTES) || !this->playback_ring_.allocate(PLAYBACK_RING_BYTES)) {
    ESP_LOGE(TAG, "failed to allocate audio buffers");
    this->mark_failed();
    return;
  }
  this->frame_ = (uint8_t *) heap_caps_malloc(FRAME_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  this->feed_ = (uint8_t *) heap_caps_malloc(this->out_rate_ / 5, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (this->frame_ == nullptr || this->feed_ == nullptr) {
    ESP_LOGE(TAG, "failed to allocate frame buffers");
    this->mark_failed();
    return;
  }
  memset(this->text_slots_, 0, sizeof(this->text_slots_));

  // Tap the microphone stream. This source is active (it holds the microphone on whenever the
  // device is not playing), so the mic never restarts between the wake detection and streaming.
  this->mic_source_->add_data_callback([this](const std::vector<uint8_t> &data) { this->on_mic_data_(data); });

  esp_websocket_client_config_t cfg = {};
  cfg.uri = this->server_url_.c_str();  // points at our member, valid for the lifetime of the client
  cfg.buffer_size = 16384;              // >= the largest downlink message (100 ms @ 48 kHz = 9600 B)
  cfg.disable_auto_reconnect = true;    // reconnection/backoff is owned by this component
  cfg.network_timeout_ms = 8000;
  cfg.ping_interval_sec = 10;           // protocol-level keepalive; answered automatically
  cfg.keep_alive_enable = true;
  esp_websocket_client_handle_t client = esp_websocket_client_init(&cfg);
  if (client == nullptr) {
    ESP_LOGE(TAG, "esp_websocket_client_init failed for %s", this->server_url_.c_str());
    this->mark_failed();
    return;
  }
  this->ws_client_ = client;
  esp_websocket_register_events(client, WEBSOCKET_EVENT_ANY, websocket_event_handler_, this);
}

void GajaDevice::dump_config() {
  ESP_LOGCONFIG(TAG, "Gaja device:");
  ESP_LOGCONFIG(TAG, "  Server: %s", this->server_url_.c_str());
  ESP_LOGCONFIG(TAG, "  Device id: %s, language: %s, firmware: %s", this->device_id_.c_str(),
                this->lang_.c_str(), this->fw_version_.c_str());
  ESP_LOGCONFIG(TAG, "  caps: aec=false out_rates=[%" PRIu32 "] preroll_ms=%" PRIu32 "", this->out_rate_, this->preroll_ms_);
  ESP_LOGCONFIG(TAG, "  wake: word=%s score=%.2f", this->wake_word_.c_str(), this->wake_score_);
  ESP_LOGCONFIG(TAG, "  adaptive volume: quiet_db=%.1f loud_db=%.1f vol_min=%.2f vol_max=%.2f", this->quiet_db_,
                this->loud_db_, this->vol_min_, this->vol_max_);
}

// ---- main loop ----------------------------------------------------------------------------

void GajaDevice::loop() {
  const uint32_t now = millis();

  this->process_text_queue_();
  this->service_reconnect_();

  switch (this->state_) {
    case GajaState::RECONNECTING:
      break;
    case GajaState::IDLE:
      // Nothing to do per iteration: the wake word component detects, the mic task fills the ring.
      // Unsolicited reply audio (a timer/reminder announcement) can arrive while idle.
      if (this->playback_ring_.available() > 0) {
        this->begin_playback_transition_();
      }
      break;
    case GajaState::WAKING:
      this->service_waking_();
      break;
    case GajaState::STREAMING:
    case GajaState::FOLLOWUP:
      this->service_streaming_();
      if (this->state_ == GajaState::FOLLOWUP && now >= this->followup_deadline_ms_) {
        ESP_LOGW(TAG, "follow-up window of %" PRIu32 " ms expired without listen_end; returning to idle",
                 this->follow_up_ms_);
        this->close_mic_stream_();
        this->enter_idle_();
      }
      break;
    case GajaState::WAIT_REPLY:
      if (this->playback_ring_.available() > 0) {
        // Audio frames beat or replaced audio_start.
        this->begin_playback_transition_();
      } else if (now - this->state_entered_ms_ > WAIT_REPLY_TIMEOUT_MS) {
        ESP_LOGI(TAG, "no reply after listen_end; back to wake word");
        this->enter_idle_();
      }
      break;
    case GajaState::PREPARE_PLAY:
      if (this->mic_stopped_()) {
        if (!this->amp_on_) {
          this->speaker_enable_->turn_on();
          this->amp_on_ = true;
          this->amp_on_ms_ = now;
        } else if (now - this->amp_on_ms_ >= AMP_SETTLE_MS) {
          if (!this->reply_volume_set_) {
            this->apply_reply_volume_();
          }
          this->state_ = GajaState::PLAYING;
          this->state_entered_ms_ = now;
        }
      } else if (now - this->state_entered_ms_ > MIC_STOP_TIMEOUT_MS) {
        ESP_LOGE(TAG, "microphone did not release the I2S bus; attempting playback anyway");
        this->state_ = GajaState::PLAYING;
        this->state_entered_ms_ = now;
      }
      break;
    case GajaState::PLAYING:
      this->service_speaker_();
      break;
    case GajaState::LINGER:
      if (this->playback_ring_.available() > 0) {
        // Another utterance of the same reply (audio_start may have been processed already).
        this->begin_playback_transition_();
      } else if (now >= this->linger_deadline_ms_) {
        this->enter_idle_();
      }
      break;
  }

  if (this->send_ring_.dropped() > 0 && now - this->last_send_drop_log_ms_ > 10000) {
    ESP_LOGW(TAG, "%" PRIu32 " uplink chunks dropped (slow network?)", this->send_ring_.dropped());
    this->last_send_drop_log_ms_ = now;
  }
  if (this->playback_ring_.dropped() > 0 && now - this->last_play_drop_log_ms_ > 10000) {
    ESP_LOGW(TAG, "%" PRIu32 " downlink chunks dropped (reply audio clipped)", this->playback_ring_.dropped());
    this->last_play_drop_log_ms_ = now;
  }
}

// ---- uplink: wake, pre-roll, live frames ----------------------------------------------------

void GajaDevice::on_wake(const std::string &word) {
  if (this->state_ != GajaState::IDLE || !this->ws_connected_ || !this->server_ready_) {
    ESP_LOGW(TAG, "wake '%s' ignored in state %s (connected=%d ready=%d)", word.c_str(), this->state_str_(),
             (int) this->ws_connected_.load(), (int) this->server_ready_);
    return;
  }
  this->wake_word_ = word;
  this->mww_->stop();  // micro_wake_word stops itself (stop_after_detection); make it explicit

  // Snapshot the pre-roll window: everything the ring holds at the wake instant, oldest first.
  const size_t head = this->preroll_ring_.head_pos();
  this->preroll_pos_ = this->preroll_ring_.tail_pos();
  this->preroll_remaining_ = head - this->preroll_pos_;
  const uint32_t preroll_ms = this->preroll_remaining_ / MIC_BYTES_PER_MS;

  // Reset the level statistics: they describe this wake's utterance (pre-roll + live).
  memset(this->level_hist_, 0, sizeof(this->level_hist_));
  this->level_frames_ = 0;
  this->frame_len_ = 0;
  this->reply_volume_set_ = false;

  ESP_LOGI(TAG, "wake word '%s' -> sending %" PRIu32 " ms of pre-roll", word.c_str(), preroll_ms);
  if (!this->send_json_wake_(preroll_ms)) {
    return;
  }
  this->state_ = GajaState::WAKING;
  this->state_entered_ms_ = millis();
}

bool GajaDevice::send_json_wake_(uint32_t preroll_ms) {
  char buf[192];
  const int len = snprintf(buf, sizeof(buf), "{\"type\":\"wake\",\"word\":\"%s\",\"score\":%.2f,\"preroll_ms\":%u}",
                           this->wake_word_.c_str(), this->wake_score_, preroll_ms);
  return this->send_text_(buf, len);
}

void GajaDevice::service_waking_() {
  // Send the pre-roll as a burst (server-side docs: "sent as fast as possible").
  uint32_t sent = 0;
  while (this->preroll_remaining_ > 0 && sent < 32) {
    const size_t want = std::min(FRAME_BYTES - this->frame_len_, this->preroll_remaining_);
    const size_t got = this->preroll_ring_.peek_abs(this->preroll_pos_, this->frame_ + this->frame_len_, want);
    if (got < want) {  // window overwritten under us (should not happen at 1.5 s depth)
      memset(this->frame_ + this->frame_len_ + got, 0, want - got);
    }
    this->preroll_pos_ += want;
    this->preroll_remaining_ -= want;
    this->frame_len_ += want;
    if (this->frame_len_ == FRAME_BYTES) {
      if (!this->send_frame_(FRAME_BYTES)) {
        return;
      }
      this->frame_len_ = 0;
      sent++;
    }
  }
  if (this->preroll_remaining_ == 0) {
    this->state_ = GajaState::STREAMING;
    this->state_entered_ms_ = millis();
    ESP_LOGI(TAG, "pre-roll sent (%u frames), streaming live audio", this->level_frames_);
  }
}

void GajaDevice::service_streaming_() {
  uint32_t sent = 0;
  while (this->frame_len_ < FRAME_BYTES && sent < 8) {
    const size_t got = this->send_ring_.peek(this->frame_ + this->frame_len_, FRAME_BYTES - this->frame_len_);
    if (got == 0) {
      break;
    }
    this->send_ring_.advance(got);
    this->frame_len_ += got;
    if (this->frame_len_ == FRAME_BYTES) {
      if (!this->send_frame_(FRAME_BYTES)) {
        return;
      }
      this->frame_len_ = 0;
      sent++;
    }
  }
}

bool GajaDevice::send_frame_(size_t len) {
  this->note_level_frame_(this->frame_, len);
  esp_websocket_client_handle_t client = (esp_websocket_client_handle_t) this->ws_client_;
  const int rc = esp_websocket_client_send_bin(client, (const char *) this->frame_, len,
                                               pdMS_TO_TICKS(SEND_TIMEOUT_TICKS));
  if (rc < 0) {
    // One stalled frame is a hiccup (audio keeps buffering in the ring); three in a row means
    // the connection is gone.
    if (++this->send_failures_ >= 3) {
      ESP_LOGW(TAG, "uplink send failed %" PRIu16 " times; reconnecting", this->send_failures_);
      this->enter_reconnecting_("uplink send failed");
    }
    return false;
  }
  this->send_failures_ = 0;
  return true;
}

void GajaDevice::close_mic_stream_() {
  // Flush a partial frame, zero-padded, like the reference simulator does.
  if (this->frame_len_ > 0 && this->ws_connected_) {
    memset(this->frame_ + this->frame_len_, 0, FRAME_BYTES - this->frame_len_);
    this->send_frame_(FRAME_BYTES);
  }
  this->frame_len_ = 0;
  this->preroll_remaining_ = 0;
  this->send_ring_.reset();
}

// ---- microphone callback (runs on the microphone task) --------------------------------------

void GajaDevice::on_mic_data_(const std::vector<uint8_t> &data) {
  if (data.empty()) {
    return;
  }
  // The pre-roll ring always rolls (audio before the wake, for the next wake).
  this->preroll_ring_.push(data.data(), data.size());
  // While a wake stream is open, the same audio goes to the uplink ring.
  const GajaState state = this->state_.load(std::memory_order_relaxed);
  if (state == GajaState::WAKING || state == GajaState::STREAMING || state == GajaState::FOLLOWUP) {
    this->send_ring_.push(data.data(), data.size());
  }
}

// ---- downlink: reply playback ----------------------------------------------------------------

void GajaDevice::begin_playback_transition_() {
  const GajaState state = this->state_;
  if (state == GajaState::WAKING || state == GajaState::STREAMING || state == GajaState::FOLLOWUP) {
    this->close_mic_stream_();
  }
  // Half-duplex I2S bus: the microphone must fully stop before the speaker may start.
  this->mic_source_->stop();
  this->mww_->stop();
  this->speaker_finishing_ = false;
  this->state_ = GajaState::PREPARE_PLAY;
  this->state_entered_ms_ = millis();
}

void GajaDevice::service_speaker_() {
  const uint32_t now = millis();
  const bool drained = this->audio_end_seen_ && this->playback_ring_.available() == 0 &&
                       !this->speaker_->has_buffered_data();
  if (!drained) {
    if (this->speaker_finishing_) {
      // New audio arrived after finish() began: wait for the speaker to stop, then resume.
      if (this->speaker_->is_stopped()) {
        this->speaker_finishing_ = false;
      }
      return;
    }
    const size_t feed_capacity = this->out_rate_ / 5;  // 100 ms of PCM16 mono
    while (this->playback_ring_.available() > 0) {
      const size_t n = this->playback_ring_.peek(this->feed_, feed_capacity);
      if (n == 0) {
        break;
      }
      const size_t written = this->speaker_->play(this->feed_, n, 0);
      if (written == 0) {
        break;  // speaker buffer full; retry on the next loop iteration
      }
      this->playback_ring_.advance(written);
      if (written < n) {
        break;
      }
    }
    return;
  }

  if (!this->speaker_finishing_) {
    this->speaker_->finish();  // stop gracefully once everything buffered has been played
    this->speaker_finishing_ = true;
    return;
  }
  if (this->speaker_->is_stopped()) {
    if (this->amp_off_deadline_ms_ == 0) {
      this->amp_off_deadline_ms_ = now + AMP_OFF_DELAY_MS;
    } else if (now >= this->amp_off_deadline_ms_) {
      this->finish_reply_();
    }
  }
}

void GajaDevice::finish_reply_() {
  if (this->amp_on_) {
    this->speaker_enable_->turn_off();
    this->amp_on_ = false;
  }
  this->speaker_finishing_ = false;
  this->amp_off_deadline_ms_ = 0;
  this->audio_end_seen_ = false;
  if (this->follow_up_received_) {
    this->open_followup_();
  } else {
    ESP_LOGD(TAG, "reply finished; lingering %" PRIu32 " ms for more sentences", this->reply_gap_ms_);
    this->state_ = GajaState::LINGER;
    this->state_entered_ms_ = millis();
    this->linger_deadline_ms_ = millis() + this->reply_gap_ms_;
  }
}

void GajaDevice::open_followup_() {
  this->follow_up_received_ = false;
  this->reply_volume_set_ = false;  // the next reply measures the follow-up speech level
  this->mww_->stop();               // no wake phrase needed while the window is open
  this->mic_source_->start();       // mic back on (the speaker has released the bus)
  this->state_ = GajaState::FOLLOWUP;
  this->state_entered_ms_ = millis();
  this->followup_deadline_ms_ = millis() + this->follow_up_ms_ + 4000;
  ESP_LOGI(TAG, "follow-up window open for %" PRIu32 " ms (no wake phrase needed)", this->follow_up_ms_);
}

// ---- state helpers ----------------------------------------------------------------------------

void GajaDevice::enter_idle_() {
  this->send_ring_.reset();
  this->frame_len_ = 0;
  this->preroll_remaining_ = 0;
  this->follow_up_received_ = false;
  this->audio_end_seen_ = false;
  this->reply_volume_set_ = false;
  this->amp_off_deadline_ms_ = 0;
  this->mic_source_->start();
  this->mww_->start();
  this->state_ = GajaState::IDLE;
  this->state_entered_ms_ = millis();
}

void GajaDevice::enter_reconnecting_(const char *reason) {
  ESP_LOGW(TAG, "connection lost (%s) in state %s", reason, this->state_str_());
  this->state_ = GajaState::RECONNECTING;
  this->state_entered_ms_ = millis();
  this->mic_source_->stop();
  this->mww_->stop();
  if (!this->speaker_->is_stopped()) {
    this->speaker_->stop();
  }
  if (this->amp_on_) {
    this->speaker_enable_->turn_off();
    this->amp_on_ = false;
  }
  this->speaker_finishing_ = false;
  this->amp_off_deadline_ms_ = 0;
  this->frame_len_ = 0;
  this->preroll_remaining_ = 0;
  this->follow_up_received_ = false;
  this->audio_end_seen_ = false;
  this->send_ring_.reset();
  this->playback_ring_.reset();
  this->hello_sent_ = false;
  this->server_ready_ = false;
  if (this->ws_client_ != nullptr) {
    esp_websocket_client_stop((esp_websocket_client_handle_t) this->ws_client_);
  }
  if (!this->reconnect_pending_) {
    this->schedule_next_attempt_();
  }
}

void GajaDevice::schedule_next_attempt_() {
  this->reconnect_failures_++;
  uint32_t delay = RECONNECT_BASE_MS;
  for (uint32_t i = 1; i < this->reconnect_failures_ && delay < RECONNECT_MAX_MS; i++) {
    delay *= 2;
  }
  delay = std::min(delay, RECONNECT_MAX_MS);
  if (this->auth_failed_) {
    delay = std::max(delay, AUTH_FAIL_MIN_DELAY_MS);
  }
  this->reconnect_pending_ = true;
  this->next_attempt_ms_ = millis() + delay;
  ESP_LOGI(TAG, "reconnecting to %s in %" PRIu32 " ms (attempt %" PRIu32 ")", this->server_url_.c_str(), delay,
           this->reconnect_failures_);
}

void GajaDevice::service_reconnect_() {
  const uint32_t now = millis();
#ifdef USE_NETWORK
  const bool network_up = network::is_connected();
#else
  const bool network_up = true;
#endif

  if (!this->ws_started_) {
    if (!network_up) {
      return;
    }
    this->ws_started_ = true;
    ESP_LOGI(TAG, "connecting to %s", this->server_url_.c_str());
    esp_websocket_client_start((esp_websocket_client_handle_t) this->ws_client_);
    this->schedule_next_attempt_();  // arm the retry in case this attempt fails
    return;
  }

  if (!this->ws_connected_.load() && this->state_ != GajaState::RECONNECTING) {
    this->enter_reconnecting_("websocket closed");
    return;
  }

  if (this->reconnect_pending_ && now >= this->next_attempt_ms_) {
    if (!network_up) {
      this->next_attempt_ms_ = now + 1000;
      return;
    }
    this->reconnect_pending_ = false;
    esp_websocket_client_stop((esp_websocket_client_handle_t) this->ws_client_);
    esp_err_t err = esp_websocket_client_start((esp_websocket_client_handle_t) this->ws_client_);
    if (err != ESP_OK) {
      ESP_LOGW(TAG, "websocket start failed: %s", esp_err_to_name(err));
    }
    this->schedule_next_attempt_();
  }

  if (this->ws_connected_.load() && !this->hello_sent_) {
    this->send_hello_();
  }
  if (this->hello_sent_ && !this->server_ready_ && now - this->hello_sent_ms_ > HELLO_TIMEOUT_MS) {
    ESP_LOGW(TAG, "server did not answer hello within %u ms", HELLO_TIMEOUT_MS);
    this->enter_reconnecting_("no ready");
    return;
  }
  if (this->pending_pong_) {
    this->pending_pong_ = false;
    static const char PONG[] = "{\"type\":\"pong\"}";
    this->send_text_(PONG, sizeof(PONG) - 1);
  }
}

void GajaDevice::send_hello_() {
  auto payload = json::build_json([this](JsonObject root) {
    root["type"] = "hello";
    root["client"] = "device";
    root["device_id"] = this->device_id_;
    root["token"] = this->token_;
    root["fw"] = this->fw_version_;
    root["lang"] = this->lang_;
    JsonObject caps = root["caps"].to<JsonObject>();
    caps["aec"] = false;  // one shared I2S bus: the mic is muted while the speaker runs
    JsonArray rates = caps["out_rates"].to<JsonArray>();
    rates.add(this->out_rate_);  // the speaker's native rate; the server resamples TTS to it
    caps["preroll_ms"] = this->preroll_ms_;
  });
  if (this->send_text_(payload.data(), payload.size())) {
    this->hello_sent_ = true;
    this->hello_sent_ms_ = millis();
    this->server_ready_ = false;
    ESP_LOGI(TAG, "connected; hello sent (device_id=%s out_rates=[%" PRIu32 "] preroll_ms=%" PRIu32 ")",
             this->device_id_.c_str(), this->out_rate_, this->preroll_ms_);
  }
}

bool GajaDevice::send_text_(const char *data, size_t len) {
  esp_websocket_client_handle_t client = (esp_websocket_client_handle_t) this->ws_client_;
  const int rc = esp_websocket_client_send_text(client, data, len, pdMS_TO_TICKS(SEND_TIMEOUT_TICKS));
  if (rc < 0) {
    ESP_LOGW(TAG, "text send failed (%.*s)", (int) std::min<size_t>(len, 64), data);
    this->enter_reconnecting_("text send failed");
    return false;
  }
  return true;
}

// ---- websocket events (websocket client task) -------------------------------------------------

void GajaDevice::websocket_event_handler_(void *arg, esp_event_base_t base, int32_t event_id, void *event_data) {
  (void) base;
  GajaDevice *self = (GajaDevice *) arg;
  switch (event_id) {
    case WEBSOCKET_EVENT_CONNECTED:
      self->ws_connected_.store(true);
      ESP_LOGI(TAG, "websocket connected");
      break;
    case WEBSOCKET_EVENT_DISCONNECTED:
    case WEBSOCKET_EVENT_CLOSED:
      self->ws_connected_.store(false);
      ESP_LOGW(TAG, "websocket %s", event_id == WEBSOCKET_EVENT_CLOSED ? "closed by server" : "disconnected");
      break;
    case WEBSOCKET_EVENT_ERROR:
      ESP_LOGW(TAG, "websocket transport error");
      break;
    case WEBSOCKET_EVENT_DATA: {
      esp_websocket_event_data_t *d = (esp_websocket_event_data_t *) event_data;
      if (d->data_len <= 0) {
        break;
      }
      if (d->op_code == 0x02) {  // binary: reply PCM, appended in order (fragmentation tolerated)
        self->playback_ring_.push((const uint8_t *) d->data_ptr, d->data_len);
      } else if (d->op_code == 0x01) {  // text JSON, accumulated per message
        if (d->payload_offset == 0) {
          self->text_slot_len_[self->text_write_] = 0;
        }
        const uint8_t slot = self->text_write_;
        size_t space = TEXT_SLOT_SIZE - self->text_slot_len_[slot];
        size_t copy = std::min((size_t) d->data_len, space);
        memcpy(self->text_slots_[slot] + self->text_slot_len_[slot], d->data_ptr, copy);
        self->text_slot_len_[slot] += copy;
        if (d->payload_offset + d->data_len >= d->payload_len) {  // message complete
          self->text_slots_[slot][self->text_slot_len_[slot]] = '\0';
          uint8_t next = (slot + 1) % TEXT_SLOTS;
          if (next != self->text_read_) {
            self->text_write_ = next;
          }  // queue full: the message stays in the slot and is overwritten by the next one
        }
      }
      break;
    }
    default:
      break;
  }
}

// ---- server messages ---------------------------------------------------------------------------

void GajaDevice::process_text_queue_() {
  while (this->text_read_ != this->text_write_) {
    const uint8_t slot = this->text_read_;
    this->handle_server_json_(this->text_slots_[slot], this->text_slot_len_[slot]);
    this->text_read_ = (this->text_read_ + 1) % TEXT_SLOTS;
  }
}

void GajaDevice::handle_server_json_(const char *data, size_t len) {
  const bool parsed = json::parse_json(std::string(data, len), [this](JsonObject root) -> bool {
    const char *type = root["type"] | "";
    if (strcmp(type, "ready") == 0) {
      const int rate = root["sample_rate"] | 0;
      JsonObject device = root["device"];
      if (!device.isNull()) {
        ESP_LOGI(TAG, "ready: out_rate=%d device=%s aec=%d verify_wake=%d verify_ms=%d follow_up_ms=%d ack=%s",
                 rate, device["id"] | "?", device["aec"] | 0, device["verify_wake"] | 0, device["verify_ms"] | 0,
                 device["follow_up_ms"] | 0, device["ack"] | "?");
      } else {
        ESP_LOGI(TAG, "ready: out_rate=%d", rate);
      }
      this->on_ready_(rate);
    } else if (strcmp(type, "wake_accepted") == 0) {
      this->on_wake_accepted_();
    } else if (strcmp(type, "wake_rejected") == 0) {
      this->on_wake_rejected_(root["reason"] | "?");
    } else if (strcmp(type, "listen_end") == 0) {
      this->on_listen_end_();
    } else if (strcmp(type, "follow_up") == 0) {
      this->on_follow_up_((uint32_t) (root["ms"] | 0));
    } else if (strcmp(type, "audio_start") == 0) {
      this->on_audio_start_(root["sample_rate"] | 0, root["id"] | 0);
    } else if (strcmp(type, "audio_end") == 0) {
      this->on_audio_end_();
    } else if (strcmp(type, "ping") == 0) {
      this->on_ping_();
    } else if (strcmp(type, "error") == 0) {
      this->on_error_(root["message"] | "?");
    } else if (strcmp(type, "transcript") == 0) {
      JsonVariantConst heard_v = root["heard"];
      const char *heard =
          (!heard_v.isNull() && heard_v.is<const char *>()) ? heard_v.as<const char *>() : nullptr;
      if (heard != nullptr) {
        ESP_LOGI(TAG, "transcript: heard '%s' -> '%s'", heard, (const char *) (root["text"] | ""));
      } else {
        ESP_LOGI(TAG, "transcript: '%s'", (const char *) (root["text"] | ""));
      }
    } else if (strcmp(type, "utterance") == 0) {
      ESP_LOGD(TAG, "utterance %d: %s", root["id"] | 0, (const char *) (root["text"] | ""));
    } else if (strcmp(type, "metrics") == 0) {
      ESP_LOGD(TAG, "metrics: turn=%d kind=%s route=%s e2e_first_audio_ms=%d", root["turn"] | 0,
               (const char *) (root["kind"] | "?"), (const char *) (root["route"] | "-"),
               root["e2e_first_audio_ms"] | 0);
    } else if (strcmp(type, "state") == 0) {
      ESP_LOGD(TAG, "state: %s", (const char *) (root["state"] | "?"));
    } else if (strcmp(type, "timer") == 0) {
      ESP_LOGI(TAG, "timer fired: %s (%s) -- announcement follows", (const char *) (root["label"] | ""),
               (const char *) (root["kind"] | "timer"));
    } else if (strcmp(type, "vad") == 0 || strcmp(type, "handoff") == 0 || strcmp(type, "tool") == 0 ||
               strcmp(type, "notice") == 0) {
      ESP_LOGD(TAG, "%s event", type);
    } else if (strcmp(type, "pong") == 0) {
      // answer to our own application-level ping (we do not send one); ignore
    } else {
      ESP_LOGD(TAG, "unhandled server message type '%s'", type);
    }
    return true;
  });
  if (!parsed) {
    ESP_LOGW(TAG, "malformed server message (%u bytes)", (unsigned) len);
  }
}

void GajaDevice::on_ready_(int sample_rate) {
  this->server_ready_ = true;
  this->reconnect_pending_ = false;
  this->reconnect_failures_ = 0;
  this->auth_failed_ = false;
  if (sample_rate != 0 && (uint32_t) sample_rate != this->out_rate_) {
    ESP_LOGW(TAG, "negotiated out_rate %d differs from the speaker rate %" PRIu32 "; audio will play at the wrong speed",
             sample_rate, this->out_rate_);
  }
  this->enter_idle_();
}

void GajaDevice::on_wake_accepted_() {
  ESP_LOGI(TAG, "wake accepted; keep talking");
  // Streaming continues until listen_end.
}

void GajaDevice::on_wake_rejected_(const std::string &reason) {
  ESP_LOGW(TAG, "wake rejected (%s) -> idle", reason.c_str());
  const GajaState state = this->state_;
  if (state == GajaState::WAKING || state == GajaState::STREAMING || state == GajaState::FOLLOWUP ||
      state == GajaState::WAIT_REPLY) {
    this->enter_idle_();
  }
}

void GajaDevice::on_listen_end_() {
  const GajaState state = this->state_;
  switch (state) {
    case GajaState::WAKING:
    case GajaState::STREAMING:
    case GajaState::FOLLOWUP:
      ESP_LOGI(TAG, "listen_end: stop streaming, wait for the reply");
      this->close_mic_stream_();
      this->mic_source_->stop();  // free the shared I2S bus for the speaker
      this->state_ = GajaState::WAIT_REPLY;
      this->state_entered_ms_ = millis();
      break;
    case GajaState::LINGER:
    case GajaState::WAIT_REPLY:
      this->enter_idle_();
      break;
    default:
      ESP_LOGD(TAG, "listen_end ignored in state %s", this->state_str_());
      break;
  }
}

void GajaDevice::on_follow_up_(uint32_t ms) {
  this->follow_up_ms_ = ms;
  this->follow_up_received_ = true;
  const GajaState state = this->state_;
  if (state == GajaState::WAIT_REPLY || state == GajaState::LINGER || state == GajaState::IDLE) {
    this->open_followup_();
  } else {
    ESP_LOGI(TAG, "follow_up ms=%" PRIu32 " queued (state %s)", ms, this->state_str_());
  }
}

void GajaDevice::on_audio_start_(int sample_rate, int id) {
  ESP_LOGI(TAG, "audio_start id=%d rate=%d", id, sample_rate);
  if (sample_rate != 0 && (uint32_t) sample_rate != this->out_rate_) {
    ESP_LOGW(TAG, "audio arrives at %d Hz but the speaker runs at %" PRIu32 " Hz", sample_rate, this->out_rate_);
  }
  this->audio_end_seen_ = false;
  const GajaState state = this->state_;
  switch (state) {
    case GajaState::IDLE:
    case GajaState::WAKING:
    case GajaState::STREAMING:
    case GajaState::FOLLOWUP:
    case GajaState::WAIT_REPLY:
    case GajaState::LINGER:
      this->begin_playback_transition_();
      break;
    case GajaState::PREPARE_PLAY:
    case GajaState::PLAYING:
      break;  // another utterance of the same reply
    default:
      break;
  }
}

void GajaDevice::on_audio_end_() {
  ESP_LOGD(TAG, "audio_end; draining speaker");
  this->audio_end_seen_ = true;
}

void GajaDevice::on_ping_() { this->pending_pong_ = true; }

void GajaDevice::on_error_(const std::string &message) {
  ESP_LOGE(TAG, "server error: %s", message.c_str());
  if (message == "unauthorized") {
    this->auth_failed_ = true;
    ESP_LOGE(TAG, "check the gaja_device_id / gaja_device_token secrets (retries throttled)");
  }
}

// ---- adaptive volume ---------------------------------------------------------------------------

void GajaDevice::note_level_frame_(const uint8_t *frame, size_t len) {
  const size_t samples = len / 2;
  if (samples == 0) {
    return;
  }
  const int16_t *pcm = (const int16_t *) (const void *) frame;
  int64_t sum = 0;
  for (size_t i = 0; i < samples; i++) {
    sum += (int64_t) pcm[i] * pcm[i];
  }
  const float rms = sqrtf((float) (sum / (int64_t) samples));
  float db = 20.0f * log10f(rms / 32768.0f + 1e-12f);
  int bin = (int) ceilf(db) - LEVEL_DB_MIN;
  if (bin < 0) {
    bin = 0;
  } else if (bin >= LEVEL_DB_BINS) {
    bin = LEVEL_DB_BINS - 1;
  }
  this->level_hist_[bin]++;
  this->level_frames_++;
}

float GajaDevice::percentile_level_db_(float fraction) const {
  if (this->level_frames_ == 0) {
    return NAN;
  }
  const float target = fraction * this->level_frames_;
  uint32_t acc = 0;
  for (int bin = 0; bin < LEVEL_DB_BINS; bin++) {
    acc += this->level_hist_[bin];
    if ((float) acc >= target) {
      return bin + LEVEL_DB_MIN;
    }
  }
  return LEVEL_DB_MAX;
}

void GajaDevice::apply_reply_volume_() {
  const float db = this->percentile_level_db_(0.90f);
  float volume;
  if (std::isnan(db)) {
    volume = this->vol_max_;  // nothing was streamed (e.g. a timer announcement): be clearly audible
  } else {
    volume = map_db_to_volume(db, this->quiet_db_, this->loud_db_, this->vol_min_, this->vol_max_);
  }
  this->reply_volume_ = volume;
  this->reply_volume_set_ = true;
  this->speaker_->set_volume(volume * SPEAKER_VOLUME_FULL_SCALE);
  ESP_LOGI(TAG, "speech level %.1f dBFS (p90 of %u frames) -> reply volume %.2f",
           std::isnan(db) ? 0.0f : db, this->level_frames_, volume);
}

// ---- misc ----------------------------------------------------------------------------------------

bool GajaDevice::mic_stopped_() const { return this->mic_ == nullptr || this->mic_->is_stopped(); }

bool GajaDevice::begin_manual_playback() {
  if (!this->is_idle()) {
    return false;
  }
  // The gaja_device microphone source holds the I2S bus whenever it is not playing; the wake word
  // component holds it too while listening. Release both so the manual sound can play.
  this->mww_->stop();
  this->mic_source_->stop();
  return true;
}

void GajaDevice::end_manual_playback() {
  if (this->state_.load(std::memory_order_relaxed) == GajaState::IDLE) {
    this->enter_idle_();  // mic and wake word back on
  }
}

}  // namespace esphome::gaja_device

#endif  // USE_ESP32
