#include "gaja_device.h"

#ifdef USE_ESP32

#include "esphome/components/audio/audio.h"
#include "esphome/components/json/json_util.h"
#include "esphome/core/application.h"
#include "esphome/core/hal.h"
#include "esphome/core/log.h"
#include "preroll.h"
#include "volume_map.h"

#ifdef USE_NETWORK
#include "esphome/components/network/util.h"
#endif

#include <esp_heap_caps.h>
#include <esp_websocket_client.h>
#include <freertos/task.h>

#include <cmath>
#include <cstring>
#include <inttypes.h>

namespace esphome::gaja_device {

static const char *const TAG = "gaja_device";

// Uplink audio: PCM16 LE mono 16 kHz in 20 ms frames (docs/DEVICE.md recommends 20 ms / 640 B).
static constexpr uint32_t MIC_RATE = 16000;
static constexpr size_t FRAME_BYTES = MIC_RATE * 2 / 50;

// Buffer sizing (PSRAM unless noted). The Atom VoiceS3R has 8 MB octal PSRAM.
// The pre-roll ring is the single uplink source, so it also absorbs loop stalls while streaming:
// it holds preroll_ms plus slack for the burst send and network jitter.
static constexpr uint32_t PREROLL_SLACK_MS = 2500;
// The server sends reply audio as fast as TTS produces it (no pacing), so the playback ring must
// hold a whole multi-sentence reply (~44 s at 48 kHz); the WS event handler also applies
// backpressure when even this fills up.
static constexpr size_t PLAYBACK_RING_BYTES = 4 * 1024 * 1024;

// Timing constants, in milliseconds unless stated otherwise.
static constexpr uint32_t AMP_SETTLE_MS = 50;        // NS4150B enable -> first sample (verified)
static constexpr uint32_t AMP_OFF_DELAY_MS = 100;    // I2S buffer drains after the last sample
static constexpr uint32_t DRAIN_GRACE_MS = 400;      // audio_end -> finish(): wait for a next sentence
static constexpr uint32_t RECONNECT_BASE_MS = 2000;
static constexpr uint32_t RECONNECT_MAX_MS = 60000;
static constexpr uint32_t AUTH_FAIL_MIN_DELAY_MS = 30000;
static constexpr uint32_t HELLO_TIMEOUT_MS = 6000;   // server must answer ready within ~5 s
static constexpr uint32_t WAIT_REPLY_TIMEOUT_MS = 5000;  // listen_end without a following reply
static constexpr uint32_t MIC_STOP_TIMEOUT_MS = 5000;
static constexpr uint32_t SEND_TIMEOUT_TICKS = 100;  // FreeRTOS ticks for a single uplink frame
static constexpr uint32_t MIC_BYTES_PER_MS = MIC_RATE * 2 / 1000;
// The i2s_audio mic task reads 16 ms chunks (512 B); two of them keep the pre-roll start cursor
// clear of the bytes the producer is about to overwrite while the burst is in flight.
static constexpr uint32_t PREROLL_TAIL_MARGIN_BYTES = 1024;
// After a mic (re)start the I2S/codec transient is loud but is not speech: frames from the first
// MIC_RESTART_SUPPRESS_MS of a window are streamed but not counted by the level statistics.
static constexpr uint32_t MIC_RESTART_SUPPRESS_MS = 200;
// Watchdogs (M5): a lost audio_end must not keep the amplifier on forever, a wedged speaker task
// must not stall the device, and an open mic window without a server verdict must end sometime.
static constexpr uint32_t WAKING_TIMEOUT_MS = 10000;     // pre-roll burst should take << 2 s
static constexpr uint32_t STREAMING_TIMEOUT_MS = 40000;  // no listen_end/wake verdict at all
// While gated speech keeps flowing the streaming deadline re-arms (like the follow-up window),
// but no window outlives the hard cap.
static constexpr uint32_t STREAMING_REARM_MS = 4000;
static constexpr uint32_t STREAMING_HARD_CAP_MS = 60000;
static constexpr uint32_t PLAY_STALL_TIMEOUT_MS = 8000;  // no audio_end and no speaker progress
// Backpressure (B2): how long the WS event handler waits for playback ring space before dropping.
static constexpr uint32_t PLAY_BACKPRESSURE_MS = 200;
static constexpr uint32_t PLAY_BACKPRESSURE_STEP_MS = 20;
// A send that fails while the client still reports connected is "busy" (lock contention / a
// transient write stall): the ring keeps buffering; only a streak this long gives up the turn.
static constexpr uint16_t SEND_BUSY_LIMIT = 25;
// The follow-up window stays open this long after the server's deadline (and is re-armed while
// gated speech frames keep arriving, so long follow-up utterances are not cut off).
static constexpr uint32_t FOLLOWUP_GRACE_MS = 4000;

/// Wrap-safe millis() deadline check (plain >= breaks when millis() wraps to 0).
static inline bool deadline_passed(uint32_t now, uint32_t deadline) {
  return (int32_t) (now - deadline) >= 0;
}

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

// ---- setup / config ----------------------------------------------------------------------

static uint8_t *alloc_psram(size_t size) {
  uint8_t *ptr = (uint8_t *) heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (ptr == nullptr) {
    // Fall back to internal RAM rather than failing the whole component.
    ptr = (uint8_t *) heap_caps_malloc(size, MALLOC_CAP_8BIT);
  }
  return ptr;
}

void GajaDevice::setup() {
  if (this->out_rate_ < 8000 || this->out_rate_ > 48000) {
    ESP_LOGE(TAG, "out_rate %" PRIu32 " Hz is outside the protocol range 8000..48000", this->out_rate_);
    this->mark_failed();
    return;
  }

  // The i2s_audio speaker's audio stream info defaults to 16 bit/mono/16 kHz and the YAML
  // sample_rate does NOT set it (only media_player/mixer/resampler pipelines do). The I2S clock
  // is taken from the stream info, so assert ours now and again before every playback start.
  this->speaker_->set_audio_stream_info(audio::AudioStreamInfo(16, 1, this->out_rate_));

  // Power-of-two ring capacity: cheap modulo and clean behavior when the 32-bit position
  // counters wrap (rings.h). The ring is ALWAYS attached, even for a tiny preroll_ms: it is
  // also the live-streaming buffer, so preroll_ms=0 would otherwise mean "no uplink at all".
  const size_t preroll_needed = (size_t) (this->preroll_ms_ + PREROLL_SLACK_MS) * MIC_BYTES_PER_MS;
  size_t preroll_capacity = 1u << 12;
  while (preroll_capacity < preroll_needed) {
    preroll_capacity <<= 1;
  }
  uint8_t *preroll_buf = alloc_psram(preroll_capacity);
  uint8_t *playback_buf = alloc_psram(PLAYBACK_RING_BYTES);
  if (preroll_buf == nullptr || playback_buf == nullptr) {
    ESP_LOGE(TAG, "failed to allocate audio rings (preroll %u, playback %u)", (unsigned) preroll_capacity,
             (unsigned) PLAYBACK_RING_BYTES);
    this->mark_failed();
    return;
  }
  this->preroll_ring_.attach(preroll_buf, preroll_capacity);
  this->preroll_ring_capacity_ = preroll_capacity;
  this->playback_ring_.attach(playback_buf, PLAYBACK_RING_BYTES);

  this->frame_ = (uint8_t *) heap_caps_malloc(FRAME_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  this->feed_ = (uint8_t *) heap_caps_malloc(this->out_rate_ / 5, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (this->frame_ == nullptr || this->feed_ == nullptr) {
    ESP_LOGE(TAG, "failed to allocate frame buffers");
    this->mark_failed();
    return;
  }

  this->text_queue_ = xQueueCreate(TEXT_QUEUE_LEN, sizeof(TextMessage));
  if (this->text_queue_ == nullptr) {
    ESP_LOGE(TAG, "failed to create the server message queue");
    this->mark_failed();
    return;
  }

  // Tap the microphone stream. This source is active (it holds the microphone on whenever the
  // device is not playing), so the mic never restarts between the wake detection and streaming.
  this->mic_source_->add_data_callback([this](const std::vector<uint8_t> &data) { this->on_mic_data_(data); });

  esp_websocket_client_config_t cfg = {};
  cfg.uri = this->server_url_.c_str();  // points at our member, valid for the lifetime of the client
  cfg.buffer_size = 8192;               // server chunks are 100 ms @ 48 kHz = 9600 B -> 2 fragments
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
  ESP_LOGCONFIG(TAG, "  rings: preroll %u B, playback %u B", (unsigned) this->preroll_ring_capacity_,
                (unsigned) PLAYBACK_RING_BYTES);
  ESP_LOGCONFIG(TAG, "  wake: word=%s score=%.2f", this->wake_word_.c_str(), this->wake_score_);
  ESP_LOGCONFIG(TAG, "  adaptive volume: quiet_db=%.1f loud_db=%.1f vol_min=%.2f vol_max=%.2f", this->quiet_db_,
                this->loud_db_, this->vol_min_, this->vol_max_);
}

// ---- main loop ----------------------------------------------------------------------------

void GajaDevice::loop() {
  const uint32_t now = millis();

  this->update_mic_power_();
  this->process_text_queue_();
  this->service_reconnect_();

  switch (this->state_) {
    case GajaState::RECONNECTING:
      // Consumer-side clear: drop any stale reply audio from the dead connection. The WS task
      // may still push briefly; clear() only moves our own tail, so it stays safe.
      this->playback_ring_.clear();
      break;
    case GajaState::IDLE:
      // Nothing to do per iteration: the wake word component detects, the mic task fills the ring.
      // Unsolicited reply audio (a timer/reminder announcement) can arrive while idle.
      if (this->playback_ring_.available() > 0) {
        this->begin_playback_transition_();
      }
      break;
    case GajaState::WAKING:
      this->service_uplink_(32);
      if (this->state_ != GajaState::WAKING) {
        break;  // the send failed and reconnected mid-burst
      }
      if (!ring_pos_before(this->stream_pos_, this->wake_head_)) {
        this->state_ = GajaState::STREAMING;
        this->state_entered_ms_ = now;
        this->streaming_deadline_ms_ = now + STREAMING_TIMEOUT_MS;
        ESP_LOGI(TAG, "pre-roll (%" PRIu32 " ms) sent, streaming live audio", this->preroll_sent_ms_);
      } else if (now - this->state_entered_ms_ > WAKING_TIMEOUT_MS) {
        ESP_LOGW(TAG, "pre-roll burst stalled; aborting the wake");
        this->close_mic_stream_();
        this->enter_idle_();
      }
      break;
    case GajaState::STREAMING: {
      const uint32_t speech_before = this->levels_.speech_frames();
      this->service_uplink_(8);
      if (this->state_ != GajaState::STREAMING) {
        break;  // reconnected mid-stream
      }
      if (this->levels_.speech_frames() > speech_before) {
        // Real speech is flowing: the server's endpointer keeps its window open, ours must too
        // (the absolute hard cap below still applies).
        this->streaming_deadline_ms_ = now + STREAMING_REARM_MS;
      }
      if (deadline_passed(now, this->streaming_deadline_ms_) ||
          now - this->state_entered_ms_ > STREAMING_HARD_CAP_MS) {
        ESP_LOGW(TAG, "no listen_end while streaming (deadline); closing the mic window");
        this->close_mic_stream_();
        this->enter_idle_();
      }
      break;
    }
    case GajaState::FOLLOWUP: {
      const uint32_t speech_before = this->levels_.speech_frames();
      this->service_uplink_(8);
      if (this->state_ != GajaState::FOLLOWUP) {
        break;  // reconnected mid-window
      }
      if (this->levels_.speech_frames() > speech_before) {
        // Real speech is flowing: the server's endpointer keeps its window open, ours must too.
        this->followup_deadline_ms_ = now + FOLLOWUP_GRACE_MS;
      }
      if (deadline_passed(now, this->followup_deadline_ms_)) {
        ESP_LOGW(TAG, "follow-up window of %" PRIu32 " ms expired without listen_end; returning to idle",
                 this->follow_up_ms_);
        this->close_mic_stream_();
        this->enter_idle_();
      }
      break;
    }
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
          this->drained_since_ms_ = 0;
          this->play_stalled_since_ms_ = 0;
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
      } else if (deadline_passed(now, this->linger_deadline_ms_)) {
        this->enter_idle_();
      }
      break;
  }

  // Warn exactly once per new drop (the ring counter is cumulative for the whole reply).
  const uint32_t dropped = this->playback_ring_.dropped();
  if (dropped > this->play_dropped_seen_) {
    ESP_LOGW(TAG, "%" PRIu32 " downlink chunks dropped so far this reply (audio clipped)",
             dropped - this->play_dropped_seen_);
    this->play_dropped_seen_ = dropped;
  }
}

// ---- mic / wake word power ------------------------------------------------------------------

void GajaDevice::update_mic_power_() {
  // Microphone: never stop before the driver actually started (a stop while STATE_STOPPED loses
  // one of the counting-semaphore "listener" counts and the mic would run forever, holding the
  // shared I2S bus); never start before it fully stopped (the source's enabled_ guard makes
  // double-starts harmless).
  if (this->mic_wanted_ && !this->mic_requested_) {
    if (this->mic_->is_stopped()) {
      // Audio pushed from now on is fresh: the previous turn's tail must never become pre-roll.
      this->preroll_valid_from_ = this->preroll_ring_.head_pos();
      // The I2S/codec restart transient is loud but is not speech: frames from the first
      // MIC_RESTART_SUPPRESS_MS of this window are streamed but not counted as speech (M3).
      this->level_skip_until_pos_ =
          this->preroll_valid_from_ + MIC_RESTART_SUPPRESS_MS * MIC_BYTES_PER_MS;
      this->mic_source_->start();
      this->mic_requested_ = true;
    }
  } else if (!this->mic_wanted_ && this->mic_requested_) {
    if (!this->mic_->is_stopped()) {
      this->mic_source_->stop();
      this->mic_requested_ = false;
    }
  }
  // Wake word: start() is ignored unless the component is fully STOPPED, and stop() only sets a
  // pending flag -- so keep retrying both until the observed state matches the wanted state, but
  // call stop() only once per stop cycle (it spams the log on every loop pass otherwise).
  if (this->mww_->is_failed()) {
    this->mww_stop_requested_ = false;
  } else {
    const bool mww_running = this->mww_->is_running();
    if (this->mww_wanted_ && !mww_running) {
      this->mww_->start();
      this->mww_stop_requested_ = false;
    } else if (!this->mww_wanted_ && mww_running && !this->mww_stop_requested_) {
      this->mww_->stop();
      this->mww_stop_requested_ = true;
    } else if (!this->mww_wanted_ && !mww_running) {
      this->mww_stop_requested_ = false;  // fully stopped: the next stop cycle may request again
    }
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
  this->mww_wanted_ = false;  // micro_wake_word stops itself (stop_after_detection); make it explicit

  // Single-ring snapshot (M6/B-1): the pre-roll starts at the newest of (head - preroll_ms),
  // (tail + one mic chunk of margin) and (the last mic restart) -- see preroll.h for why each
  // bound matters. Bytes arriving after this instant are picked up seamlessly by the send
  // cursor, so nothing is lost between the snapshot and the WAKING state.
  const ring_pos_t head = this->preroll_ring_.head_pos();
  uint32_t preroll_bytes = 0;
  this->stream_pos_ =
      preroll_start_pos(head, this->preroll_ring_.tail_pos(), this->preroll_valid_from_,
                        this->preroll_ms_ * MIC_BYTES_PER_MS, PREROLL_TAIL_MARGIN_BYTES, &preroll_bytes);
  this->wake_head_ = head;
  this->frame_len_ = 0;
  const uint32_t preroll_ms = preroll_bytes / MIC_BYTES_PER_MS;
  this->preroll_sent_ms_ = preroll_ms;

  // The level statistics describe this wake's utterance (pre-roll + live), nothing older.
  this->levels_.clear();
  this->reply_volume_set_ = false;

  ESP_LOGI(TAG, "wake word '%s' -> sending %" PRIu32 " ms of pre-roll", word.c_str(), preroll_ms);
  this->send_json_wake_(preroll_ms);  // on failure it reconnects; the wake is simply lost
  if (this->state_ != GajaState::IDLE) {
    return;  // send_json_wake_ failed and moved us to RECONNECTING
  }
  this->state_ = GajaState::WAKING;
  this->state_entered_ms_ = millis();
}

void GajaDevice::send_json_wake_(uint32_t preroll_ms) {
  auto payload = json::build_json([this, preroll_ms](JsonObject root) {
    root["type"] = "wake";
    root["word"] = this->wake_word_;
    root["score"] = this->wake_score_;
    root["preroll_ms"] = preroll_ms;
  });
  this->send_text_(payload.data(), payload.size());
}

void GajaDevice::service_uplink_(uint32_t max_frames) {
  const ring_pos_t head = this->preroll_ring_.head_pos();
  if (ring_pos_before(this->stream_pos_, this->preroll_ring_.tail_pos())) {
    // The producer lapped us (loop stall): skip the overwritten gap, keep the stream aligned.
    ESP_LOGW(TAG, "uplink ring lapped; skipping %" PRIu32 " ms of audio",
             (uint32_t) ((uint32_t) ring_pos_diff(this->preroll_ring_.tail_pos(), this->stream_pos_) /
                         MIC_BYTES_PER_MS));
    this->stream_pos_ = this->preroll_ring_.tail_pos();
    this->frame_len_ = 0;
  }
  uint32_t sent = 0;
  while (sent < max_frames) {
    // Top up the partial frame from the ring.
    const size_t want = std::min(FRAME_BYTES - this->frame_len_, (size_t) (head - this->stream_pos_));
    if (want > 0) {
      const size_t got = this->preroll_ring_.peek_abs(this->stream_pos_, this->frame_ + this->frame_len_, want);
      if (got == 0) {  // overwritten mid-frame (should not happen at this depth)
        this->frame_len_ = 0;
        this->stream_pos_ = this->preroll_ring_.tail_pos();
        continue;
      }
      this->stream_pos_ += got;
      this->frame_len_ += got;
      if (this->frame_len_ == FRAME_BYTES) {
        // Count the level exactly once, when the frame is fully assembled (never on send
        // retries). Frames inside the mic-restart transient window do not count: the I2S/amp
        // restart click is loud but is not speech.
        const ring_pos_t frame_start = this->stream_pos_ - FRAME_BYTES;
        if (!ring_pos_before(frame_start, this->level_skip_until_pos_)) {
          this->levels_.note_frame(this->frame_, FRAME_BYTES);
        }
      }
    }
    if (this->frame_len_ < FRAME_BYTES) {
      break;  // nothing new; wait for the mic task
    }
    if (!this->send_frame_(FRAME_BYTES)) {
      return;  // frame stays assembled; retried on the next loop pass
    }
    this->frame_len_ = 0;
    sent++;
  }
}

bool GajaDevice::send_frame_(size_t len) {
  esp_websocket_client_handle_t client = (esp_websocket_client_handle_t) this->ws_client_;
  const int rc = esp_websocket_client_send_bin(client, (const char *) this->frame_, len,
                                               pdMS_TO_TICKS(SEND_TIMEOUT_TICKS));
  if (rc < 0) {
    if (!this->ws_connected_.load()) {
      // The client already dispatched DISCONNECTED (a failed write aborts the connection and the
      // event fires before send returns): the link is really gone, reconnect now.
      this->send_failures_ = 0;
      this->enter_reconnecting_("uplink send failed (disconnected)");
      return false;
    }
    // Still connected: treat the failure as "busy" (TX lock contention / a transient write
    // stall). The ring keeps buffering the audio; only a long streak gives up the turn.
    if (++this->send_failures_ >= SEND_BUSY_LIMIT) {
      ESP_LOGW(TAG, "uplink send failed %" PRIu16 " times in a row; reconnecting", this->send_failures_);
      this->send_failures_ = 0;
      this->enter_reconnecting_("uplink send stuck");
    }
    return false;
  }
  this->send_failures_ = 0;
  return true;
}

void GajaDevice::close_mic_stream_() {
  // Send what is still queued (bounded), then flush a partial frame zero-padded like the
  // reference simulator does; count only the real bytes towards the level statistics.
  this->service_uplink_(32);
  if (this->frame_len_ > 0 && this->ws_connected_) {
    const ring_pos_t frame_start = this->stream_pos_ - this->frame_len_;
    if (!ring_pos_before(frame_start, this->level_skip_until_pos_)) {
      this->levels_.note_frame(this->frame_, this->frame_len_ & ~(size_t) 1);
    }
    memset(this->frame_ + this->frame_len_, 0, FRAME_BYTES - this->frame_len_);
    this->send_frame_(FRAME_BYTES);
  }
  this->frame_len_ = 0;
  this->stream_pos_ = this->preroll_ring_.head_pos();
}

// ---- microphone callback (runs on the microphone task) --------------------------------------

void GajaDevice::on_mic_data_(const std::vector<uint8_t> &data) {
  if (data.empty()) {
    return;
  }
  // The rolling ring is the single uplink source: it always captures (pre-roll for the next wake
  // while idle; the live stream while WAKING/STREAMING/FOLLOWUP -- the send cursor decides).
  this->preroll_ring_.push(data.data(), data.size());
}

// ---- downlink: reply playback ----------------------------------------------------------------

void GajaDevice::begin_playback_transition_() {
  const GajaState state = this->state_;
  if (state == GajaState::WAKING || state == GajaState::STREAMING || state == GajaState::FOLLOWUP) {
    this->close_mic_stream_();
  }
  // Half-duplex I2S bus: the microphone must fully stop before the speaker may start.
  this->mic_wanted_ = false;
  this->mww_wanted_ = false;
  // Re-assert the stream info: a media_player beep (the manual test) may have changed it, and the
  // I2S clock is derived from it (B1).
  this->speaker_->set_audio_stream_info(audio::AudioStreamInfo(16, 1, this->out_rate_));
  this->speaker_finishing_ = false;
  this->drained_since_ms_ = 0;
  this->play_stalled_since_ms_ = 0;
  this->amp_off_deadline_ms_ = 0;
  this->play_dropped_seen_ = this->playback_ring_.dropped();  // warn only about this reply's drops
  this->state_ = GajaState::PREPARE_PLAY;
  this->state_entered_ms_ = millis();
}

void GajaDevice::service_speaker_() {
  const uint32_t now = millis();
  const size_t ring_available = this->playback_ring_.available();
  const bool speaker_buffered = this->speaker_->has_buffered_data();
  const bool drained = this->audio_end_seen_ && ring_available == 0 && !speaker_buffered;

  if (!drained) {
    this->drained_since_ms_ = 0;
    if (this->speaker_finishing_) {
      // New audio arrived after finish() began: wait for the speaker to stop, then resume.
      if (this->speaker_->is_stopped()) {
        this->speaker_finishing_ = false;
        this->amp_off_deadline_ms_ = 0;
      }
      return;
    }
    const size_t feed_capacity = this->out_rate_ / 5;  // 100 ms of PCM16 mono
    bool moved = false;
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
      moved = true;
      if (written < n) {
        break;
      }
    }
    // Watchdogs (M5): either the audio_end was lost (ring empty, speaker idle, nothing new) or
    // the speaker task is wedged (ring has data, play() accepts nothing). Both end the reply.
    if (moved || speaker_buffered) {
      this->play_stalled_since_ms_ = 0;
    } else if (this->play_stalled_since_ms_ == 0) {
      this->play_stalled_since_ms_ = now;
    } else if (now - this->play_stalled_since_ms_ > PLAY_STALL_TIMEOUT_MS) {
      ESP_LOGE(TAG, "playback stalled for %" PRIu32 " ms (audio_end lost or speaker wedged); ending the reply",
               now - this->play_stalled_since_ms_);
      if (!this->speaker_->is_stopped()) {
        this->speaker_->stop();
      }
      // Drop the wedged reply's leftovers so a later audio_start cannot re-enter PREPARE_PLAY on
      // data nobody can drain (M8).
      this->playback_ring_.clear();
      this->finish_reply_();
      return;
    }
    return;
  }

  if (!this->speaker_finishing_) {
    if (this->drained_since_ms_ == 0) {
      this->drained_since_ms_ = now;
      return;
    }
    if (now - this->drained_since_ms_ < DRAIN_GRACE_MS) {
      return;  // inter-sentence grace: the next utterance of this reply may still arrive
    }
    this->speaker_->finish();  // stop gracefully once everything buffered has been played
    this->speaker_finishing_ = true;
    return;
  }
  if (this->speaker_->is_stopped()) {
    if (this->amp_off_deadline_ms_ == 0) {
      this->amp_off_deadline_ms_ = now + AMP_OFF_DELAY_MS;
    } else if (deadline_passed(now, this->amp_off_deadline_ms_)) {
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
  this->drained_since_ms_ = 0;
  this->play_stalled_since_ms_ = 0;
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
  this->levels_.clear();            // only this window's speech counts (M4)
  // A fresh cursor (like on_wake): the previous turn's cursor is stale and would stream up to
  // four seconds of old audio from IDLE.
  this->frame_len_ = 0;
  this->stream_pos_ = this->preroll_ring_.head_pos();
  this->mww_wanted_ = false;        // no wake phrase needed while the window is open
  this->mic_wanted_ = true;         // mic back on (the speaker has released the bus)
  this->state_ = GajaState::FOLLOWUP;
  this->state_entered_ms_ = millis();
  this->followup_deadline_ms_ = millis() + this->follow_up_ms_ + FOLLOWUP_GRACE_MS;
  ESP_LOGI(TAG, "follow-up window open for %" PRIu32 " ms (no wake phrase needed)", this->follow_up_ms_);
}

// ---- state helpers ----------------------------------------------------------------------------

void GajaDevice::enter_idle_() {
  this->frame_len_ = 0;
  this->stream_pos_ = this->preroll_ring_.head_pos();
  this->follow_up_received_ = false;
  this->audio_end_seen_ = false;
  this->reply_volume_set_ = false;
  this->send_failures_ = 0;    // a new turn gets a fresh "busy" budget (M-2)
  this->levels_.clear();  // an unsolicited announcement (no mic window) must use vol_max
  this->amp_off_deadline_ms_ = 0;
  this->mic_wanted_ = true;
  this->mww_wanted_ = true;
  this->state_ = GajaState::IDLE;
  this->state_entered_ms_ = millis();
}

void GajaDevice::enter_reconnecting_(const char *reason) {
  const uint32_t now = millis();
  const bool already = this->state_ == GajaState::RECONNECTING;
  if (already) {
    // A repeated trigger (e.g. a send racing the teardown, or a hello timeout re-firing) must
    // not flood the log -- but it still falls through to the flags/stop/reschedule below.
    if (now - this->last_reconnect_log_ms_ > 5000) {
      ESP_LOGW(TAG, "connection still down (%s)", reason);
      this->last_reconnect_log_ms_ = now;
    }
  } else {
    ESP_LOGW(TAG, "connection lost (%s) in state %s", reason, this->state_str_());
    this->last_reconnect_log_ms_ = now;
    this->state_ = GajaState::RECONNECTING;
    this->state_entered_ms_ = now;
    this->mic_wanted_ = false;
    this->mww_wanted_ = false;
    if (!this->speaker_->is_stopped()) {
      this->speaker_->stop();
    }
    if (this->amp_on_) {
      this->speaker_enable_->turn_off();
      this->amp_on_ = false;
    }
    this->speaker_finishing_ = false;
    this->amp_off_deadline_ms_ = 0;
    this->drained_since_ms_ = 0;
    this->play_stalled_since_ms_ = 0;
    this->frame_len_ = 0;
    this->stream_pos_ = this->preroll_ring_.head_pos();
    this->follow_up_received_ = false;
    this->audio_end_seen_ = false;
    // No ring resets from here: the playback ring is cleared by its consumer (the RECONNECTING
    // loop) and the pre-roll ring is invalidated by preroll_valid_from_ at the next mic start.
  }
  this->send_failures_ = 0;
  // esp_websocket_client_stop() dispatches ONLY WEBSOCKET_EVENT_FINISH -- never DISCONNECTED --
  // so the connection flags must be cleared here, ALWAYS (even when the early return used to
  // skip them), or the retry gate in service_reconnect_ (!ws_connected_) never opens again and
  // the device stays deaf until power-cycled.
  this->ws_connected_.store(false);
  this->hello_sent_ = false;
  this->server_ready_ = false;
  if (this->ws_client_ != nullptr && this->ws_task_running_.load()) {
    this->ws_task_running_.store(false);
    // Bounded: the WS task notices run=false within one poll/read timeout, and its backpressure
    // wait loop also aborts as soon as ws_connected_ goes false.
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
    if (esp_websocket_client_start((esp_websocket_client_handle_t) this->ws_client_) == ESP_OK) {
      this->ws_task_running_.store(true);
    }
    this->schedule_next_attempt_();  // arm the retry in case this attempt fails
    return;
  }

  if (!this->ws_connected_.load() && this->state_ != GajaState::RECONNECTING) {
    this->enter_reconnecting_("websocket closed");
    return;
  }

  // Retry only when the connection is actually down (M9): awaiting a slow first `ready` (the
  // server builds VAD/TTS) is not a failed attempt -- the hello timeout below covers it.
  if (this->reconnect_pending_ && !this->ws_connected_.load() && deadline_passed(now, this->next_attempt_ms_)) {
    if (!network_up) {
      this->next_attempt_ms_ = now + 1000;
      return;
    }
    this->reconnect_pending_ = false;
    if (this->ws_task_running_.load()) {
      this->ws_task_running_.store(false);
      esp_websocket_client_stop((esp_websocket_client_handle_t) this->ws_client_);
    }
    esp_err_t err = esp_websocket_client_start((esp_websocket_client_handle_t) this->ws_client_);
    if (err != ESP_OK) {
      ESP_LOGW(TAG, "websocket start failed: %s", esp_err_to_name(err));
    } else {
      this->ws_task_running_.store(true);
    }
    this->schedule_next_attempt_();
  }

  if (this->ws_connected_.load() && !this->hello_sent_) {
    this->send_hello_();
  }
  if (this->hello_sent_ && !this->server_ready_ && now - this->hello_sent_ms_ > HELLO_TIMEOUT_MS) {
    // Fires at most once per attempt: enter_reconnecting_ takes the connection down and arms
    // a backoff of >= RECONNECT_BASE_MS (repeated triggers are rate-limited there).
    ESP_LOGW(TAG, "server did not answer hello within %" PRIu32 " ms", HELLO_TIMEOUT_MS);
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
    rates.add(this->out_rate_);  // our configured reply rate; the server resamples TTS to it
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
    if (!this->ws_connected_.load()) {
      // The link is gone (DISCONNECTED already fired); take the connection down properly.
      this->enter_reconnecting_("text send failed (disconnected)");
      return false;
    }
    // Still connected: busy (TX lock contention). The caller retries on the next loop pass
    // (hello/pong) or the wake is simply lost -- reconnecting here would kill a live link.
    ESP_LOGW(TAG, "text send busy/failed (%.*s)", (int) std::min<size_t>(len, 64), data);
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
      // Reset the per-connection flags BEFORE publishing ws_connected_: the main loop sends its
      // hello as soon as it sees ws_connected_, and a stale hello_sent_ would double it.
      self->hello_sent_ = false;
      self->server_ready_ = false;
      self->ws_connected_.store(true);
      ESP_LOGI(TAG, "websocket connected");
      break;
    case WEBSOCKET_EVENT_DISCONNECTED:
    case WEBSOCKET_EVENT_CLOSED:
      self->ws_connected_.store(false);
      self->hello_sent_ = false;
      self->server_ready_ = false;
      self->dl_skip_leading_ = false;
      self->dl_dropped_bytes_ = 0;  // the stream restarts aligned on the next connection
      ESP_LOGW(TAG, "websocket %s", event_id == WEBSOCKET_EVENT_CLOSED ? "closed by server" : "disconnected");
      break;
    case WEBSOCKET_EVENT_FINISH:
      // The client task exited. esp_websocket_client_stop() fires ONLY this event (no
      // DISCONNECTED), so a FINISH must clear the connection flags or service_reconnect_'s
      // retry gate (!ws_connected_) never opens again.
      self->ws_connected_.store(false);
      self->hello_sent_ = false;
      self->server_ready_ = false;
      self->ws_task_running_.store(false);
      self->dl_skip_leading_ = false;
      self->dl_dropped_bytes_ = 0;
      ESP_LOGI(TAG, "websocket task finished");
      break;
    case WEBSOCKET_EVENT_ERROR:
      ESP_LOGW(TAG, "websocket transport error");
      break;
    case WEBSOCKET_EVENT_DATA: {
      esp_websocket_event_data_t *d = (esp_websocket_event_data_t *) event_data;
      if (d->data_len <= 0) {
        break;
      }
      if (d->op_code == 0x02) {  // binary: reply PCM (fragmentation tolerated)
        // Fragments have ARBITRARY lengths, so sample alignment must be tracked explicitly:
        // when a drop leaves an odd number of bytes missing, the next fragment gives its first
        // byte back, or every later sample would be byte-swapped. The full accounting runs on
        // bytes since the connection's audio began (a fresh connection starts aligned).
        const uint8_t *ptr = (const uint8_t *) d->data_ptr;
        size_t len = (size_t) d->data_len;
        size_t dropped = 0;
        if (self->dl_skip_leading_ && len > 0) {
          self->dl_skip_leading_ = false;
          ptr += 1;
          len -= 1;
          dropped += 1;
        }
        size_t pushed = 0;
        if (len > 0) {
          // The server sends whole sentences as fast as TTS produces them: apply backpressure
          // (blocking here closes the TCP window and slows the server) before dropping anything.
          bool ok = false;
          for (uint32_t waited = 0; waited < PLAY_BACKPRESSURE_MS && !ok; waited += PLAY_BACKPRESSURE_STEP_MS) {
            ok = self->playback_ring_.push(ptr, len);
            if (!ok) {
              if (!self->ws_connected_.load()) {
                break;  // teardown: never hold the ws task (and stop()) on a dead connection
              }
              vTaskDelay(pdMS_TO_TICKS(PLAY_BACKPRESSURE_STEP_MS));
            }
          }
          if (ok) {
            pushed = len;
          } else {
            // Last resort: drop the odd trailing byte (and only if even that cannot fit, the
            // whole fragment) -- the parity accounting above repairs the alignment.
            const size_t even = len & ~(size_t) 1;
            if (even > 0 && self->playback_ring_.push(ptr, even)) {
              pushed = even;
            }
          }
        }
        dropped += len - pushed;
        if (dropped > 0) {
          self->dl_dropped_bytes_ += (uint32_t) dropped;
          if ((self->dl_dropped_bytes_ & 1u) != 0) {
            self->dl_skip_leading_ = true;
          }
        }
        App.wake_loop_threadsafe();
      } else if (d->op_code == 0x01) {  // text JSON, accumulated per message
        if (d->payload_offset == 0) {
          self->text_scratch_.len = 0;
        }
        const size_t space = sizeof(TextMessage::buf) - 1 - self->text_scratch_.len;  // keep the NUL byte
        const size_t copy = std::min((size_t) d->data_len, space);
        memcpy(self->text_scratch_.buf + self->text_scratch_.len, d->data_ptr, copy);
        self->text_scratch_.len = (uint16_t) (self->text_scratch_.len + copy);
        if (copy < (size_t) d->data_len) {
          ESP_LOGW(TAG, "server message longer than %u bytes; truncated", (unsigned) sizeof(TextMessage::buf));
        }
        if (d->payload_offset + d->data_len >= d->payload_len) {  // message complete
          self->text_scratch_.buf[self->text_scratch_.len] = '\0';
          // Capture hello_sent_ NOW: a close event clears it before the main loop parses this
          // message, and the auth backoff must not be lost to that race.
          self->text_scratch_.hello_sent = self->hello_sent_;
          if (xQueueSend(self->text_queue_, &self->text_scratch_, 0) != pdTRUE) {
            // The loop drains the queue every pass, so a full queue means it stalled. Evict the
            // OLDEST message (per-turn order puts droppable transcripts/metrics there) instead
            // of silently dropping the NEW one -- control messages (listen_end, audio_end,
            // follow_up, error) are what the state machine still needs (M1).
            TextMessage evicted;
            if (xQueueReceive(self->text_queue_, &evicted, 0) == pdTRUE &&
                xQueueSend(self->text_queue_, &self->text_scratch_, 0) == pdTRUE) {
              ESP_LOGW(TAG, "server message queue full; evicted older '%.24s'", evicted.buf);
            } else {
              ESP_LOGE(TAG, "server message queue full; dropping '%.32s'", self->text_scratch_.buf);
            }
          }
          App.wake_loop_threadsafe();
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
  TextMessage msg;
  while (xQueueReceive(this->text_queue_, &msg, 0) == pdTRUE) {
    this->handle_server_json_(msg);
  }
}

void GajaDevice::handle_server_json_(const TextMessage &msg) {
  const bool parsed = json::parse_json(std::string(msg.buf, msg.len), [this, &msg](JsonObject root) -> bool {
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
      this->on_audio_end_(root["empty"] | false);
    } else if (strcmp(type, "ping") == 0) {
      this->on_ping_();
    } else if (strcmp(type, "error") == 0) {
      this->on_error_(root["message"] | "?", msg.hello_sent);
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
    ESP_LOGW(TAG, "malformed server message (%u bytes)", (unsigned) msg.len);
  }
}

void GajaDevice::on_ready_(int sample_rate) {
  this->server_ready_ = true;
  this->reconnect_pending_ = false;
  this->reconnect_failures_ = 0;
  this->auth_failed_ = false;
  if (sample_rate != 0 && (uint32_t) sample_rate != this->out_rate_) {
    ESP_LOGW(TAG, "negotiated out_rate %d differs from the configured rate %" PRIu32
                  "; audio will play at the wrong speed",
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
      ESP_LOGI(TAG, "listen_end: stop streaming, wait for the reply");
      this->close_mic_stream_();
      this->mic_wanted_ = false;  // free the shared I2S bus for the speaker
      this->state_ = GajaState::WAIT_REPLY;
      this->state_entered_ms_ = millis();
      break;
    case GajaState::FOLLOWUP:
      // Two meanings (M2): the user's follow-up utterance ended (speech was streamed -- a reply
      // is coming) or the server's follow-up window expired with nobody speaking (no reply).
      if (this->levels_.speech_seen()) {
        ESP_LOGI(TAG, "listen_end after follow-up speech: wait for the reply");
        this->close_mic_stream_();
        this->mic_wanted_ = false;
        this->state_ = GajaState::WAIT_REPLY;
        this->state_entered_ms_ = millis();
      } else {
        ESP_LOGI(TAG, "listen_end with no speech in the follow-up window; back to wake word");
        this->close_mic_stream_();
        this->enter_idle_();
      }
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
  this->drained_since_ms_ = 0;
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

void GajaDevice::on_audio_end_(bool empty) {
  if (empty) {
    // The server sends audio_end{empty:true} INSTEAD of audio for an utterance with no TTS
    // output: nothing will play for it, so it must not mark the reply as drained.
    ESP_LOGD(TAG, "audio_end (empty): utterance has no audio");
    if (this->state_ == GajaState::WAIT_REPLY && this->playback_ring_.available() == 0) {
      ESP_LOGI(TAG, "reply had no audio; back to idle");
      this->enter_idle_();
    }
    return;
  }
  ESP_LOGD(TAG, "audio_end; draining speaker");
  this->audio_end_seen_ = true;
}

void GajaDevice::on_ping_() { this->pending_pong_ = true; }

void GajaDevice::on_error_(const std::string &message, bool hello_was_sent) {
  ESP_LOGE(TAG, "server error: %s", message.c_str());
  // Only an answer to a hello WE sent on THIS connection proves bad credentials; a stale
  // "unauthorized" from a previous connection must not trigger the 30 s auth backoff. The flag
  // was captured when the message was queued (the close event cleared hello_sent_ meanwhile).
  if (message == "unauthorized" && hello_was_sent) {
    this->auth_failed_ = true;
    ESP_LOGE(TAG, "check the gaja_device_id / gaja_device_token secrets (retries throttled)");
  }
}

// ---- adaptive volume ---------------------------------------------------------------------------

void GajaDevice::apply_reply_volume_() {
  const float db = this->levels_.speech_percentile_db(0.90f);
  float volume;
  if (std::isnan(db)) {
    // Nothing was streamed (e.g. a timer announcement): be clearly audible.
    volume = this->vol_max_;
    ESP_LOGI(TAG, "no speech level measured (announcement?) -> reply volume %.2f", volume);
  } else {
    volume = map_db_to_volume(db, this->quiet_db_, this->loud_db_, this->vol_min_, this->vol_max_);
    ESP_LOGI(TAG, "speech level %.1f dBFS (p90 of %" PRIu32 " speech frames, floor %.1f) -> reply volume %.2f",
             db, this->levels_.speech_frames(), this->levels_.floor_db(), volume);
  }
  this->reply_volume_ = volume;
  this->reply_volume_set_ = true;
  this->speaker_->set_volume(volume * SPEAKER_VOLUME_FULL_SCALE);
}

// ---- misc ----------------------------------------------------------------------------------------

bool GajaDevice::mic_stopped_() const {
  if (this->mic_ == nullptr) {
    return true;
  }
  if (this->mic_requested_) {
    // Our stop request is still in flight (or we still want the mic): the shared I2S bus has
    // not been released -- PREPARE_PLAY must wait, the mic task may be about to claim it.
    return false;
  }
  return this->mic_->is_stopped();
}

bool GajaDevice::begin_manual_playback() {
  if (!this->is_idle()) {
    return false;
  }
  // The gaja_device microphone source holds the I2S bus whenever it is not playing; the wake word
  // component holds it too while listening. Release both so the manual sound can play.
  this->mww_wanted_ = false;
  this->mic_wanted_ = false;
  // Our per-turn reply volume must not leak into the manual beep: the beep is the volume
  // reference (100% = full verified scale).
  this->speaker_->set_volume(SPEAKER_VOLUME_FULL_SCALE);
  return true;
}

void GajaDevice::end_manual_playback() {
  if (this->state_.load(std::memory_order_relaxed) == GajaState::IDLE) {
    this->enter_idle_();  // mic and wake word back on
  }
}

}  // namespace esphome::gaja_device

#endif  // USE_ESP32
