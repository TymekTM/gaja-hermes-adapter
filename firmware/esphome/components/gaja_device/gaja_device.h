#pragma once

#ifdef USE_ESP32

#include "esphome/components/microphone/microphone.h"
#include "esphome/components/microphone/microphone_source.h"
#include "esphome/components/micro_wake_word/micro_wake_word.h"
#include "esphome/components/speaker/speaker.h"
#include "esphome/components/switch/switch.h"
#include "esphome/core/component.h"

#include <atomic>
#include <cstdint>
#include <esp_event.h>
#include <freertos/FreeRTOS.h>
#include <string>
#include <vector>

namespace esphome::gaja_device {

// Speech-level histogram range for the adaptive reply volume (1 dB bins, dBFS of 20 ms frame RMS).
static constexpr int LEVEL_DB_MIN = -90;
static constexpr int LEVEL_DB_MAX = 0;
static constexpr int LEVEL_DB_BINS = LEVEL_DB_MAX - LEVEL_DB_MIN + 1;

// The Atom VoiceS3R's verified full-scale: media_player volume 1.0 * volume_max 0.75 = ES8311 DAC
// register 0xBF = 0 dB (the loudest level that does not clip). Volumes passed to
// speaker->set_volume() are scaled by this constant, so 1.0 here means "loudest verified".
static constexpr float SPEAKER_VOLUME_FULL_SCALE = 0.75f;

/// Single-producer/single-consumer byte ring buffer (drops new data when full).
/// Producer and consumer are on different FreeRTOS tasks; only the producer writes ``head_``,
/// only the consumer writes ``tail_`` (peek/advance split keeps the playback path lock-free).
class SpscRing {
 public:
  bool allocate(size_t capacity);
  void reset() {
    this->head_.store(0, std::memory_order_relaxed);
    this->tail_.store(0, std::memory_order_relaxed);
    this->dropped_ = 0;
  }
  /// Producer side. Returns false when the ring was full and the data was dropped.
  bool push(const uint8_t *data, size_t len);
  /// Consumer side: copy up to ``len`` bytes without consuming them.
  size_t peek(uint8_t *dst, size_t len) const;
  /// Consumer side: consume ``len`` bytes previously peeked.
  void advance(size_t len);
  size_t available() const { return this->head_.load(std::memory_order_acquire) - this->tail_.load(std::memory_order_relaxed); }
  size_t dropped() const { return this->dropped_; }

 protected:
  uint8_t *buffer_{nullptr};
  size_t capacity_{0};
  std::atomic<size_t> head_{0};  // write position (producer)
  std::atomic<size_t> tail_{0};  // read position (consumer)
  uint32_t dropped_{0};
};

/// Rolling pre-wake window: the microphone task keeps overwriting the oldest audio. Positions are
/// absolute counters, so a reader can pin a snapshot (tail/head pair taken at one instant) and peek
/// it while the producer keeps running; it only goes stale once the producer laps it (one full
/// window, 1.5 s by default -- far longer than the pre-roll burst takes to send).
class RollingRing {
 public:
  bool allocate(size_t capacity);
  /// Producer side; overwrites the oldest bytes when full.
  void push(const uint8_t *data, size_t len);
  /// Absolute position of the oldest retained byte ([tail_pos, head_pos) is the live window).
  size_t tail_pos() const;
  /// Absolute position one past the newest byte.
  size_t head_pos() const;
  /// Copy ``len`` bytes starting at absolute position ``abs`` (within the retained window).
  size_t peek_abs(size_t abs, uint8_t *dst, size_t len) const;

 protected:
  uint8_t *buffer_{nullptr};
  size_t capacity_{0};
  std::atomic<size_t> head_{0};  // absolute bytes ever written
};

enum class GajaState : uint8_t {
  RECONNECTING,  // no server connection; mic off, wake word stopped
  IDLE,          // connected; wake word listening, pre-roll ring filling
  WAKING,        // wake sent, pre-roll burst in flight
  STREAMING,     // live 20 ms frames after a wake
  FOLLOWUP,      // live frames in the follow-up window (no wake phrase needed)
  WAIT_REPLY,    // mic closed (listen_end); reply or follow-up may arrive
  PREPARE_PLAY,  // audio_start received: waiting for the mic/I2S bus, amplifier warm-up
  PLAYING,       // feeding the reply PCM to the speaker
  LINGER,        // reply drained; waiting a moment for another utterance of the same reply
};

const char *gaja_state_to_string(GajaState state);

/**
 * Gaja voice device client (protocol v1.9, docs/DEVICE.md) for the M5Stack Atom VoiceS3R.
 *
 * One I2S bus is shared by the microphone and the speaker, so the device is strictly
 * half-duplex: the mic is stopped before playback and restarted afterwards (caps.aec = false).
 * Wake detection runs in ESPHome's micro_wake_word component; this component owns everything
 * else: the WebSocket connection, the pre-roll ring buffer (PSRAM), live frame streaming,
 * reply playback with the NS4150B amplifier, and the adaptive reply volume.
 */
class GajaDevice : public Component {
 public:
  void setup() override;
  void loop() override;
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::AFTER_CONNECTION; }

  // Called from YAML (micro_wake_word.on_wake_word_detected -> lambda: id(<id>).on_wake(wake_word)).
  void on_wake(const std::string &word);

  // For YAML automations (e.g. only run the manual test beep when no turn is in flight).
  bool is_idle() const {
    GajaState state = this->state_.load(std::memory_order_relaxed);
    return state == GajaState::IDLE || state == GajaState::RECONNECTING;
  }

  // Manual playback from YAML (the button beep test): releases the microphone and the wake word
  // component so the shared I2S bus is free for the speaker. Returns false when a turn is active.
  // After the sound finished, call end_manual_playback().
  bool begin_manual_playback();
  void end_manual_playback();

  void set_server_url(const std::string &url) { this->server_url_ = url; }
  void set_device_id(const std::string &id) { this->device_id_ = id; }
  void set_token(const std::string &token) { this->token_ = token; }
  void set_firmware_version(const std::string &fw) { this->fw_version_ = fw; }
  void set_lang(const std::string &lang) { this->lang_ = lang; }
  void set_microphone(microphone::Microphone *mic) { this->mic_ = mic; }
  void set_microphone_source(microphone::MicrophoneSource *source) { this->mic_source_ = source; }
  void set_wake_word(micro_wake_word::MicroWakeWord *mww) { this->mww_ = mww; }
  void set_speaker(speaker::Speaker *speaker) { this->speaker_ = speaker; }
  void set_speaker_enable(switch_::Switch *enable) { this->speaker_enable_ = enable; }
  void set_preroll_ms(uint32_t ms) { this->preroll_ms_ = ms; }
  void set_wake_score(float score) { this->wake_score_ = score; }
  void set_quiet_db(float db) { this->quiet_db_ = db; }
  void set_loud_db(float db) { this->loud_db_ = db; }
  void set_vol_min(float v) { this->vol_min_ = v; }
  void set_vol_max(float v) { this->vol_max_ = v; }
  void set_reply_gap_ms(uint32_t ms) { this->reply_gap_ms_ = ms; }

 protected:
  // ---- websocket plumbing ----------------------------------------------------------------
  static void websocket_event_handler_(void *arg, esp_event_base_t base, int32_t event_id, void *event_data);
  void schedule_next_attempt_();
  bool send_text_(const char *data, size_t len);
  bool send_frame_(size_t len);
  bool send_json_wake_(uint32_t preroll_ms);
  void send_hello_();

  // ---- server messages -------------------------------------------------------------------
  void handle_server_json_(const char *data, size_t len);
  void on_ready_(int sample_rate);
  void on_wake_accepted_();
  void on_wake_rejected_(const std::string &reason);
  void on_listen_end_();
  void on_follow_up_(uint32_t ms);
  void on_audio_start_(int sample_rate, int id);
  void on_audio_end_();
  void on_ping_();
  void on_error_(const std::string &message);

  // ---- state machine ---------------------------------------------------------------------
  void enter_reconnecting_(const char *reason);
  void enter_idle_();
  void begin_playback_transition_();  // from any streaming/idle state: mic off -> amplifier -> PLAYING
  void finish_reply_();               // drained: amplifier off, then follow-up or idle
  void open_followup_();
  void close_mic_stream_();

  // ---- audio paths -----------------------------------------------------------------------
  void on_mic_data_(const std::vector<uint8_t> &data);  // microphone task context
  void service_waking_();
  void service_streaming_();
  void service_speaker_();
  void service_reconnect_();

  // ---- adaptive volume -------------------------------------------------------------------
  void note_level_frame_(const uint8_t *frame, size_t len);
  float percentile_level_db_(float fraction) const;
  void apply_reply_volume_();

  // ---- misc ------------------------------------------------------------------------------
  void process_text_queue_();
  bool mic_stopped_() const;
  const char *state_str_() const { return gaja_state_to_string(this->state_); }

  std::string server_url_{};
  std::string device_id_{};
  std::string token_{};
  std::string fw_version_{"gaja-esphome/0.1.0"};
  std::string lang_{"pl"};

  microphone::Microphone *mic_{nullptr};
  microphone::MicrophoneSource *mic_source_{nullptr};
  micro_wake_word::MicroWakeWord *mww_{nullptr};
  speaker::Speaker *speaker_{nullptr};
  switch_::Switch *speaker_enable_{nullptr};

  uint32_t preroll_ms_{1500};
  float wake_score_{0.97f};
  float quiet_db_{-24.0f};
  float loud_db_{-12.0f};
  float vol_min_{0.7f};
  float vol_max_{1.0f};
  uint32_t reply_gap_ms_{1500};

  // ---- connection ------------------------------------------------------------------------
  void *ws_client_{nullptr};  // esp_websocket_client_handle_t
  std::atomic<bool> ws_connected_{false};
  bool ws_started_{false};
  bool hello_sent_{false};
  uint32_t hello_sent_ms_{0};
  bool server_ready_{false};
  bool reconnect_pending_{false};
  uint32_t next_attempt_ms_{0};
  uint32_t reconnect_failures_{0};
  bool auth_failed_{false};
  bool pending_pong_{false};
  uint32_t out_rate_{48000};

  // ---- state -----------------------------------------------------------------------------
  std::atomic<GajaState> state_{GajaState::RECONNECTING};
  uint32_t state_entered_ms_{0};
  uint32_t linger_deadline_ms_{0};
  uint32_t followup_deadline_ms_{0};
  uint32_t follow_up_ms_{0};
  bool follow_up_received_{false};
  bool audio_end_seen_{false};
  bool reply_volume_set_{false};
  float reply_volume_{1.0f};
  bool amp_on_{false};
  uint32_t amp_on_ms_{0};
  uint32_t amp_off_deadline_ms_{0};
  bool speaker_finishing_{false};
  uint16_t send_failures_{0};
  uint32_t last_send_drop_log_ms_{0};
  uint32_t last_play_drop_log_ms_{0};

  // wake / preroll
  std::string wake_word_{"hej_gaja"};
  size_t preroll_remaining_{0};   // bytes of pre-roll left to send during WAKING
  size_t preroll_pos_{0};         // absolute read position of the pre-roll snapshot

  // ---- buffers ---------------------------------------------------------------------------
  RollingRing preroll_ring_;   // pre-wake audio (PSRAM), written by the mic task
  SpscRing send_ring_;         // live frames to the server (PSRAM), mic task -> main loop
  SpscRing playback_ring_;     // reply PCM from the server (PSRAM), ws task -> main loop
  uint8_t *frame_{nullptr};    // one 20 ms uplink frame being assembled (640 B)
  size_t frame_len_{0};
  uint8_t *feed_{nullptr};     // one reply chunk peeked from playback_ring_ for the speaker

  // text messages from the server: fixed slots, ws task copies, main loop parses
  static constexpr size_t TEXT_SLOTS = 4;
  static constexpr size_t TEXT_SLOT_SIZE = 512;
  char text_slots_[TEXT_SLOTS][TEXT_SLOT_SIZE];
  size_t text_slot_len_[TEXT_SLOTS]{};
  volatile uint8_t text_read_{0};
  volatile uint8_t text_write_{0};

  // ---- level statistics ------------------------------------------------------------------
  uint32_t level_hist_[LEVEL_DB_BINS]{};
  uint32_t level_frames_{0};
  uint32_t last_level_log_ms_{0};
};

}  // namespace esphome::gaja_device

#endif  // USE_ESP32
