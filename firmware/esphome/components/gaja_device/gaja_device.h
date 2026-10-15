#pragma once

#ifdef USE_ESP32

#include "esphome/components/microphone/microphone.h"
#include "esphome/components/microphone/microphone_source.h"
#include "esphome/components/micro_wake_word/micro_wake_word.h"
#include "esphome/components/speaker/speaker.h"
#include "esphome/components/switch/switch.h"
#include "esphome/core/component.h"

#include "level_stats.h"
#include "rings.h"

#include <atomic>
#include <cstdint>
#include <esp_event.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <string>
#include <vector>

namespace esphome::gaja_device {

// The Atom VoiceS3R's verified full-scale: media_player volume 1.0 * volume_max 0.75 = ES8311 DAC
// register 0xBF = 0 dB (the loudest level that does not clip). Volumes passed to
// speaker->set_volume() are scaled by this constant, so 1.0 here means "loudest verified".
// 0.5 (i.e. ~-32 dB on the ES8311's ~dB-linear volume scale) was verified INAUDIBLE on the device,
// so vol_min/vol_max are clamped to <= 1.0 and the default vol_min stays well above that.
static constexpr float SPEAKER_VOLUME_FULL_SCALE = 0.75f;

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
 *
 * Audio uplink uses ONE ring (the rolling pre-roll ring): at a wake the sender starts at the
 * pre-roll snapshot and simply keeps reading to the ring's live head, first as a burst (WAKING),
 * then paced by the loop (STREAMING). There is no second ring to reset and no hole between the
 * pre-roll snapshot and the live stream.
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
  void set_out_rate(uint32_t rate) { this->out_rate_ = rate; }
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
  void send_json_wake_(uint32_t preroll_ms);
  void send_hello_();

  // ---- server messages -------------------------------------------------------------------
  void handle_server_json_(const char *data, size_t len);
  void on_ready_(int sample_rate);
  void on_wake_accepted_();
  void on_wake_rejected_(const std::string &reason);
  void on_listen_end_();
  void on_follow_up_(uint32_t ms);
  void on_audio_start_(int sample_rate, int id);
  void on_audio_end_(bool empty);
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
  void service_uplink_(uint32_t max_frames);            // WAKING burst / STREAMING live pace
  void service_speaker_();
  void service_reconnect_();
  /// Desired mic/wake-word power, applied when the underlying components have settled: a start()
  /// while they are still stopping is silently ignored by ESPHome, and a stop() before the mic
  /// ever started leaks one of its "active listener" semaphore counts (the mic would never stop
  /// again and would hold the shared I2S bus forever).
  void update_mic_power_();

  // ---- misc ------------------------------------------------------------------------------
  void process_text_queue_();
  bool mic_stopped_() const;
  void apply_reply_volume_();
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

  uint32_t out_rate_{48000};   // configured reply rate; asserted on the speaker before playback
  uint32_t preroll_ms_{1500};
  float wake_score_{0.97f};
  float quiet_db_{-24.0f};
  float loud_db_{-12.0f};
  float vol_min_{0.88f};
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
  uint32_t drained_since_ms_{0};      // 0 = reply not drained; else the millis() it drained at
  uint32_t play_stalled_since_ms_{0}; // 0 = speaker making progress
  bool speaker_finishing_{false};
  uint16_t send_failures_{0};
  uint32_t last_send_drop_log_ms_{0};
  uint32_t last_play_drop_log_ms_{0};

  // mic / wake word power (see update_mic_power_)
  bool mic_wanted_{false};
  bool mic_requested_{false};  // our last actual mic_source_->start()/stop() call
  bool mww_wanted_{false};

  // wake / preroll: the rolling ring is the single uplink source
  std::string wake_word_{"hej_gaja"};
  size_t stream_pos_{0};            // absolute send cursor into preroll_ring_
  size_t wake_head_{0};             // ring head at the wake instant (end of the pre-roll burst)
  size_t preroll_valid_from_{0};    // ring position of the last mic (re)start: audio before it is
                                    // from a previous turn and must not be sent as pre-roll
  uint32_t preroll_sent_ms_{0};

  // ---- buffers ---------------------------------------------------------------------------
  RollingRing preroll_ring_;   // pre-wake + live audio (PSRAM), written by the mic task
  SpscRing playback_ring_;     // reply PCM from the server (PSRAM), ws task -> main loop
  uint8_t *frame_{nullptr};    // one 20 ms uplink frame being assembled (640 B)
  size_t frame_len_{0};
  uint8_t *feed_{nullptr};     // one reply chunk peeked from playback_ring_ for the speaker

  // text messages from the server: FreeRTOS queue (ws task copies whole messages, main loop
  // parses). 1 KiB items; control messages are tiny, transcripts/utterances may be long.
  static constexpr size_t TEXT_QUEUE_LEN = 16;
  static constexpr size_t TEXT_ITEM_SIZE = 1024;  // last byte reserved for the NUL
  struct TextMessage {
    uint16_t len;
    char buf[TEXT_ITEM_SIZE - sizeof(uint16_t)];
  };
  QueueHandle_t text_queue_{nullptr};
  TextMessage text_scratch_{};  // accumulation buffer for a fragmented message (ws task only)

  // ---- level statistics ------------------------------------------------------------------
  LevelTracker levels_{};
};

}  // namespace esphome::gaja_device

#endif  // USE_ESP32
