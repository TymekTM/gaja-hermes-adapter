#pragma once

#ifdef USE_ESP32

#include "esphome/components/microphone/microphone.h"
#include "esphome/components/microphone/microphone_source.h"
#include "esphome/components/micro_wake_word/micro_wake_word.h"
#include "esphome/components/speaker/speaker.h"
#include "esphome/components/switch/switch.h"
#include "esphome/core/component.h"
#include "esphome/core/defines.h"
#if defined(USE_WEBSERVER) && defined(USE_LOGGER) && defined(USE_LOG_LISTENERS)
#define GAJA_LOG_DEFER 1
#include "esphome/components/logger/logger.h"
#include "esphome/components/web_server/web_server.h"
#endif

#include "level_stats.h"
#include "log_defer.h"
#include "rings.h"
#include "text_queue.h"
// Protocol v1.11 frame tags (docs/FULL_DUPLEX.md), one definition for every translation unit;
// kept in sync with gaja/devices/aec.py via tests/test_device_aec.py.
#include "frame_tag.h"

#include <atomic>
#include <cstdint>
#include <esp_event.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
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
  WAKE_BEEP,     // pre-roll sent; mic paused while the short "listening" beep plays, then STREAMING
  STREAMING,     // live 20 ms frames after a wake
  FOLLOWUP,      // live frames in the follow-up window (no wake phrase needed)
  WAIT_REPLY,    // mic closed (listen_end); reply or follow-up may arrive
  PREPARE_PLAY,  // audio_start received: waiting for the mic/I2S bus, amplifier warm-up
  PLAYING,       // feeding the reply PCM to the speaker
  LINGER,        // reply drained; waiting a moment for another utterance of the same reply
  RECORDING,     // wake-word training take (v1.10): live mic -> server, detections become
                 // record_mark (no conversation, no beep); speaker stays off, mic never stops
};

const char *gaja_state_to_string(GajaState state);

/**
 * Gaja voice device client (protocol v1.10/v1.11, docs/DEVICE.md, docs/FULL_DUPLEX.md) for the
 * M5Stack Atom VoiceS3R.
 *
 * HALF-DUPLEX (duplex: false, the verified 0.2.x behaviour): one I2S bus is shared by the
 * microphone and the speaker, so the mic is stopped before playback and restarted afterwards
 * (caps.aec = false). Wake detection runs in ESPHome's micro_wake_word component; this component
 * owns everything else: the WebSocket connection, the pre-roll ring buffer (PSRAM), live frame
 * streaming, reply playback with the NS4150B amplifier, and the adaptive reply volume.
 *
 * FULL-DUPLEX (duplex: true): the microphone (platform: gaja_device) and the speaker (platform:
 * gaja_device) share ONE ESP-IDF I2S controller in IDF full-duplex mode — both directions run at
 * the ES8311-verified 48000 Hz and the mic path is decimated 48k->16k on device (decim3.h), so
 * the mic NEVER stops: wake beep and replies play over a live mic. The TX side feeds a rolling
 * reference ring (decimated playback copy) that is streamed to the server as `caps.ref` frames
 * for its echo canceller, which is what makes speech barge-in possible without hearing Gaja's
 * own voice (phase 3, docs/FULL_DUPLEX.md).
 *
 * Audio uplink uses ONE ring (the rolling pre-roll ring): at a wake the sender starts at the
 * pre-roll snapshot (at most preroll_ms behind the head, never before the last mic restart, one
 * mic chunk ahead of the ring tail) and simply keeps reading to the ring's live head, first as a
 * burst (WAKING), then paced by the loop (STREAMING). There is no second ring to reset and no
 * hole between the pre-roll snapshot and the live stream.
 *
 * v1.10 adds wake-word training recordings (caps.record): the server records takes through the
 * device microphone to retrain the model and measures on-device detection per take. A recording
 * reuses the same single ring with the send cursor set to the head at record_start (no
 * pre-roll); the mic and micro_wake_word keep running for the whole take, so detection keeps
 * working and no mic-restart transient lands inside the audio.
 */
class GajaDevice : public Component {
 public:
  void setup() override;
  void loop() override;
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::AFTER_CONNECTION; }

  // Called from YAML (micro_wake_word.on_wake_word_detected -> lambda: id(<id>).on_wake(wake_word)).
  // During a training recording (RECORDING) a detection is a record_mark instead: no turn starts.
  void on_wake(const std::string &word);

  // For YAML automations (e.g. only run the manual test beep when no turn is in flight).
  // NOTE: true also while RECONNECTING (no link): use is_connected() for "the server accepted us".
  bool is_idle() const {
    GajaState state = this->state_.load(std::memory_order_relaxed);
    return state == GajaState::IDLE || state == GajaState::RECONNECTING;
  }

  // For YAML automations that must wait for the link (the power-on chime means "ready", not
  // "powered"): true only once the server answered `ready`, i.e. hello was accepted.
  bool is_connected() const { return this->state_.load(std::memory_order_relaxed) == GajaState::IDLE; }

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
  void set_wake_beep(bool enabled) { this->wake_beep_ = enabled; }
  void set_duplex(bool duplex) { this->duplex_ = duplex; }
  /// Full-duplex audio path: the mic never stops and the speaker never owns the bus alone.
  bool duplex() const { return this->duplex_; }
  /// The gaja_device microphone platform instance (full-duplex bus owner; set by to_code).
  void set_duplex_microphone(class GajaDuplexMicrophone *mic) { this->duplex_mic_ = mic; }
#ifdef GAJA_LOG_DEFER
  void set_log_web_server(web_server::WebServer *server) { this->log_web_server_ = server; }
#endif

 protected:
  // ---- websocket plumbing ----------------------------------------------------------------
  static void websocket_event_handler_(void *arg, esp_event_base_t base, int32_t event_id, void *event_data);
  void schedule_next_attempt_();
  void start_ws_attempt_(uint32_t now);
  /// Ask for the ws client task to be stopped. The stop itself (which BLOCKS until the task has
  /// exited: ~1 s on a live link, up to the whole connect timeout mid-connect) runs on a short-lived
  /// helper task, never on the main loop whose 5 s watchdog it would trip.
  void request_ws_stop_();
  void service_ws_stop_();
  static void ws_stop_task_(void *arg);
  bool send_text_(const char *data, size_t len);
  /// True when the uplink socket has room for a frame (a write then never times out).
  bool ws_writable_() const;
  /// Full duplex with the server's echo canceller: the mic streams through replies.
  bool stream_through_reply_() const { return this->duplex_ && this->server_aec_; }
  bool send_frame_(size_t len);
  bool send_json_wake_(uint32_t preroll_ms);  // false: the wake message was NOT sent
  void send_hello_();

  // ---- server messages -------------------------------------------------------------------
  void on_ready_(int sample_rate);
  void on_wake_accepted_();
  void on_wake_rejected_(const std::string &reason);
  void on_listen_end_();
  void on_follow_up_(uint32_t ms);
  void on_audio_start_(int sample_rate, int id);
  void on_audio_end_(bool empty);
  void on_playback_stop_();  // v1.11: the server cancelled the reply (barge-in / superseded)
  void on_ping_();
  void on_error_(const std::string &message, bool hello_was_sent);

  // ---- recording: wake-word training takes (protocol v1.10) ------------------------------
  void on_record_start_(const std::string &id, uint32_t max_ms);
  void on_record_stop_(const std::string &id);
  /// End the take: send the queued frames, flush the partial frame zero-padded, send
  /// record_end (retried from the loop when the send is only "busy"), back to idle.
  void finish_record_(const char *reason);
  bool send_record_started_(const std::string &id);
  void send_record_error_(const std::string &id, const char *reason);
  void send_record_mark_(const std::string &id, const std::string &word, uint32_t t_ms);
  bool send_record_end_(const std::string &id, const char *reason, uint32_t ms);
  /// Retry a "busy" record_end until its watchdog (runs from loop(), any state).
  void service_record_end_();

  // ---- state machine ---------------------------------------------------------------------
  void enter_reconnecting_(const char *reason);
  void enter_idle_();
  void enter_idle_(bool reset_preroll_window);  // full duplex: a wake barge-in keeps the window
  /// Stop the running reply at once: flush the playback ring (up to ~44 s buffered), drop the
  /// downlink sample-alignment debt with it, stop the speaker, amplifier off. Used by the
  /// wake-during-playback barge-in and by the server's `playback_stop` message.
  void cancel_playback_(const char *why);
  /// Resume the live mic stream after a cancelled reply while the server still owns an open
  /// mic window (VAD barge-in: the ongoing utterance continues into the new turn).
  void resume_live_streaming_();
  /// True when the server expects mic frames in the current state (full-duplex uplink gate).
  bool mic_window_open() const { return this->mic_window_open_; }
  void begin_playback_transition_();  // from any streaming/idle state: mic off -> amplifier -> PLAYING
  void finish_reply_();               // drained: amplifier off, then follow-up or idle
  void amp_off_();                    // switch the NS4150B off if we switched it on
  void recover_wedged_speaker_(const char *why);  // stop the speaker, drop the reply, finish
  void open_followup_();
  void enter_followup_();
  void close_mic_stream_();

  // ---- audio paths -----------------------------------------------------------------------
  void on_mic_data_(const std::vector<uint8_t> &data);  // microphone task context
  void service_uplink_(uint32_t max_frames);            // WAKING burst / STREAMING live pace
  void service_ref_uplink_(uint32_t max_frames);        // v1.11: playback reference frames
  /// Wire offset of the PCM payload inside frame_ (v1.11): a full-duplex device tags every
  /// binary frame (frame_tag.h) because it advertises caps.ref; a half-duplex device keeps the
  /// raw v1.10 frames byte-for-byte (0.2.5 behaviour; the server strips the tag only for
  /// caps.ref devices -- Session.handle_audio).
  size_t frame_payload_off_() const { return this->duplex_ ? 1U : 0U; }
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
  std::string fw_version_{"gaja-esphome/0.3.4"};
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
  bool wake_beep_{true};
  bool server_aec_{false};      // ready.device.aec: the server cancels our echo, so the mic may stream through replies
  bool audio_since_cue_{true};  // reply audio started since the last wake beep (the follow-up beep is a new cue)
  bool duplex_{false};  // full-duplex I2S: mic runs during playback (see class comment)

  // ---- wake beep (WAKE_BEEP) -----------------------------------------------------------------
  // Half-duplex bus: the beep pauses the mic AFTER the pre-roll (with the wake phrase) was sent.
  // Full duplex: the mic keeps running (MIC_STOP is skipped); the beep's echo lands in the ring
  // and is dropped from the uplink by resetting the send cursor at the beep's end.
  // The same beep opens every follow-up window (BeepExit::FOLLOWUP), before the mic restarts.
  // Server messages that arrive meanwhile only choose where the beep exits to (beep_exit_).
  enum class BeepPhase : uint8_t { MIC_STOP, AMP, PLAY, FINISH, TAIL };
  enum class BeepExit : uint8_t { STREAM, WAIT_REPLY, IDLE, FOLLOWUP };
  void enter_wake_beep_(BeepExit exit);
  void service_wake_beep_(uint32_t now);
  void end_wake_beep_();
  uint8_t *beep_pcm_{nullptr};  // PCM16 mono at out_rate_, generated in setup() (PSRAM)
  size_t beep_bytes_{0};
  size_t beep_pos_{0};
  BeepPhase beep_phase_{BeepPhase::MIC_STOP};
  uint32_t beep_drained_ms_{0};
  BeepExit beep_exit_{BeepExit::STREAM};
  bool server_thinking_{false};  // last server `state` was "thinking": the reply wait may be long

  // ---- connection ------------------------------------------------------------------------
  void *ws_client_{nullptr};  // esp_websocket_client_handle_t
  void *ws_tcp_{nullptr};     // esp_transport_handle_t: our TCP transport under the client (ws_writable_)
  std::string ws_path_{};     // backs the websocket transport's path
  std::atomic<bool> ws_connected_{false};
  // True while the esp_websocket_client task is believed running: set before start(), cleared ONLY
  // by the join helper after esp_websocket_client_stop() returned (the task is really gone then). With disable_auto_reconnect every
  // failed connect and every dropped connection ends the task by itself, so the main loop never
  // has to stop a task that is mid-connect (that stop would block for the connect timeout).
  std::atomic<bool> ws_task_running_{false};
  std::atomic<bool> ws_stop_inflight_{false};  // the helper task is inside esp_websocket_client_stop()
  // Set by WEBSOCKET_EVENT_FINISH: the task is on its way out but still closes the transport
  // after dispatching the event, so a restart must JOIN it first (esp_websocket_client_stop()
  // returns only once the task's STOPPED bit is set) before ws_task_running_ is cleared.
  std::atomic<bool> ws_task_ended_{false};
  uint32_t ws_stop_fail_log_ms_{0};            // rate limit for "could not create the stop task"
  bool ws_stop_wanted_{false};                 // main loop: a live connection must still be stopped
  uint32_t ws_attempt_ms_{0};                  // millis() of the last esp_websocket_client_start()
  bool ws_started_{false};
  // hello_sent_ / server_ready_ are written by BOTH the ws task (events clear them) and the main
  // loop (hello send, ready), so they are atomics.
  std::atomic<bool> hello_sent_{false};
  uint32_t hello_sent_ms_{0};
  std::atomic<bool> server_ready_{false};
  // The server's `ready.device.logs`: it accepts `log` messages (the deferred log lines are
  // forwarded to its journal). Cleared with server_ready_ on every disconnect.
  std::atomic<bool> server_logs_{false};
  bool boot_logged_{false};  // the reset reason goes to the server once per boot
  bool reconnect_pending_{false};
  uint32_t next_attempt_ms_{0};
  uint32_t reconnect_failures_{0};
  bool auth_failed_{false};
  bool pending_pong_{false};
  uint32_t last_reconnect_log_ms_{0};  // rate-limits repeated enter_reconnecting_ warnings

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
  uint32_t speaker_finish_ms_{0};      // millis() when speaker_->finish() was called
  uint16_t send_failures_{0};
  uint32_t send_blocked_since_ms_{0};  // 0 = uplink socket has room; else millis() it filled up
  uint32_t play_dropped_seen_{0};   // playback_ring_.dropped() (bytes) at the last drop warning
  uint32_t streaming_deadline_ms_{0};  // re-armed while gated speech flows (hard cap applies)

  // ---- recording (protocol v1.10) --------------------------------------------------------
  // Main-loop-only state. rec_sent_bytes_ counts mic bytes successfully handed to the
  // WebSocket during this take: record_mark.t_ms and record_end.ms are derived from it, so a
  // stuck frame (assembled but unsent) or a zero-padded flush never inflates the audio time.
  std::string rec_id_{};           // active take id ([a-z0-9_-]{1,64}); irrelevant once stopped
  uint32_t rec_max_ms_{0};         // the server's cap for this take
  uint32_t rec_deadline_ms_{0};    // auto-stop (reason "max") deadline
  uint32_t rec_sent_bytes_{0};     // real mic bytes sent so far in this take
  bool rec_end_pending_{false};    // record_end send failed "busy"; retry from the loop
  std::string rec_end_id_{};       // id of the finished take (a new take may already be rec_id_)
  uint32_t rec_end_ms_{0};         // payload of the pending record_end
  const char *rec_end_reason_{"stop"};  // a string literal ("stop"/"max"): static storage
  uint32_t rec_end_deadline_ms_{0};     // give up retrying record_end after this

  // mic / wake word power (see update_mic_power_)
  bool mic_wanted_{false};
  uint32_t mic_not_running_since_ms_{0};  // 0 = ok; else millis() since requested-but-not-RUNNING
  bool mic_requested_{false};       // our last actual mic_source_->start()/stop() call
  bool mww_wanted_{false};
  bool mww_stop_requested_{false};  // we already called mww_->stop() for this stop cycle
  // Full duplex (v1.11): the server-side mic window and the playback-reference stream cursor.
  class GajaDuplexMicrophone *duplex_mic_{nullptr};
  bool mic_window_open_{false};     // server expects mic frames (open since the wake, until idle)
  ring_pos_t ref_pos_{0};           // absolute send cursor into the duplex mic's reference ring

  // wake / preroll: the rolling ring is the single uplink source. Positions are 32-bit modular
  // counters (ring_pos_t, they wrap after ~37 h): compare them only through ring_pos_diff() /
  // ring_pos_before(), never as raw integers.
  std::string wake_word_{"hej_gaja"};
  ring_pos_t stream_pos_{0};         // absolute send cursor into preroll_ring_
  ring_pos_t wake_head_{0};          // ring head at the wake instant (end of the pre-roll burst)
  ring_pos_t preroll_valid_from_{0}; // ring position of the last mic (re)start: audio before it is
                                     // from a previous turn and must not be sent as pre-roll
  ring_pos_t level_skip_until_pos_{0};  // mic restart transient: frames before it are not "speech"
  uint32_t preroll_sent_ms_{0};

  // downlink sample alignment (WS task only): fragments have arbitrary lengths, so a drop can
  // leave the byte stream an odd number of bytes short -- the next fragment then gives one byte
  // back so every pushed byte lands on a 2-byte sample boundary again.
  uint32_t dl_dropped_bytes_{0};
  bool dl_skip_leading_{false};
  // Set by the main loop when it discards the playback ring mid-stream (stall recovery): the ws
  // task owns the two fields above, so it applies the reset itself at the next audio fragment.
  std::atomic<bool> dl_reset_request_{false};
  // Downlink diagnostics (ws task writes, the main loop reads and resets once a second).
  std::atomic<uint32_t> diag_rx_bytes_{0};
  std::atomic<uint32_t> diag_rx_last_ms_{0};
  std::atomic<uint32_t> diag_rx_max_gap_ms_{0};
  std::atomic<uint32_t> diag_cb_max_ms_{0};
  uint32_t diag_next_ms_{0};
  uint32_t diag_loop_last_ms_{0};
  uint32_t diag_loop_max_ms_{0};
  // Per-reply worst cases (log_diag_ folds each second in), summarised once per reply at INFO.
  uint32_t reply_rx_bytes_{0};
  uint32_t reply_gap_max_ms_{0};
  uint32_t reply_cb_max_ms_{0};
  uint32_t reply_loop_max_ms_{0};
  void log_diag_(uint32_t now);
  void log_reply_summary_();

  // ---- buffers ---------------------------------------------------------------------------
  RollingRing preroll_ring_;   // pre-wake + live audio (PSRAM), written by the mic task
  size_t preroll_ring_capacity_{0};
  SpscRing playback_ring_;     // reply PCM from the server (PSRAM), ws task -> main loop
  uint8_t *frame_{nullptr};    // one 20 ms uplink frame being assembled (640 B)
  size_t frame_len_{0};
  uint8_t *feed_{nullptr};     // one reply chunk peeked from playback_ring_ for the speaker

  // Text messages from the server: the ws task copies whole messages into a class-aware bounded
  // queue (text_queue.h: log-only types are discarded under pressure and evicted first, control
  // messages never), the main loop parses them. 1 KiB items (PSRAM); control messages are tiny,
  // transcripts/utterances may be long.
  static constexpr size_t TEXT_QUEUE_LEN = 32;
  static constexpr size_t TEXT_ITEM_SIZE = 1024;  // last byte reserved for the NUL
  struct TextMessage {
    uint16_t len;
    bool hello_sent;  // hello_sent_ at receive time: CLOSED clears it before the loop parses
    bool droppable;   // log-only type (server_msg.h): may be discarded/evicted under pressure
    char buf[TEXT_ITEM_SIZE - sizeof(uint16_t) - 2 * sizeof(bool)];
  };
  TextQueue<TextMessage, TEXT_QUEUE_LEN> text_queue_;
  SemaphoreHandle_t text_lock_{nullptr};  // serialises the ws task (push) and the main loop (pop)
  TextMessage text_scratch_{};  // accumulation buffer for a fragmented message (ws task only)
  TextMessage text_rx_{};       // the message being handled (main loop only; keeps 1 KiB off its stack)
  std::atomic<uint32_t> text_discarded_{0};  // log-only messages dropped/evicted under pressure
  std::atomic<uint32_t> text_lost_{0};       // CONTROL messages that found no room (should stay 0)
  uint32_t text_discarded_seen_{0};
  uint32_t text_lost_seen_{0};
  void enqueue_text_(const TextMessage &msg);        // ws task: queue one complete message
  void handle_server_json_(const TextMessage &msg);  // main loop: parse and dispatch one message

  // ---- level statistics ------------------------------------------------------------------
  LevelTracker levels_{};

  // ---- deferred web log stream (log_defer.h) -------------------------------------------------
  // The web server's own log forwarding is off (yaml `web_server: log: false`): every line we
  // capture is held in a PSRAM ring and handed out only once the device has been IDLE for
  // LOG_FLUSH_IDLE_MS, a few lines per loop iteration: to the web server, and the lines marked
  // LOG_FORWARD_FLAG also to the Gaja server as `log` messages (its journal keeps them when no
  // web UI is open). While the link is down the lines are held so the server still gets the
  // reason for the outage; after LOG_HOLD_OFFLINE_MS offline they go to the web UI only.
  static constexpr size_t LOG_DEFER_BYTES = 32 * 1024;
  static constexpr uint32_t LOG_FLUSH_IDLE_MS = 1000;
  static constexpr int LOG_FLUSH_LINES = 8;
  static constexpr uint32_t LOG_HOLD_OFFLINE_MS = 60000;
  static constexpr uint8_t LOG_FORWARD_FLAG = 0x80;  // OR-ed into the stored level byte
  bool forward_log_(uint8_t level, const char *line, size_t len);
  void log_link_();
  void setup_log_defer_();
  void flush_deferred_logs_(uint32_t now);
#ifdef GAJA_LOG_DEFER
  web_server::WebServer *log_web_server_{nullptr};
  LogDefer log_defer_;
  SemaphoreHandle_t log_lock_{nullptr};      // any logging task (push) vs the main loop (pop)
  std::atomic<uint32_t> log_lost_{0};        // lines that met a held lock (not recorded)
  uint32_t log_busy_ms_{0};                  // last loop iteration the device was not idle
  char log_line_[LogDefer::MAX_LINE + 1]{};  // main loop only
#endif
};

}  // namespace esphome::gaja_device

#endif  // USE_ESP32
