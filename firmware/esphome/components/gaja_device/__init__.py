"""ESPHome external component: Gaja voice device client (protocol v1.9, docs/DEVICE.md).

Implements the device side of ``WS /device`` on the M5Stack Atom VoiceS3R: wake word
(via ESPHome's micro_wake_word), pre-roll ring buffer, live PCM streaming, reply playback
on the shared I2S bus (half-duplex, ``caps.aec = false``) and the adaptive reply volume.

The WebSocket client is Espressif's managed component ``espressif/esp_websocket_client``:
maintained, supports ws:// and wss://, auto-answers protocol pings, and delivers frames
through events we can push into PSRAM ring buffers. Reconnection/backoff is owned by the
component itself.
"""

import esphome.codegen as cg
from esphome.components import esp32, microphone, micro_wake_word, speaker, switch
import esphome.config_validation as cv
from esphome.const import CONF_ID

CONF_LANG = "lang"
CONF_TOKEN = "token"

CODEOWNERS = ["@gaja"]

DEPENDENCIES = ["microphone", "micro_wake_word", "speaker", "switch", "json"]
AUTO_LOAD = ["network"]

gaja_ns = cg.esphome_ns.namespace("gaja_device")
GajaDevice = gaja_ns.class_("GajaDevice", cg.Component)

CONF_SERVER_URL = "server_url"
CONF_DEVICE_ID = "device_id"
CONF_FIRMWARE_VERSION = "firmware_version"
CONF_MICROPHONE = "microphone"
CONF_WAKE_WORD = "wake_word"
CONF_SPEAKER = "speaker"
CONF_SPEAKER_ENABLE = "speaker_enable"
CONF_OUT_RATE = "out_rate"
CONF_PREROLL_MS = "preroll_ms"
CONF_WAKE_SCORE = "wake_score"
CONF_QUIET_DB = "quiet_db"
CONF_LOUD_DB = "loud_db"
CONF_VOL_MIN = "vol_min"
CONF_VOL_MAX = "vol_max"
CONF_REPLY_GAP_MS = "reply_gap_ms"


def _websocket_url(value):
    value = cv.string_strict(value)
    if not (value.startswith("ws://") or value.startswith("wss://")):
        raise cv.Invalid("server_url must start with ws:// or wss:// (e.g. ws://192.168.0.10:8792/device)")
    return value


def _volume_range(value):
    # 1.0 = full verified scale (ES8311 DAC at 0 dB on the Atom VoiceS3R). Above 1.0 the DAC is
    # boosted past its clean range and 0.5 (about -32 dB) was verified inaudible on the device,
    # so the usable window is clamped to [0.05, 1.0]: volume 0 would mute replies and anything
    # below ~0.5 is below the verified-audible floor anyway.
    return cv.float_range(0.05, 1.0)(value)


CONFIG_SCHEMA = cv.All(
    cv.Schema(
        {
            cv.GenerateID(): cv.declare_id(GajaDevice),
            cv.Required(CONF_SERVER_URL): _websocket_url,
            cv.Required(CONF_DEVICE_ID): cv.string_strict,
            cv.Required(CONF_TOKEN): cv.string_strict,
            cv.Optional(CONF_FIRMWARE_VERSION, default="gaja-esphome/0.1.0"): cv.string_strict,
            cv.Optional(CONF_LANG, default="pl"): cv.one_of("pl", "en", "auto", lower=True),
            cv.Required(CONF_MICROPHONE): microphone.microphone_source_schema(
                min_bits_per_sample=16,
                max_bits_per_sample=16,
                min_channels=1,
                max_channels=1,
            ),
            cv.Required(CONF_WAKE_WORD): cv.use_id(micro_wake_word.MicroWakeWord),
            cv.Required(CONF_SPEAKER): cv.use_id(speaker.Speaker),
            cv.Required(CONF_SPEAKER_ENABLE): cv.use_id(switch.Switch),
            # The rate advertised in hello caps.out_rates and asserted on the speaker before every
            # playback (the i2s_audio speaker's stream info defaults to 16 kHz and the YAML
            # sample_rate does NOT set it -- the I2S clock follows the stream info).
            cv.Optional(CONF_OUT_RATE, default=48000): cv.int_range(min=8000, max=48000),
            # Pre-roll ring: protocol caps.preroll_ms is 0..5000, but this firmware always
            # buffers live audio through the same ring (it IS the uplink stream), so a floor of
            # 250 ms is enforced; 0 would mean "no uplink at all" once the ring were skipped.
            cv.Optional(CONF_PREROLL_MS, default=1500): cv.int_range(min=250, max=5000),
            cv.Optional(CONF_WAKE_SCORE, default=0.97): cv.float_range(0.0, 1.0),
            # Adaptive reply volume: a high percentile of the streamed utterance's 20 ms frame
            # RMS over the noise-gated speech frames, in dBFS, mapped linearly to
            # [vol_min, vol_max] and clamped. Volume 1.0 is the loudest verified clean level
            # (ES8311 DAC at 0 dB on the Atom VoiceS3R); 0.5 was verified inaudible there, so
            # both ends are clamped to <= 1.0.
            cv.Optional(CONF_QUIET_DB, default=-24.0): cv.float_range(-90.0, 0.0),
            cv.Optional(CONF_LOUD_DB, default=-12.0): cv.float_range(-90.0, 0.0),
            cv.Optional(CONF_VOL_MIN, default=0.88): _volume_range,
            cv.Optional(CONF_VOL_MAX, default=1.0): _volume_range,
            # How long to keep the reply "open" for another utterance of the same answer.
            cv.Optional(CONF_REPLY_GAP_MS, default=1500): cv.int_range(min=100, max=10000),
        }
    ).extend(cv.COMPONENT_SCHEMA),
    cv.only_on_esp32,
    cv.require_esphome_version(2025, 8, 0),
)


def _validate_ranges(config):
    if config[CONF_QUIET_DB] >= config[CONF_LOUD_DB]:
        raise cv.Invalid("quiet_db must be smaller than loud_db")
    if config[CONF_VOL_MIN] > config[CONF_VOL_MAX]:
        raise cv.Invalid("vol_min must not be greater than vol_max")
    return config


FINAL_VALIDATE_SCHEMA = cv.All(
    cv.Schema(
        {
            cv.Required(CONF_MICROPHONE): microphone.final_validate_microphone_source_schema(
                "gaja_device", sample_rate=16000
            ),
        },
        extra=cv.ALLOW_EXTRA,
    ),
    _validate_ranges,
)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    cg.add(var.set_server_url(config[CONF_SERVER_URL]))
    cg.add(var.set_device_id(config[CONF_DEVICE_ID]))
    cg.add(var.set_token(config[CONF_TOKEN]))
    cg.add(var.set_firmware_version(config[CONF_FIRMWARE_VERSION]))
    cg.add(var.set_lang(config[CONF_LANG]))
    cg.add(var.set_out_rate(config[CONF_OUT_RATE]))
    cg.add(var.set_preroll_ms(config[CONF_PREROLL_MS]))
    cg.add(var.set_wake_score(config[CONF_WAKE_SCORE]))
    cg.add(var.set_quiet_db(config[CONF_QUIET_DB]))
    cg.add(var.set_loud_db(config[CONF_LOUD_DB]))
    cg.add(var.set_vol_min(config[CONF_VOL_MIN]))
    cg.add(var.set_vol_max(config[CONF_VOL_MAX]))
    cg.add(var.set_reply_gap_ms(config[CONF_REPLY_GAP_MS]))

    mic_source = await microphone.microphone_source_to_code(config[CONF_MICROPHONE])
    cg.add(var.set_microphone_source(mic_source))
    mic = await cg.get_variable(config[CONF_MICROPHONE][CONF_MICROPHONE])
    cg.add(var.set_microphone(mic))

    mww = await cg.get_variable(config[CONF_WAKE_WORD])
    cg.add(var.set_wake_word(mww))
    speaker = await cg.get_variable(config[CONF_SPEAKER])
    cg.add(var.set_speaker(speaker))
    enable = await cg.get_variable(config[CONF_SPEAKER_ENABLE])
    cg.add(var.set_speaker_enable(enable))

    # WebSocket client (ws:// and wss://), resolved by the IDF component manager.
    esp32.add_idf_component(name="espressif/esp_websocket_client", ref="^1.8.0")
