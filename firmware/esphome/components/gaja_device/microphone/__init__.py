"""Microphone platform backed by the gaja_device duplex I2S bus.

Full duplex with the matching ``speaker: platform: gaja_device`` (yaml ``microphone_id``):
ONE ESP-IDF I2S controller runs TX (speaker) and RX (this microphone) at the same time, both at
the ES8311-verified 48000 Hz, and this microphone decimates its 48 kHz stream to the 16 kHz
PCM16 mono every consumer (micro_wake_word, sound_level, gaja_device) expects. The clock pins
live here because this platform owns (starts) the bus. See docs/FULL_DUPLEX.md.
"""

from esphome import pins
import esphome.codegen as cg
from esphome.components import audio, esp32, microphone
import esphome.config_validation as cv
from esphome.const import CONF_ID

from .. import gaja_ns

CODEOWNERS = ["@gaja"]
DEPENDENCIES = ["esp32"]

CONF_I2S_LRCLK_PIN = "i2s_lrclk_pin"
CONF_I2S_BCLK_PIN = "i2s_bclk_pin"
CONF_I2S_MCLK_PIN = "i2s_mclk_pin"
CONF_I2S_DIN_PIN = "i2s_din_pin"

GajaDuplexMicrophone = gaja_ns.class_(
    "GajaDuplexMicrophone", microphone.Microphone, cg.Component
)


def _set_stream_limits(config):
    # Fixed stream: the duplex bus always runs 48 kHz and decimates to 16 kHz on device, so the
    # source-level validations (micro_wake_word, gaja_device) see exactly the old rates.
    audio.set_stream_limits(
        min_bits_per_sample=16,
        max_bits_per_sample=16,
        min_channels=1,
        max_channels=1,
        min_sample_rate=16000,
        max_sample_rate=16000,
    )(config)
    return config


CONFIG_SCHEMA = cv.All(
    microphone.MICROPHONE_SCHEMA.extend(
        {
            cv.GenerateID(): cv.declare_id(GajaDuplexMicrophone),
            cv.Required(CONF_I2S_LRCLK_PIN): pins.internal_gpio_output_pin_number,
            cv.Required(CONF_I2S_BCLK_PIN): pins.internal_gpio_output_pin_number,
            cv.Optional(CONF_I2S_MCLK_PIN): pins.internal_gpio_output_pin_number,
            cv.Required(CONF_I2S_DIN_PIN): pins.internal_gpio_input_pin_number,
        }
    ).extend(cv.COMPONENT_SCHEMA),
    _set_stream_limits,
    cv.only_on_esp32,
)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    await microphone.register_microphone(var, config)

    cg.add(
        var.set_pins(
            config[CONF_I2S_LRCLK_PIN],
            config[CONF_I2S_BCLK_PIN],
            config.get(CONF_I2S_MCLK_PIN, -1),
            config[CONF_I2S_DIN_PIN],
        )
    )

    # Same driver inclusion the i2s_audio component performs for its mic/speaker.
    esp32.include_builtin_idf_component("esp_driver_i2s")
    # Helps avoid callbacks being skipped due to processor load (matches i2s_audio).
    esp32.add_idf_sdkconfig_option("CONFIG_I2S_ISR_IRAM_SAFE", True)
