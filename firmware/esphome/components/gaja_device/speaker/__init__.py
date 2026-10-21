"""Speaker platform on the gaja_device duplex I2S bus.

Attaches to the ``microphone: platform: gaja_device`` bus (yaml ``microphone_id``): the same
ESP-IDF controller runs TX (this speaker) and RX (microphone) at once — the verified half-duplex
restriction is gone. Audio plays at 48 kHz mono 16-bit on the left slot (the ES8311-verified
configuration); DAC volume/mute go through the optional ``audio_dac`` (es8311) exactly like on
the stock i2s_audio speaker. The bus itself never stops: the TX task writes silence while no
audio is queued, because the microphone on the same controller dies with the shared clock. See
docs/FULL_DUPLEX.md.
"""

from esphome import pins
import esphome.codegen as cg
from esphome.components import esp32, speaker
import esphome.config_validation as cv
from esphome.const import CONF_ID

from .. import gaja_ns
from ..microphone import CONF_I2S_DIN_PIN, GajaDuplexMicrophone

CODEOWNERS = ["@gaja"]
DEPENDENCIES = ["esp32"]

CONF_MICROPHONE_ID = "microphone_id"
CONF_I2S_DOUT_PIN = "i2s_dout_pin"

GajaDuplexSpeaker = gaja_ns.class_("GajaDuplexSpeaker", speaker.Speaker, cg.Component)

CONFIG_SCHEMA = cv.All(
    speaker.SPEAKER_SCHEMA.extend(
        {
            cv.GenerateID(): cv.declare_id(GajaDuplexSpeaker),
            cv.Required(CONF_MICROPHONE_ID): cv.use_id(GajaDuplexMicrophone),
            # The codec's DAC data input (ES8311 DSDIN). Without it the TX channel clocks the bus
            # and fills the AEC reference, but no sample ever leaves the chip (silent speaker).
            cv.Required(CONF_I2S_DOUT_PIN): pins.internal_gpio_output_pin_number,
        }
    ).extend(cv.COMPONENT_SCHEMA),
    cv.only_on_esp32,
)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)
    await speaker.register_speaker(var, config)

    mic = await cg.get_variable(config[CONF_MICROPHONE_ID])
    cg.add(var.set_bus_mic(mic))
    cg.add(var.set_dout_pin(config[CONF_I2S_DOUT_PIN]))

    # The TX task drives the I2S TX channel, so the driver must be linked for this platform too
    # (idempotent with the microphone platform's call).
    esp32.include_builtin_idf_component("esp_driver_i2s")

# CONF_I2S_DIN_PIN is re-exported on purpose: the gaja_device component's final validation uses
# it as the marker that the configured microphone is THIS platform (full-duplex capable).
__all__ = ["CONF_I2S_DIN_PIN", "CONF_MICROPHONE_ID", "GajaDuplexSpeaker"]
