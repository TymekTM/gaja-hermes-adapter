#include "duplex_i2s.h"

#ifdef USE_ESP32

#include "esphome/core/log.h"

#include <esp_err.h>

namespace esphome::gaja_device {

static const char *const TAG = "gaja_device.duplex";

bool DuplexBus::start() {
  if (this->started_) {
    return true;
  }
  if (this->dout_ < 0) {
    // TX still clocks the bus (and feeds the AEC reference), but no sample reaches the codec.
    ESP_LOGE(TAG, "no TX data pin (speaker i2s_dout_pin): the speaker will be SILENT");
  }
  // Both directions in ONE allocation: full-duplex mode (i2s_common.h note, see duplex_i2s.h).
  i2s_chan_config_t chan_cfg = {
      .id = I2S_NUM_0,  // the only controller wired to the codec (shared clock pins)
      .role = I2S_ROLE_MASTER,
      .dma_desc_num = 6,
      .dma_frame_num = RATE / 50,  // 960 frames = 20 ms per descriptor
      .auto_clear = false,
  };
  esp_err_t err = i2s_new_channel(&chan_cfg, &this->tx_, &this->rx_);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "i2s_new_channel(tx+rx) failed: %s", esp_err_to_name(err));
    return false;
  }

  // Identical std clock configs on both channels constitute full duplex (i2s_std.h note).
  // mclk_multiple 256 = 12.288 MHz at 48 kHz, the verified codec clocking.
  i2s_std_clk_config_t clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(RATE);
  clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;

  i2s_std_gpio_config_t tx_gpio = {
      .mclk = (gpio_num_t) this->mclk_,
      .bclk = (gpio_num_t) this->bclk_,
      .ws = (gpio_num_t) this->lrclk_,
      .dout = (gpio_num_t) this->dout_,
      .din = I2S_GPIO_UNUSED,  // owned by the RX init; I2S_GPIO_UNUSED pins are skipped
      .invert_flags = {false, false, false},
  };
  i2s_std_gpio_config_t rx_gpio = tx_gpio;
  rx_gpio.dout = I2S_GPIO_UNUSED;
  rx_gpio.din = (gpio_num_t) this->din_;

  // Left slot both ways: the ES8311's ADC data and DAC input are on the left slot (verified).
  i2s_std_slot_config_t tx_slot =
      I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
  tx_slot.slot_mask = I2S_STD_SLOT_LEFT;
  i2s_std_slot_config_t rx_slot = tx_slot;

  i2s_std_config_t tx_cfg = {.clk_cfg = clk_cfg, .slot_cfg = tx_slot, .gpio_cfg = tx_gpio};
  err = i2s_channel_init_std_mode(this->tx_, &tx_cfg);
  if (err == ESP_OK) {
    i2s_std_config_t rx_cfg = {.clk_cfg = clk_cfg, .slot_cfg = rx_slot, .gpio_cfg = rx_gpio};
    err = i2s_channel_init_std_mode(this->rx_, &rx_cfg);
  }
  if (err == ESP_OK) {
    err = i2s_channel_enable(this->tx_);  // clocks the bus; stays enabled for the lifetime
  }
  if (err == ESP_OK) {
    err = i2s_channel_enable(this->rx_);
  }
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "duplex I2S start failed: %s", esp_err_to_name(err));
    if (this->tx_ != nullptr) {
      i2s_del_channel(this->tx_);
      this->tx_ = nullptr;
    }
    if (this->rx_ != nullptr) {
      i2s_del_channel(this->rx_);
      this->rx_ = nullptr;
    }
    return false;
  }
  this->started_ = true;
  ESP_LOGI(TAG, "duplex I2S started: %" PRIu32 " Hz, 16 bit, left slot, tx+rx on one port "
                "(mclk %d bclk %d ws %d dout %d din %d)",
           RATE, this->mclk_, this->bclk_, this->lrclk_, this->dout_, this->din_);
  return true;
}

}  // namespace esphome::gaja_device

#endif  // USE_ESP32
