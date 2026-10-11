# Firmware urządzenia Gaja (Atom VoiceS3R / EchoS3R)

Urządzenie: M5Stack Atom VoiceS3R / EchoS3R (SKU C126-ECHO): ESP32-S3-PICO-1-N8R8 (8 MB flash, 8 MB PSRAM),
jeden mikrofon MEMS, kodek ES8311, wzmacniacz NS4150B, Wi-Fi 2.4 GHz. Sprzętu jeszcze nie ma; wszystko tutaj
jest **NIETESTOWANE** (pisane na podstawie dokumentacji). Po stronie serwera istnieje protokół v1.8
(`docs/DEVICE.md`) i symulator (`clients/device_sim.py`), więc firmware można pisać pod gotowy kontrakt.

## Etap 1: walidacja modelu wake word przez ESPHome

Plik `esphome/atom-voices3r-wakeword-test.yaml` nie rozmawia z serwerem Gaja. Robi jedno: słucha, a po
wykryciu "Hej Gaja" pika i loguje. Służy do sprawdzenia modelu microWakeWord (czułość, fałszywe alarmy)
na prawdziwym mikrofonie i obudowie, zanim powstanie własny firmware.

### Co jest potrzebne

- model wytrenowany osobno (microWakeWord): plik `hej_gaja.tflite` i manifest `hej_gaja.json`
  (w repo jest tylko **placeholder** manifestu w `esphome/models/hej_gaja.json`),
- krótki sygnał `esphome/models/beep.wav` (16 kHz, mono, PCM16, ok. 200 ms),
- ESPHome (>= 2025.8 zalecane; wspólna magistrala I2S dla mikrofonu i głośnika): `uv tool install esphome`
  albo `pipx install esphome`.

### Kroki

```bash
cd firmware/esphome
cp secrets.yaml.example secrets.yaml      # uzupełnij Wi-Fi (2.4 GHz) i hasła; plik jest w .gitignore
cp /ścieżka/do/hej_gaja.tflite models/    # razem z własnym hej_gaja.json
esphome compile atom-voices3r-wakeword-test.yaml
esphome run atom-voices3r-wakeword-test.yaml   # pierwszy raz przez USB-C, kolejne przez OTA
```

Pierwsze wgranie przez USB: jeśli urządzenie nie wchodzi w tryb bootloadera, przytrzymaj przycisk
(G41) podczas podłączania (patrz dokumentacja M5Stack). `esphome logs ...` pokazuje logi.

### Co sprawdzić po pierwszym wgraniu

1. Kodek ES8311 odpowiada na I2C (adres 0x18); brak odpowiedzi oznacza złe piny SDA/SCL (G45/G0).
2. Poziom mikrofonu: w logach `micro_wake_word` (poziom DEBUG) widać prawdopodobieństwo. Gdy zawsze ok. 0,
   zmień `mic_gain` (ES8311, 0..42 dB) i `gain_factor`; gdy mikrofon ma ciszę, spróbuj `channel: right`.
3. Wspólna magistrala I2S (mikrofon + głośnik): jeśli kompilacja lub działanie nie pasuje do wersji ESPHome,
   to pierwsze miejsce do poprawek.
4. Wzmacniacz NS4150B włącza pin G18 (przełącznik `speaker_enable`); beep słychać tylko po jego włączeniu.
5. Test na sucho: 20-30 wypowiedzi "Hej Gaja" z różnych odległości (0,5 / 2 / 4 m), z TV/muzyką w tle,
   i godzina zwykłej rozmowy bez wake word; spisz trafienia i fałszywe alarmy dla każdego `probability_cutoff`.
   Te liczby przesądzają o progu dla właściwego firmware i o tym, czy `verify_wake` na serwerze ma zostać włączone.

Piny (źródło: https://docs.m5stack.com/en/core/Atom_VoiceS3R): I2C SDA G45, SCL G0; I2S MCLK G11, BCLK G17,
WS G3, DOUT G48 (do kodeka), DIN G4 (z kodeka); wzmacniacz G18; przycisk G41; IR G47. Brak diody RGB.
Zweryfikuj ze schematem płytki przed wgraniem.

## Etap 2: własny firmware (ESP-IDF)

ESPHome wystarczy do walidacji modelu, ale nie do protokołu Gaja (strumień PCM po WebSocket, pre-roll,
barge-in, negocjacja częstotliwości). Plan własnego firmware w ESP-IDF:

| Warstwa | Komponent |
| --- | --- |
| Kodek ES8311 + I2S | `esp_codec_dev` (sterownik ES8311, I2C 0x18, MCLK z ESP32), `i2s_std` w trybie full-duplex |
| Przetwarzanie sygnału | ESP-SR AFE: AEC (echo z własnego głośnika), NS, AGC; AEC pozwala na `caps.aec=true` (barge-in) |
| Wake word | TFLite Micro + mikroWakeWord (ten sam model `.tflite`, front-end cech 40 x 10 ms), próg wybrany w etapie 1 |
| Bufor pre-roll | pierścień w PSRAM, ok. 1,5 s PCM16 16 kHz (48 KB); wysyłany zaraz po `wake` |
| Sieć | `esp_websocket_client` (wss gdy włączony TLS na serwerze), protokół v1.8 z `docs/DEVICE.md` |
| Odtwarzanie | PCM16 mono z serwera w częstotliwości wynegocjowanej w `ready` (zwykle 16 kHz), bufor ok. 300 ms |

Przebieg w pętli: mikrofon -> AFE -> (wake word) -> `hello` z `caps`, potem `wake` + pre-roll + ramki 20 ms
-> `wake_accepted` / `wake_rejected` -> odtwarzanie odpowiedzi -> `follow_up{ms}` otwiera mikrofon bez
wake word, `listen_end` go zamyka. Bez AEC urządzenie przestaje wysyłać dźwięk podczas odtwarzania i
przerywa odpowiedź po lokalnym wake word. Szczegóły i komunikaty: `docs/DEVICE.md`.

Przed pisaniem etapu 2 warto uruchomić `clients/device_sim.py` na serwerze testowym, aby zobaczyć
dokładną sekwencję komunikatów i opóźnienia, które firmware ma odtworzyć.

## Bezpieczeństwo

- Token urządzenia trzymaj w NVS (nie w kodzie ani repo); generuje go `scripts/device_token.py`.
- Port urządzeń (`[devices]`, domyślnie 8792) jest wyłączony w `config.toml`; po włączeniu ruch jest jawny
  (ws://) w sieci lokalnej, chyba że skonfigurujesz TLS (`tls_certfile`, `tls_keyfile`).
