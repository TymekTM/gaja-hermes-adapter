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

## Etap 2: firmware Gaja (ESPHome) — `atom-voices3r-gaja.yaml`

Pełny klient protokołu urządzenia v1.9 (`docs/DEVICE.md`) jako komponent zewnętrzny ESPHome
(`firmware/esphome/components/gaja_device/`, C++). Działanie: mikrofon nasłuchuje wake word
(microWakeWord) -> po wykryciu `wake` + pre-roll 1,5 s (pierścień w PSRAM) + żywe ramki
20 ms / 640 B PCM16 16 kHz -> serwer kończy wypowiedź (`listen_end` dla urządzeń bez AEC)
-> mikrofon zwalnia wspólną magistralę I2S, wzmacniacz NS4150B (G18) się włącza, odpowiedź
(PCM16 mono 48 kHz) gra z głośnika -> `follow_up` otwiera mikrofon bez wake worda,
`listen_end` wraca do nasłuchiwania wake worda. Głośność odpowiedzi adaptuje się do poziomu
mowy użytkownika (patrz niżej). Klient łączy się z powrotem z wykładniczym backoffem
(2 s -> 60 s), odpowiada na `ping` aplikacyjny i loguje `error`.

Dlaczego ESPHome, a nie czyste ESP-IDF (plan poniżej): microWakeWord, ES8311, I2S i OTA
działają w ESPHome out-of-the-box i są zweryfikowane na urządzeniu w etapie 1; własna warstwa
to tylko WebSocket + automat stanów. Klient WebSocket to komponent managed
`espressif/esp_websocket_client` (utrzymywany przez Espressif, ws:// i wss://, automatyczne
ping/pong na poziomie protokołu; backoff kontroluje nasz komponent). Negocjacja częstotliwości:
`out_rate` (domyślnie 48000) trafia do `caps.out_rates` w hello i jest wymuszana na głośniku przez
`set_audio_stream_info` przed każdym odtwarzaniem — stream info głośnika i2s_audio domyślnie
wynosi 16 kHz (YAML-owy `sample_rate` go nie ustawia), więc bez tego I2S grałby złą prędkość.
Serwer sam resampluje TTS do tej częstotliwości (scipy polyphase, przetestowana ścieżka serwera)
— urządzenie nie resampluje w ogóle.

### Kompilacja

```bash
cd firmware/esphome
cp secrets.yaml.example secrets.yaml      # uzupełnij Wi-Fi, hasła i dane urządzenia (gitignored)
cp /ścieżka/do/hej_gaja.{json,tflite} models/   # + models/beep.wav (gitignored)
uvx esphome compile atom-voices3r-gaja.yaml     # lub: esphome compile ...
uvx esphome run atom-voices3r-gaja.yaml         # pierwszy raz USB-C, potem OTA (robi to lead)
```

Klucze w `secrets.yaml`: `wifi_ssid`, `wifi_password`, `ap_password`, `ota_password`
oraz `gaja_server_url` (np. `ws://192.168.0.10:8792/device`), `gaja_device_id`,
`gaja_device_token` (generuje `scripts/device_token.py` po stronie serwera; token drukowany
raz, trzymany w `~/.config/gaja/devices.toml`). **Nigdy nie commituj `secrets.yaml`.**

### Opcje `gaja_device:` (w yaml)

| Opcja | Domyślnie | Znaczenie |
| --- | --- | --- |
| `server_url`, `device_id`, `token` | — | dane połączenia (sekrety) |
| `out_rate` | 48000 | częstotliwość odpowiedzi: `caps.out_rates` w hello + I2S przy odtwarzaniu |
| `preroll_ms` | 1500 | pierścień pre-roll w PSRAM (0..5000; zgodne z `caps.preroll_ms`) |
| `wake_score` | 0.97 | stały score w `wake` — trigger ESPHome nie zwraca prawdopodobieństwa; server używa score tylko do barge-inu (na tym sprzęcie niemożliwego) |
| `quiet_db`, `loud_db` | -24, -12 | poziom mowy w dBFS (p90 RMS ramek 20 ms nad progiem szumu) mapowany liniowo na głośność |
| `vol_min`, `vol_max` | 0.88, 1.0 | zakres głośności odpowiedzi; 1.0 = 0 dB DAC ES8311 (= beep przy 100%); 0.5 zweryfikowane jako niesłyszalne, więc obie granice są ograniczone do <= 1.0 |
| `reply_gap_ms` | 1500 | jak długo czekać na kolejne zdanie tej samej odpowiedzi (amp jeszcze gra) |

Głośność adaptacyjna: p90 z RMS ramek 20 ms (wokół średniej ramki — offset DC się nie liczy) nad
progiem szumu (minimum okna + 6 dB; cisza między słowami i zero-pad końcówki nie liczą się do
percentyla), `vol = vol_min + clamp((db - quiet_db)/(loud_db - quiet_db)) * (vol_max - vol_min)`.
Statystyki są czyszczone przy każdym otwarciu okna mikrofonu, więc zapowiedzi minutników (bez
okna) grają z `vol_max`. Funkcje czyste (`volume_map.h`, `level_stats.h`, `rings.h`), testy
host-side: `uv run pytest tests/test_firmware_volume_map.py tests/test_firmware_level_stats.py
tests/test_firmware_rings.py -q`.
Kalibracja na urządzeniu: patrz log INFO `speech level ... dBFS ... -> reply volume ...` i sensor
`Mic RMS` (ten sam gain_factor 4, co słyszy model). Szum tła po wzmocnieniu to ok. -28 dBFS,
normalna mowa z ~1 m ok. -14 dBFS — stąd domyślne -24/-12.

### Sprawdzenie przepływu bez urządzenia

Serwer i symulator: `uv run pytest tests/test_device_protocol.py tests/test_device_sim.py -q`
oraz `clients/device_sim.py` jako referencja komunikatów (firmware odwzorowuje ten sam przepływ;
różnice: `out_rates=[out_rate]` (domyślnie 48000) wymuszane też na I2S, brak syntetycznych ramek
ciszy — mikrofon wysyła prawdziwą ciszę — i brak pacingu odpowiedzi: serwer wysyła audio tak
szybko, jak robi to TTS, więc urządzenie ma 4 MB pierścień odtwarzania plus backpressure na
WebSockecie).

## Etap 2b (odrzucone na razie): własny firmware w ESP-IDF

ESPHome wystarcza do protokołu Gaja (strumień PCM po WebSocket, pre-roll, negocjacja
częstotliwości — patrz wyżej). Poniższy plan ESP-IDF zostaje jako opcja, gdyby trzeba było
AEC (barge-in podczas odpowiedzi) — wymaga pełnego duplexu I2S i ESP-SR AFE:

| Warstwa | Komponent |
| --- | --- |
| Kodek ES8311 + I2S | `esp_codec_dev` (sterownik ES8311, I2C 0x18), `i2s_std` w trybie full-duplex |
| Przetwarzanie sygnału | ESP-SR AFE: AEC (echo z własnego głośnika), NS, AGC; AEC pozwala na `caps.aec=true` (barge-in) |
| Wake word | TFLite Micro + mikroWakeWord (ten sam model `.tflite`) |
| Sieć | `esp_websocket_client`, protokół z `docs/DEVICE.md` |

## Bezpieczeństwo

- Token urządzenia trzymaj w NVS (nie w kodzie ani repo); generuje go `scripts/device_token.py`.
- Port urządzeń (`[devices]`, domyślnie 8792) jest wyłączony w `config.toml`; po włączeniu ruch jest jawny
  (ws://) w sieci lokalnej, chyba że skonfigurujesz TLS (`tls_certfile`, `tls_keyfile`).
