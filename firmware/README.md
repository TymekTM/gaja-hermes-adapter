# Firmware urządzenia Gaja (Atom VoiceS3R / EchoS3R)

> **Stan wersji.**
> - **Stabilna: 0.2.5 (półdupleks)**, tag git `fw-0.2.5` (commit `9a2f1b2`). Wersja używana na co dzień; znany otwarty problem to sporadyczne zacięcia odbioru w trakcie odpowiedzi (`docs/TODO.md`).
> - **Eksperymentalna: 0.3.x (pełny dupleks)**, 0.3.1-0.3.3 (obecny HEAD). Serwerowe usuwanie echa nie działa na Atom VoiceS3R (echo ok. -7 dBFS przy mowie ok. -26 dBFS, obcinanie próbek, dryf opóźnienia), więc `[devices] speech_barge_in` jest domyślnie wyłączone, a urządzenie pełnodupleksowe jest obsługiwane jak półdupleksowe. W teście na żywo na 0.3.2 "Hej Gaja" nie przerwało odpowiedzi, a urządzenie zapętlało się na własnym echu; 0.3.3 naprawia pętlę, ale nie było jeszcze testowane na żywo (`docs/FULL_DUPLEX.md`, `docs/TODO.md`).
> - Zatwierdzonego firmware 0.2.6 nie ma; 0.2.6t było niezatwierdzonym eksperymentem (`docs/TODO.md`).
> - Poniższy opis (etapy 1-2) pisano pod półdupleks, zanim powstało 0.3.x.

Urządzenie: M5Stack Atom VoiceS3R / EchoS3R (SKU C126-ECHO): ESP32-S3-PICO-1-N8R8 (8 MB flash, 8 MB PSRAM),
jeden mikrofon MEMS, kodek ES8311, wzmacniacz NS4150B, Wi-Fi 2.4 GHz. Firmware działa na tym
urządzeniu. Kontrakt z serwerem: protokół `docs/DEVICE.md` (v1.11) i symulator (`clients/device_sim.py`).

## Etap 1: walidacja modelu wake word przez ESPHome

Plik `esphome/atom-voices3r-wakeword-test.yaml` nie rozmawia z serwerem Gaja. Robi jedno: słucha, a po
wykryciu "Hej Gaja" pika i loguje. Służy do sprawdzenia modelu microWakeWord (czułość, fałszywe alarmy)
na prawdziwym mikrofonie i obudowie, zanim powstanie własny firmware.

### Co jest potrzebne

- wytrenowany model microWakeWord: `esphome/models/hej_gaja.tflite` i manifest `hej_gaja.json`
  (są w repo; własny model po treningu podmienia oba pliki),
- krótki sygnał `esphome/models/beep.wav` (16 kHz, mono, PCM16, ok. 200 ms),
- ESPHome (>= 2025.8 zalecane; wspólna magistrala I2S dla mikrofonu i głośnika): `uv tool install esphome`
  albo `pipx install esphome`.

### Kroki

```bash
cd firmware/esphome
cp secrets.yaml.example secrets.yaml      # uzupełnij Wi-Fi (2.4 GHz) i hasła; plik jest w .gitignore
esphome compile atom-voices3r-wakeword-test.yaml
esphome run atom-voices3r-wakeword-test.yaml   # pierwszy raz przez USB-C, kolejne przez OTA
```

Pierwsze wgranie przez USB: jeśli urządzenie nie wchodzi w tryb bootloadera, przytrzymaj przycisk
(G41) podczas podłączania (patrz dokumentacja M5Stack). `esphome logs ...` pokazuje logi.

W `atom-voices3r-gaja.yaml` logi w web UI (`http://<ip>/`) są opóźnione: w trakcie rozmowy urządzenie
tylko je buforuje (32 KiB w PSRAM), a wysyła ok. 1 s po powrocie do bezczynności (`log_web_server`,
`web_server: log: false`). Strumień logów na żywo zacinał sieć w trakcie odpowiedzi. Przy przepełnieniu
najstarsze linie przepadają z komunikatem „N log lines dropped while busy”.

Od 0.2.4 urządzenie wysyła też (w bezczynności) swoje linie INFO/WARN/ERROR oraz wszystkie z `gaja_device` i
`micro_wake_word` do serwera (wiadomość `log`, patrz `docs/DEVICE.md`); trafiają do journala adaptera jako
`gaja.devices.remote`: `journalctl --user -u gaja-adapter | grep gaja.devices.remote`. Po zerwaniu połączenia
linie czekają w buforze i idą po ponownym połączeniu (po 60 s offline trafiają już tylko do web UI). Na każde
połączenie urządzenie loguje `link: bssid ... rssi ...`, raz po starcie `boot: reset reason ...`, a po każdej
odpowiedzi `reply: N KiB down, worst downlink gap ... ms`. `wifi: post_connect_roaming: false`: bez skanów
roamingu co 5 min (Atom stoi w miejscu, a skan poza kanałem zacina odbiór).

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

Pełny klient protokołu urządzenia v1.10 (`docs/DEVICE.md`; część v1.10 to rejestracja próbek
opisana niżej) jako komponent zewnętrzny ESPHome
(`firmware/esphome/components/gaja_device/`, C++). Działanie: mikrofon nasłuchuje wake word
(microWakeWord) -> po wykryciu `wake` + pre-roll 1,5 s (pierścień w PSRAM) + żywe ramki
20 ms / 640 B PCM16 16 kHz -> serwer kończy wypowiedź (`listen_end` dla urządzeń bez AEC)
-> mikrofon zwalnia wspólną magistralę I2S, wzmacniacz NS4150B (G18) się włącza, odpowiedź
(PCM16 mono 48 kHz) gra z głośnika -> `follow_up` otwiera mikrofon bez wake worda (od 0.2.5
poprzedzony tym samym beepem co po wake wordzie),
`listen_end` wraca do nasłuchiwania wake worda. Głośność odpowiedzi adaptuje się do poziomu
mowy użytkownika (patrz niżej). Klient łączy się z powrotem z wykładniczym backoffem
(2 s -> 60 s), odpowiada na `ping` aplikacyjny i loguje `error`.

Dlaczego ESPHome, a nie czyste ESP-IDF (plan poniżej): microWakeWord, ES8311, I2S i OTA
działają w ESPHome out-of-the-box i są zweryfikowane na urządzeniu w etapie 1; własna warstwa
to tylko WebSocket + automat stanów. Klient WebSocket to komponent managed
`espressif/esp_websocket_client` (utrzymywany przez Espressif, ws:// i wss://, automatyczne
ping/pong na poziomie protokołu; backoff kontroluje nasz komponent). Yaml włącza dodatkowo
`CONFIG_ESP_WS_CLIENT_SEPARATE_TX_LOCK` (sdkconfig): bez niego task WebSocketu trzyma jedną
blokadę przez cały recv — łącznie z naszym backpressure do 200 ms na pełnym pierścieniu
odtwarzania — i wysyłki z głównej pętli kończą się timeoutem. Negocjacja częstotliwości:
`out_rate` (domyślnie 48000) trafia do `caps.out_rates` w hello i jest wymuszana na głośniku przez
`set_audio_stream_info` przed każdym odtwarzaniem — stream info głośnika i2s_audio domyślnie
wynosi 16 kHz (YAML-owy `sample_rate` go nie ustawia), więc bez tego I2S grałby złą prędkość.
Serwer sam resampluje TTS do tej częstotliwości (scipy polyphase, przetestowana ścieżka serwera)
— urządzenie nie resampluje w ogóle.

### Kompilacja

```bash
cd firmware/esphome
cp secrets.yaml.example secrets.yaml      # uzupełnij Wi-Fi, hasła i dane urządzenia (gitignored)
# models/beep.wav i models/boot_chime.wav dołóż sam (gitignored; boot_chime: make_boot_chime.py)
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
| `preroll_ms` | 1500 | pierścień pre-roll w PSRAM (250..5000; protokół dopuszcza 0..5000, ale ten firmware streamuje też żywe audio przez ten sam pierścień, więc wymuszony jest minimalny rozmiar) |
| `wake_score` | 0.97 | stały score w `wake` — trigger ESPHome nie zwraca prawdopodobieństwa; server używa score tylko do barge-inu (na tym sprzęcie niemożliwego) |
| `quiet_db`, `loud_db` | -24, -12 | poziom mowy w dBFS (p90 RMS ramek 20 ms nad progiem szumu) mapowany liniowo na głośność |
| `vol_min`, `vol_max` | 0.88, 1.0 | zakres głośności odpowiedzi; 1.0 = 0 dB DAC ES8311 (= beep przy 100%); 0.5 zweryfikowane jako niesłyszalne, więc obie granice są ograniczone do <= 1.0 |
| `reply_gap_ms` | 1500 | jak długo czekać na kolejne zdanie tej samej odpowiedzi (amp jeszcze gra) |

Głośność adaptacyjna: p90 z RMS ramek 20 ms (wokół średniej ramki — offset DC się nie liczy) nad
progiem szumu (minimum okna + 6 dB, ale nie niżej niż -60 dBFS — pojedyncza rama cyfrowego zera
nie może przypiąć podłogi na -240 dBFS; cisza między słowami i zero-pad końcówki nie liczą się
do percentyla), `vol = vol_min + clamp((db - quiet_db)/(loud_db - quiet_db)) * (vol_max - vol_min)`.
Pierwsze ~200 ms po restarcie mikrofonu (klik I2S/wzmacniacza) nie liczy się jako mowa, a okno
liczy się jako "była mowa" dopiero od 5 ramek nad progiem (100 ms). Statystyki są czyszczone
przy każdym otwarciu okna mikrofonu, więc zapowiedzi minutników (bez okna) grają z `vol_max`.
Funkcje czyste (`volume_map.h`, `level_stats.h`, `rings.h`, `preroll.h`, `record.h`), testy host-side:
`uv run pytest tests/test_firmware_volume_map.py tests/test_firmware_level_stats.py
tests/test_firmware_rings.py tests/test_firmware_preroll.py tests/test_firmware_record.py -q` (w tym zawijanie pozycji
pierścieni po 2^32 bajtów i czysta funkcja startu pre-rollu). Pozycje pierścieni są 32-bitowe
(size_t na ESP32-S3) i porównywane wyłącznie modularnie; pojemność pierścienia pre-roll jest
zaokrąglana w górę do potęgi dwójki.
Kalibracja na urządzeniu: patrz log INFO `speech level ... dBFS ... -> reply volume ...` i sensor
`Mic RMS` (ten sam gain_factor 4, co słyszy model). Szum tła po wzmocnieniu to ok. -28 dBFS,
normalna mowa z ~1 m ok. -14 dBFS — stąd domyślne -24/-12.

### Rejestracja próbek wake-word (protokół v1.10)

`caps.record: true` w hello zapowiada, że urządzenie umie nagrywać próbki treningowe ("Hej Gaja",
twarde negatywy, ambient) przez własny mikrofon — z UI na serwerze, żeby przetrenować model i
zmierzyć detekcję na urządzeniu w każdej próbce. Bez nowych opcji w yaml; cały przepływ steruje
serwer.

Komunikaty (serwer -> urządzenie):

- `record_start {id, max_ms}` — `id` to `[a-z0-9_-]{1,64}`, `max_ms` 1000..120000. Ważne tylko
  w stanie IDLE (nasłuch wake word, bez tury, bez odtwarzania). Odpowiedź: `record_started {id}`
  i strumień binarnych ramek 640 B (20 ms PCM16 LE mono 16 kHz) żywego mikrofonu od tej chwili
  — **bez pre-rollu** (kursor = głowica pierścienia w chwili startu). W innym stanie:
  `record_error {id, reason}` z `busy`; uszkodzony mikrofon: `mic`; żądanie poza zakresem
  (zły `id`/`max_ms`): `unsupported`; `record_start` podczas nagrywania (nawet z tym samym
  `id`): `busy`.
- `record_stop {id}` — dosłanie niepełnej ramki (uzupełnionej zerami do 640 B), koniec strumienia,
  `record_end {id, ms, reason:"stop"}`, powrót do nasłuchiwania wake word. `record_stop` z
  nieznanym/innym `id` jest ignorowany (z logiem).
- Auto-stop po `max_ms`: `record_end {reason:"max"}`. Watchdog stanu RECORDING: `max_ms` + 2 s
  wymusza powrót do IDLE.

Podczas nagrywania:

- **Żadna rozmowa się nie zaczyna**: detekcja wake word nie wysyła `wake`, tylko
  `record_mark {id, word, t_ms}`, gdzie `t_ms` to milisekundy audio **faktycznie wysłanego**
  w tej próbce w chwili detekcji. Bez beepu. micro_wake_word po detekcji zatrzymuje się sam
  (`stop_after_detection`), a `update_mic_power_()` restartuje go w kolejnym obrocie pętli —
  kolejne wypowiedzi w tej samej próbce też są wykrywane. Mikrofon działa nieprzerwanie przez
  całą próbkę (żadnego restartu, czyli żadnego kliku I2S w audio treningowym).
- Statystyki poziomu (adaptive volume) nie liczą ramek z nagrywania — próbka to dane
  treningowe, nie mowa tury, i nie może wpływać na głośność następnej odpowiedzi.
- Timer/zapowiedź podczas nagrywania: próbka kończy się najpierw (`record_end`, `reason:"stop"`),
  dopiero potem odtwarzanie (półdupleks: odtwarzanie zatrzyma mikrofon).
- Rozłączenie w trakcie: stan nagrywania jest porzucany **bez** `record_end` (serwer zachowuje
  częściową próbkę), po ponownym połączeniu urządzenie wraca do nasłuchiwania.
- Głośnik i wzmacniacz pozostają wyłączone przez całą próbkę.

`record_end.ms` liczy tylko realne bajty mikrofonu (dopełnienie zerami końcowej ramki to
wyrównanie transportu, nie audio). Logika czysta (`record.h`): walidacja `id`/`max_ms` i
przeliczanie bajtów na ms, testy host-side `tests/test_firmware_record.py`.

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
