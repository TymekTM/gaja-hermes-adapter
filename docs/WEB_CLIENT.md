# Gaja Web Client & CLI Tools

Dokumentacja interfejsu przeglądarkowego oraz klienta CLI dla adaptera głosowego Gaja.

---

## 1. Interfejs webowy (`web/`)

Statyczna aplikacja bez etapu budowania (HTML + ES Modules + CSS + dwa AudioWorklety), serwowana przez adapter pod `GET /`.
Główne urządzenie to telefon (iPhone / Android przez Tailscale), ale na komputerze wygląda równie dobrze: szeroki ekran
dostaje układ dwukolumnowy (orb po lewej, rozmowa po prawej).

Zrzuty ekranu każdego stanu (telefon 390×844 @2x, pulpit 1440×900) są w `docs/ui/`, odtwarza je
`clients/ui_screenshots.py` (patrz §5).

### Co jest na ekranie

| Element | Co robi |
|---|---|
| **Orb** (WebGL, z zapasowym Canvas 2D) | Jedyny „żywy” element. Zmienia kolor i ruch w zależności od stanu (słucham / słyszę Cię / myślę / mówię / brak połączenia / błąd), pulsuje wraz z prawdziwym poziomem dźwięku (mikrofon albo głośnik), a emocja zdania wypowiadanego przez Gaję delikatnie barwi całość. Gdy pracuje Hermes, od orba odrywa się mały satelita. Dotknięcie orba w czasie mówienia przerywa Gaję. |
| **Pasek stanu** | Jedno krótkie zdanie: „Gotowa”, „Słucham…”, „Słyszę Cię”, „Myślę…”, „Hermes pracuje w tle”, „Mówię · radośnie”… |
| **Duży przycisk mikrofonu** | Jedna kontrolka do wszystkiego: dotknięcie włącza/wyłącza mikrofon (tryb *wolne ręce*), przytrzymanie = mów (*push-to-talk*). Gdy Gaja mówi lub myśli, zamienia się w **Stop** (przerywa). Pierścień wokół przycisku pulsuje z poziomem mikrofonu. |
| **Rozmowa** | Dymki: Ty po prawej, Gaja po lewej. Odpowiedź Gai powstaje ze zdań (`utterance`; serwer od v1.7 nie wysyła `assistant_delta`, UI nadal toleruje go ze starszych serwerów), a każde zdanie ma stan: *w kolejce → mówione → powiedziane*. Zdanie, które właśnie brzmi, jest podświetlone kolorem swojej emocji (kropka przy zdaniu pokazuje emocję), więc tekst nadąża za głosem, a nie za siecią. Przerwana wypowiedź dostaje znacznik „przerwane”. Gdy mówisz, pojawia się dymek z wyrównywaczem. Gdy Gaja użyje lokalnego narzędzia (zdarzenie `tool`), między pytaniem a odpowiedzią pojawia się mała **plakietka**, np. „🌤 pogoda · Warszawa” (czas, pogoda, obliczenie, przeliczenie, kurs walut, Wikipedia, serwer, najbliższe święto); nieudane narzędzie ma ostrzegawczy kolor i dopisek „nie udało się”. |
| **Karta Hermesa** | Jedna karta na zadanie (klucz = `handoff.id`; kilka zadań naraz to kilka kart), przypięta nad przyciskami: „Hermes pracuje · 12 s” + ostatnia linijka postępu + pasek aktywności. Po dotknięciu rozwija się w oś czasu (zlecono / postęp / wynik / błąd / przerwano). Na działającej karcie jest przycisk **Przerwij** (na wąskim ekranie sama ikona ■ z etykietą dla czytników), który wysyła `{"type":"cancel_task","id":"…"}`; karta zmienia się w „Zadanie przerwane” dopiero, gdy serwer odpowie `handoff{state:"cancelled"}` (tak samo po głosowym „anuluj zadanie” albo `interrupt{scope:"all"}`; po `cancelled` nie przychodzi `done`). **Przerwanie Gai (głosem, tekstem, przyciskiem Stop) nie anuluje zadań** – Hermes pracuje dalej, a wynik Gaja ogłasza jako osobną turę. Karta zostaje, dopóki jej nie zamkniesz (po 90 s znika sama; „przerwane” po 12 s). Błąd rozwija się automatycznie. Ślad jest też w rozmowie. Strumieniowaną odpowiedź Hermesa (fragmenty tekstu) skleja w jedną linijkę. |
| **Powiadomienia serwera** | Zdarzenie `notice` (`stt_degraded` / `stt_recovered`): dyskretny toast + stała pastylka „Mowa: tryb awaryjny” w górnym pasku, dopóki rozpoznawanie mowy działa na silniku zapasowym (pastylka ma `role="status"`, toast informacyjny też jest czytany grzecznie, nie jako alert). Stan zna też `GET /health` (`stt_degraded`), więc pastylka pojawia się także, gdy sesja zaczęła się już w trybie awaryjnym. |
| **Pole tekstowe** | Przycisk klawiatury (telefon) lub zawsze widoczne (pulpit). Tekst idzie jako `{"type":"text"}` i **odtwarza się z dźwiękiem, nawet jeśli mikrofon nigdy nie był włączony**. Na pusty ekran zaproszą też trzy podpowiedzi. |
| **Ustawienia** (koło zębate) | Język rozmowy (pl / en / auto – razem z językiem interfejsu), **głos / silnik TTS** (lista z serwera), tryb mikrofonu (wolne ręce / przytrzymaj), przerywanie głosem, wygląd (systemowy / jasny / ciemny), pokaż metryki, diagnostyka. |
| **Metryki** (opcjonalne) | Cienki pasek: do dźwięku (e2e), STT, LLM TTFT, 1. zdanie, TTS – ostatnia wartość + p50. Przy zdaniach pojawiają się też parametry stylu (tempo, wysokość, energia). Domyślnie schowane. |

Interfejs jest po polsku; po wybraniu `en` (albo `auto` w angielskiej przeglądarce) cały po angielsku
(`web/i18n.js` – słownik, brakujące klucze angielskie wracają do polskich).
Motyw ciemny i jasny podąża za systemem (`prefers-color-scheme`), można go przełączyć na stałe w ustawieniach.
Animacje respektują `prefers-reduced-motion`.

### Skróty (pulpit)

`Spacja` – przytrzymaj i mów · `/` – pole tekstowe · `Esc` – przerwij Gaję / zamknij panel.

### Audio – jak to działa (i dlaczego tak)

- **Odblokowanie dźwięku.** `AudioContext` jest tworzony i wznawiany *synchronicznie* w pierwszym geście użytkownika
  (dotknięcie czegokolwiek, wysłanie tekstu), razem z cichą próbką – tylko tak robi to iOS Safari. Nie zależy od
  mikrofonu, więc odpowiedzi na tekst są słyszalne bez włączania mikrofonu. Po „interrupted” (połączenie, Siri)
  kolejny gest wznawia dźwięk. Na iOS 17+ ustawiamy `navigator.audioSession.type`: `playback` (gra mimo wyciszonego
  przełącznika) i `play-and-record` przy włączonym mikrofonie.
- **Częstotliwość próbkowania.** Każda wypowiedź niesie `audio_start.sample_rate` (Piper 22,05 kHz, Paradee 24 kHz,
  Supertonic 44,1 kHz…). Worklet odtwarzania (`playback-processor.js`) przelicza ją liniowo i ciągle, osobno dla każdej
  wypowiedzi, do częstotliwości kontekstu (bufor kołowy, 80 ms przedbuforu, bez szczelin między zdaniami).
  `ready.sample_rate` to tylko wartość domyślna.
- **Przechwytywanie.** `getUserMedia` z `echoCancellation`, `noiseSuppression`, `autoGainControl`; worklet
  (`capture-processor.js`) miksuje do mono, filtruje (dolnoprzepustowo 7 kHz) i przelicza do **16 kHz PCM16**
  (iOS ignoruje żądaną częstotliwość kontekstu i działa na 44,1/48 kHz), wysyłając ramki po 40 ms.
- **Przerwanie (barge-in).** Serwer wysyła `state: listening` albo VAD `speech_start` → klient **natychmiast** czyści
  bufor odtwarzacza i odrzuca spóźnione ramki (`droppedStale`). Dotknięcie przycisku Stop / orba / Esc / wpisanie
  tekstu robi to samo i wysyła `{"type":"interrupt"}` (działa też, gdy Gaja jeszcze „myśli”).
- **Echo i samoprzerywanie (tryb wolne ręce).** Przy głośniku mikrofon słyszy Gaję. AEC przeglądarki usuwa większość,
  ale resztka (≈ −20…−30 dB) bywa dość głośna, by serwerowy VAD uznał ją za użytkownika i uciął własną wypowiedź
  Gai. Dlatego **dopóki Gaja jest słyszalna** (odtwarzanie, oczekujące zdania, +250 ms), klient wysyła zamiast
  mikrofonu *ciszę* (zerowe ramki – strumień VAD zostaje ciągły), chyba że mikrofon jest wyraźnie głośniejszy
  od tego, co sam odtwarzamy: ≥ 3 kolejne ramki 40 ms powyżej `max(0.03, 0.22 × RMS wyjścia)`. Wtedy bramka się
  otwiera (z ~320 ms „przedwybiegu”, żeby nie uciąć pierwszej sylaby) i trzyma 700 ms po ostatniej głośnej ramce –
  serwer widzi prawdziwą mowę i robi normalny barge-in. W słuchawkach działa bez zarzutu.
  Przełącznik **„Przerywanie głosem”** wyłącza to całkowicie (tryb *half-duplex*: gdy Gaja mówi, mikrofon jest
  zawsze „głuchy”, a przerwać można tylko dotknięciem) – wybierz to, jeśli Gaja nadal słyszy samą siebie.
  W trybie *przytrzymaj* bramki nie ma (naciśnięcie przycisku już przerwało odtwarzanie).
- **Ekran.** W trybie wolne ręce (mikrofon włączony) strona prosi o `wakeLock`, żeby telefon nie zgasł.

### Ciekawostki techniczne

- Zmiana języka lub głosu w ustawieniach wysyła ponownie `hello` w tej samej sesji (historia LLM zostaje); jeśli
  serwer nie odpowie `ready` w 2,5 s – pełne ponowne połączenie.
- Lista głosów pochodzi z `ready.engines` (albo `GET /health` → `engines`); wpisy mogą być napisami albo obiektami
  `{id,label,note,langs}`. Bez listy UI pokazuje podstawowe silniki (`piper`, `paradee`) i mówi o tym.
  Zapamiętany głos, którego serwer już nie zna, jest po cichu zamieniany na „Automatycznie”.
- Ustawienia trzyma `localStorage['gaja.settings']`. Parametry adresu: `?metrics=1`, `?lang=en`, `?theme=light`.
- Połączenie: wykładniczy backoff (500 ms × 1,7ⁿ, max 10 s, ±15 % jitter), przycisk „Teraz”, ponowna próba przy
  powrocie karty / sieci (`visibilitychange`, `online`). Przy zerwaniu uruchomione zadania Hermesa dostają błąd
  „Połączenie zostało przerwane” (nowa sesja nie zna starych `id`).
- Diagnostyka w ustawieniach pokazuje też `stt: <silnik>/<urządzenie>` i `tools on/off` z `GET /health`
  (`stt_engine`, `stt_device`, `stt_degraded`, `router.tools`).
- Metryki `kind: "result"` (ogłoszenie wyniku Hermesa) nie trafiają do paska opóźnień – to nie jest odpowiedź na pytanie.
- Błąd serwera `no running Hermes task …` (kliknięto Przerwij w chwili, gdy zadanie właśnie się skończyło) to
  informacyjny toast „To zadanie już się zakończyło”, bez czerwonego błędu w rozmowie.
- PWA: `manifest.webmanifest` + ikony (`web/icons/`, `icon.svg`), bez service workera (aplikacja i tak wymaga serwera).

---

## 2. Uruchamianie i dostęp

### Lokalnie
```
http://localhost:8790
```
(lub `http://127.0.0.1:8790`). `localhost` jest bezpiecznym kontekstem, więc mikrofon działa.

### Z telefonu przez Tailscale (HTTPS)

Mikrofon (`getUserMedia`) działa tylko w bezpiecznym kontekście, więc z telefonu potrzebny jest HTTPS.
Adapter słucha tylko na `127.0.0.1:8790`; HTTPS zapewnia `tailscale serve` (wystawia go lead / właściciel serwera –
klient sam niczego nie uruchamia).
1. Na serwerze (raz):
   ```bash
   tailscale serve --bg --https=8443 http://127.0.0.1:8790
   ```
   (`tailscale serve status` pokazuje adres; wyłączenie: `tailscale serve --https=8443 off`).
2. Na telefonie (z Tailscale włączonym, w tej samej sieci tailnet) otwórz:
   ```
   https://<nazwa-hosta>.ts.net:8443
   ```
   np. `https://<host>.<twoj-tailnet>.ts.net:8443`. Pełną nazwę zobaczysz w `tailscale status` / `tailscale serve status`.
3. Dotknij mikrofonu i zezwól na dostęp.
4. **Dodaj do ekranu głównego**: iPhone – Safari → Udostępnij → „Do ekranu początkowego”; Android – menu Chrome →
   „Zainstaluj aplikację”. Aplikacja otwiera się wtedy pełnoekranowo, bez paska przeglądarki.

Bez HTTPS (np. `http://100.x.y.z:8790`) strona działa, ale nie ma mikrofonu: interfejs od razu pokazuje to po
polsku razem z komendą `tailscale serve`, a pisanie do Gai nadal działa.

### Rozwiązywanie problemów

| Objaw | Co zrobić |
|---|---|
| „Mikrofon wymaga bezpiecznego połączenia” | Otwórz stronę przez `https://…ts.net:8443` (patrz wyżej) albo `localhost`. |
| „Nie mam dostępu do mikrofonu” | iPhone: Ustawienia → Safari → Mikrofon → „Pytaj/Zezwól” (dla aplikacji z ekranu głównego: Ustawienia → Gaja → Mikrofon). Android/Chrome: kłódka obok adresu → Uprawnienia → Mikrofon. Pulpit: kłódka → mikrofon. Gdy zezwolisz, panel zniknie sam. |
| „Nie widzę mikrofonu” | Podłącz mikrofon; w międzyczasie można pisać. |
| „Mikrofon jest zajęty” | Inna aplikacja (rozmowa, nagrywanie) go używa – zamknij ją i spróbuj ponownie. |
| Brak dźwięku na iPhonie | Sprawdź przełącznik wyciszenia i głośność; dotknij ekranu (iOS pozwala na dźwięk dopiero po geście). Dźwięk przerywa połączenie telefoniczne – kolejne dotknięcie go wznawia. |
| Gaja ucina się, „słyszy samą siebie” | Użyj słuchawek albo wyłącz „Przerywanie głosem” w ustawieniach (tryb half-duplex, przerywasz dotknięciem). Zmniejsz głośność. |
| „Brak połączenia · ponawiam” | Adapter nie działa lub sieć/Tailscale jest wyłączony. Strona próbuje sama z narastającym odstępem; „Teraz” ponawia od razu. |
| Dziwny głos / zła prędkość | Sprawdź wybrany głos w ustawieniach; „Automatycznie” dobiera go do języka. Szczegóły w *Diagnostyka* (częstotliwości, underruny, AEC/NS/AGC). |
| Strona się nie odświeża po aktualizacji | Odśwież ręcznie (adapter wysyła pliki bez cache, ale zainstalowane PWA warto zamknąć i otworzyć). |

---

## 3. Klient CLI (`clients/cli_wav.py`)

Narzędzie wiersza poleceń symulujące mikrofon za pomocą pliku audio `.wav` lub tekstu.

### Składnia:
```bash
.venv/bin/python clients/cli_wav.py [input.wav] [opcje]
```

### Opcje:
- `input.wav` – ścieżka do pliku dźwiękowego wejściowego (dowolna częstotliwość/kanały; automatyczna konwersja do mono 16 kHz PCM16).
- `--url <ws_url>` – adres WebSocket (domyślnie `ws://127.0.0.1:8790/ws`).
- `--lang <pl|en|auto>` – kod języka (domyślnie `pl`).
- `--realtime` – tempo nadawania pakietów w czasie rzeczywistym (co 40 ms); bez tej flagi pakiety wysyłane są natychmiastowo.
- `--out <reply.wav>` – ścieżka do zapisu odebranego dźwięku asystenta.
- `--text "<treść>"` – wysłanie bezpośredniego tekstu zamiast pliku audio.
- `--timeout <sekundy>` – limit czasu oczekiwania na zakończenie tury (domyślnie 30.0s).

### Przykłady:
1. Strumieniowanie pliku audio w czasie rzeczywistym i zapis odpowiedzi:
   ```bash
   .venv/bin/python clients/cli_wav.py mowa.wav --realtime --out odpowiedz.wav
   ```

2. Test tekstowy ze sprawdzeniem metryk:
   ```bash
   .venv/bin/python clients/cli_wav.py --text "Która godzina?"
   ```

---

## 4. Serwer testowy / deweloperski (`clients/mock_server.py`)

Do testowania interfejsu WWW oraz klienta CLI bez prawdziwego adaptera (STT/LLM/TTS):
```bash
.venv/bin/python clients/mock_server.py --port 8799
```
Serwer symuluje:
- pliki statyczne `GET /…`, `GET /health` (z listą silników),
- pełny cykl WebSocket `WS /ws` w kształcie v1.7: hello (także ponowny – zmiana ustawień), ready (z `engines`), vad,
  transcript, utterance, audio binarne (Piper 22,05 kHz / Paradee 24 kHz / Supertonic 44,1 kHz), metrics, state,
  interrupt (`scope:"all"` anuluje zadania), `tool` (np. „pogoda w Warszawie”, „ile to”, „awaria”), `notice`
  (`stt-degraded` / `stt-recovered` wpisane jako tekst), zadania Hermesa w tle z `id`, postępem, `done` albo
  `cancelled`, `cancel_task{id?}` (bez zadania: `error`), `metrics{kind:"result"}`. Barge-in nie rusza zadań.

Scenariusze wybiera się parametrem adresu (klient przekazuje go do `/ws`): `?demo=live|user|thinking|speak|emotions|hermes|barge|stale|error|drop`,
oraz `&emotion=…`, `&pace=realtime`, `&tts=paradee`, `&engines=none` (starszy serwer bez listy silników), `&deltas=1` (wysyła `assistant_delta` jak
serwer sprzed v1.7), `&stt=degraded` (sesja zaczyna się powiadomieniem o trybie awaryjnym), `&hermes_s=N` (czas
zadania; domyślnie 8 s), `?demo=notice`. Pełna lista – nagłówek pliku.

---

## 5. Testy i zrzuty ekranu

```bash
# testy (Chromium z fałszywym mikrofonem odtwarzającym polskie nagranie z bench/data/stt/)
TMPDIR=/tmp .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_web_e2e.py tests/test_cli_client.py

# te testy, które nie potrzebują scenariuszy mocka, na prawdziwym adapterze:
GAJA_E2E_BASE=http://127.0.0.1:8790/ TMPDIR=/tmp .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_web_e2e.py

# prawdziwy adapter na 8790: wpisany tekst, tura z fałszywym mikrofonem, zmiana głosu, Hermes, przerwanie, EN, ponowne
# połączenie (przez własny proxy TCP – adaptera nie dotyka), v1.7: tura z narzędziem (i), zadanie Hermesa przeżywa
# wpisaną turę (j), anulowanie przyciskiem (k), galeria zrzutów docs/ui/real-*.png
TMPDIR=/tmp .venv/bin/python clients/real_adapter_check.py [--only a,b,c,d,e,f,g,h,i,j,k,s]

# zrzuty wszystkich stanów do docs/ui/ (mock na 8799; --base wskazuje inny serwer)
.venv/bin/python clients/mock_server.py --port 8799 &
TMPDIR=/tmp .venv/bin/python clients/ui_screenshots.py [--only speaking-happy,history] [--sizes phone]
```

---

## 6. Weryfikacja na prawdziwym adapterze (2026-10-09)

`clients/real_adapter_check.py` prowadzi Chromium (headless, Playwright) po prawdziwym adapterze `:8790`; mikrofon to plik
WAV (`--use-file-for-fake-audio-capture`, polskie nagranie z `bench/`). Wyniki:

| # | Sprawdzenie | Wynik |
|---|---|---|
| a | Wpisany tekst → dźwięk + transkrypt + podświetlanie zdań, bez włączania mikrofonu | OK (0 underrunów, 0 ramek mikrofonu) |
| b | Tura z mikrofonu (WAV „Zapytaj Hermesa…”) → transkrypt PL, odpowiedź słychać | OK (STT 465 ms, e2e do dźwięku 1,7 s) |
| c | Zmiana głosu w ustawieniach (`hello.tts` → `piper-darkman`, `supertonic`), ta sama sesja | OK (`ready` < 1 s, odpowiedź kolejnym głosem) |
| d | Karta Hermesa („sprawdź w Hermesie, jaki jest dziś dzień tygodnia”): zlecono → postęp → wynik, wynik wypowiedziany | OK (≈ 15–30 s, zależnie od obciążenia) |
| e | Przerwanie dotykiem w trakcie odpowiedzi → bufor pusty < 200 ms, zero dalszych próbek | OK |
| f | Przełączenie na English: interfejs po angielsku, odpowiedź po angielsku głosem `paradee` (24 kHz) | OK |
| g | Restart adaptera (symulowany proxy TCP, które ubijamy i podnosimy) → „Brak połączenia”, po powrocie `ready` w ułamku sekundy, rozmowa działa dalej | OK |
| h | Przerwanie głosem (pętla fałszywego mikrofonu mówi w trakcie długiej odpowiedzi) → `state: listening, reason: barge_in`, odtwarzanie ucięte, dymek „przerwane” | OK |

Błędy znalezione i naprawione w `web/`: (1) rozbite na dziesiątki wierszy fragmenty odpowiedzi Hermesa na osi czasu
karty; (2) karta „Hermes pracuje” wisiała w nieskończoność po przerwaniu (serwer anuluje zadanie bez zdarzenia
`handoff`) – teraz „Zadanie przerwane”; (3) etykieta routingu `safety-net` była pokazywana jako opis zadania;
(4) wynik powtarzał ostatni wiersz postępu.

Zrzuty prawdziwego przebiegu (`docs/ui/`): `real-{phone,desktop}-{dark,light}-{conversation-speaking,conversation,
hermes-working,hermes-done}.png` oraz `real-typed-done`, `real-speaking`, `real-listening`, `real-user-speaking`,
`real-mic-reply`, `real-settings*`, `real-hermes-*`, `real-barge-in`, `real-voice-barge-in`, `real-english`,
`real-offline`.

### Rozwiązywanie problemów z adapterem (widoczne w UI)

| Objaw | Przyczyna |
|---|---|
| Toast „language model unavailable” w środku odpowiedzi | llama-server padł / restartuje się; adapter mówi „Model językowy jest chwilowo niedostępny”. Spróbuj ponownie za chwilę. |
| Karta „Hermes: błąd · ACP request … timed out” | Hermes (ACP) nie odpowiada; to po stronie adaptera/Hermesa (`/health` → `hermes`). |
| Pierwsza tura po restarcie adaptera trwa dłużej | Modele STT/TTS się rozgrzewają. |

---

## 7. Protokół v1.7 w UI (web-final, runda 3, 2026-10-09)

Dostosowanie do zmian rdzenia (round 6b, patrz „Requests” w `docs/ARCHITECTURE.md`):

| Zdarzenie / pole | Co robi UI |
|---|---|
| `state{listening, reason:"barge_in"}` | Czyści odtwarzanie i oznacza dymek „przerwane”. **Nie** rusza kart Hermesa (wcześniej: „Zadanie przerwane” – błąd). |
| `handoff{started/progress/done/error, id}` | Karta wg `id` (kilka naraz). Bez `id` (stary serwer) klucz = tekst zadania. |
| `handoff{state:"cancelled", id, text:""}` | Karta „Zadanie przerwane”, wpis „Przerwano zadanie: …” w rozmowie, bez `done`. |
| `cancel_task{id}` (przycisk **Przerwij**) | Wysyłane z karty; przycisk dostaje „Przerywam…” do czasu `cancelled`. `error: no running Hermes task …` odblokowuje go i pokazuje informacyjny toast. |
| `notice{code, message, engine}` | Toast + pastylka „Mowa: tryb awaryjny” (`stt_degraded`) / toast „znów działa normalnie” (`stt_recovered`). Teksty z `i18n.js` (PL/EN), nieznany `code` → `message` z serwera. |
| `tool{name, arguments, ok, text}` | Plakietka w rozmowie; mówiony tekst przychodzi i tak jako zwykłe `utterance`. |
| `metrics{kind:"result"}` | Pomijane w pasku opóźnień. |
| `assistant_delta` | Serwer go nie wysyła; obsługa zostaje dla starszych serwerów (mock: `?deltas=1`). |
| `GET /health` | `engines` (lista głosów), `stt_degraded` (pastylka), `stt_engine`/`stt_device`/`router.tools` (diagnostyka w ustawieniach). |

### Weryfikacja na prawdziwym adapterze (8790), runda 3

`clients/real_adapter_check.py --only i,j,k` (tylko pytania tylko-do-odczytu; Hermes dostaje wyłącznie „jaki jest dziś
dzień tygodnia / miesiąca”):

| # | Sprawdzenie | Wynik |
|---|---|---|
| i | „jaka jest pogoda w Warszawie” → zdarzenie `tool` (`get_weather`, `location: Warszawa`), plakietka „🌤 pogoda · Warszawa” między pytaniem a odpowiedzią, odpowiedź mówiona | OK (3 z 4 przebiegów w UI; pierwszy poszedł lokalnie – patrz uwaga o routingu w „Requests” w `docs/ARCHITECTURE.md`) |
| j | „Zapytaj Hermesa jaki jest dziś dzień tygodnia” + wpisana tura („dowcip o kocie”) w trakcie pracy | OK: karta została `working` (1,5 s po wpisaniu), brak `cancel_task`/`interrupt`/`cancelled`, zdarzenia `started, progress×4, done`, jedno `id`, wynik „Dziś jest piątek, dziewiątego października.” |
| k | Drugie zadanie („…dzień miesiąca”) anulowane przyciskiem | OK: wysłano `cancel_task{id}`, serwer odpowiedział `handoff{cancelled}` z tym samym `id`, karta „Zadanie przerwane”, przez 9 s żadnego `done` ani błędu w rozmowie |

Zrzuty: `docs/ui/real-tool-chip.png`, `real-tool-chip-start.png`, `real-settings-diag.png`, `real-hermes-running.png`,
`real-hermes-running-typed.png`, `real-hermes-done-v17.png`, `real-hermes-cancel-button.png`, `real-hermes-cancelled.png`.

## 8. Microphone selector (lead, 2026-10-10)
Settings → "Urządzenie wejściowe": the first option, "Domyślny systemowy", stores `mic: ''` and opens the
mic with no `deviceId` (OS default). The list comes from `enumerateDevices()` (Chrome's `default` /
`communications` pseudo-devices are hidden; names appear once permission was granted). The choice is kept
in `localStorage['gaja.settings'].mic`. `AudioController.openMicStream` tries the chosen device, then the
default with AEC/NS/AGC, then a bare `{audio:true}`; if the remembered device is gone it resets to the
default and shows a toast. Changing the device while the mic is open reopens it (the PTT gate is kept).
The list refreshes on `devicechange`. Orb: the shader fades its output over the outer 14% of the canvas,
so the Hermes satellite's halo can't be cut by the canvas edge; the satellite orbit is now 1.32·R0.

### 8.1 "No signal" watchdog (lead, 2026-10-10)

While the mic is open (`armed` or `live`), `app.js` samples the raw input RMS every 250 ms (`audio.getMicRms()`, not gated by
push-to-talk). If it stays at or below `0.0003` (digital silence — a real room is ~1e-3 and up) for 5 s, a `nosignal` notice tells
the user that no sound is reaching the app, names the device in use, and offers "Wybierz mikrofon" (opens settings, focuses
`#set-mic`). It never covers another notice, fires once per mic-open/device-switch, and clears itself as soon as signal appears.

## 9. Wake-word sample recorder (2026-10-10)

`/record.html` (linked from the bottom of the settings sheet: "Nagraj próbki słowa kluczowego") records real "Hej Gaja"
training/eval samples for the custom wake word (microWakeWord pipeline: 16 kHz mono 16-bit WAV). Files: `web/record.html`,
`web/record.js`, the prompt list in `web/record-script.js` (one `SECTIONS` array: id, what to say, how, target count – edit
it freely, but never rename an `id` once recordings exist, it is the folder name), styles `.rec-*` at the end of `style.css`.
Screenshots: `docs/ui/record.png` (desktop, dark), `docs/ui/record-phone-light.png`.

- **Script.** Sections *Pozytywy* (`wake`), *Trudne negatywy* (`hard_negative`, ×3 each) and *Tło* (`ambient`: 6 takes of
  up to 10 min, auto-uploaded in 60 s chunks; an ambient prompt counts as done after 5 min). Per-prompt and overall progress
  come from the server (`GET /api/wakeword/progress`), so reloading resumes the session; "Sesja" lets you start a new one
  or continue an old one.
- **Recording.** Raw microphone: `AudioController.openMicStream(deviceId, { raw: true })` turns echo cancellation, noise
  suppression and AGC off (the page shows what the browser reports, `AEC/NS/AGC off`). Same device list and the same
  setting as the main app (`gaja.settings.mic`). Audio goes through `capture-processor.js` (16 kHz PCM16) and is encoded
  to WAV in the browser. 100 ms are cut from both ends of a take (key clicks), so the speaker should wait ~0.5 s before and
  after speaking. A take auto-stops after 12 s.
- **Keys.** Space = start/stop, Enter = Zapisz, Backspace or R = Powtórz. After stopping, the take is played back (toggle
  "Odtwarzaj po nagraniu"), with a waveform; warnings for silence (peak below -40 dBFS), clipping (peak >= 0.99) and takes
  shorter than 0.8 s. "Cofnij ostatni zapis" deletes the last saved take on the server.
- **Server.** `gaja/wakeword_samples.py` (router included in `create_app`):
  `POST /api/wakeword/samples?session&label&prompt_id&take[&mic]` (raw `audio/wav` body, PCM16 mono 16 kHz, <= 4 MB;
  409 if the take exists), `GET /api/wakeword/progress?session=`, `GET /api/wakeword/sessions`,
  `DELETE /api/wakeword/samples/<session>/<label>/<prompt_id>/<take>`, `GET /api/wakeword/export.zip[?session=]`.
  Stored as `data/wakeword/recordings/<session>/<label>/<prompt_id>/<take:03d>.wav` plus `<session>/meta.jsonl` (ts, label,
  prompt_id, take, duration_s, peak, rms, user agent, mic label; an undo appends a `"deleted": true` line). Path components
  are `[a-z0-9_-]` only. Every endpoint refuses a present-but-unlisted `Origin` (same `server.allowed_origins` as `/ws`).
  Config: `[wakeword] recordings_dir`, `max_total_mb` (507 beyond it). `data/` is git-ignored.
- **To Drive/Colab.** Open `/api/wakeword/export.zip` (or the "Pobierz archiwum" button) and upload the zip; folder names are the
  labels (`wake`, `hard_negative`, `ambient`).
- Tests: `tests/test_wakeword_samples.py` (TestClient). Run by hand with a fake mic: Chromium
  `--use-fake-ui-for-media-stream --use-fake-device-for-media-stream [--use-file-for-fake-audio-capture=x.wav]` against any server that mounts
  the router and `web/`.
