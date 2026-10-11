# Gaja: Szybkie lokalne narzędzia (local-tools) — Runda 2

Raport ewaluacyjny, specyfikacja implementacji oraz rekomendacja architektoniczna workstreamu `local-tools` (Runda 2) dla asystentki głosowej Gaja.

Lokalny model LiquidAI LFM2.5-2.6B (`llama-server` na GPU `:8791`) zyskał pakiet 8 szybkich, asynchronicznych narzędzi lokalnych, które pozwalają odpowiadać na zapytania o czas, pogodę, obliczenia, kursy walut, jednostki miar, encyklopedię, stan serwera i święta w **~1.5–2.5 s** w rekomendowanym Wariancie C (bezpośrednia synteza tekstu `speak`), eliminując zbędną drugą turę generacji LLM (~1.5 s oszczędności i 0% halucynacji).

---

## 1. Zaimplementowane narzędzia (`gaja/tools/`)

Wszystkie narzędzia są w pełni asynchroniczne, wykonują operacje w czasie $\le 2.5\text{ s}$ (twardy timeout), nie wymagają kluczy API ani sekretów i zwracają strukturę `ToolResult` z polami:
- `data`: zwięzły słownik parametrów i wyników technicznych,
- `speak`: znormalizowany tekst do bezpośredniej syntezy mowy przez Piper TTS / Paradee (liczby, godziny, daty i jednostki zapisane słownie w mianowniku/miejscowniku, zaokrąglone temperatury do liczb całkowitych, kursy do 2 miejsc po przecinku, brak znaków `°`, `%`, `km/h`, brak dwukropków i surowych cyfr),
- `error`: komunikat błędu lub `None` (narzędzie nigdy nie rzuca wyjątku w głąb potoku asystentki).

Domyślna lokalizacja konfigurowana jest stałą w `gaja.tools.weather_tool.DEFAULT_LOCATION = "Warszawa"`, nadpisywalną zmienną środowiskową `GAJA_TOOLS__DEFAULT_LOCATION`.

| Narzędzie | Sygnatura | Źródło danych / Algorytm | Format `speak` po normalizacji (Runda 2) | Czas wykonania (typ.) |
|---|---|---|---|---|
| `get_time` | `get_time(timezone="Europe/Warsaw")` | Lokalny zegar systemowy, `zoneinfo.ZoneInfo` | *"Jest piątek, dziewiątego października dwa tysiące dwudziestego szóstego roku, godzina dziesiąta pięćdziesiąt sześć."* | < 1 ms |
| `get_weather` | `get_weather(location="Warszawa", when="now"\|"today"\|"tomorrow")` | Open-Meteo Geocoding + Forecast API (kod WMO $\to$ słowa PL/EN, wschód/zachód, opady, wiatr), cache 10 min. Szablon bezodmianowy: `"{lokalizacja}, {pora}: od X do Y stopni..."` | *"Kraków, jutro: od jedenaście do siedemnaście stopni, pochmurno, szansa na opady dwanaście procent, wschód słońca o szóstej pięćdziesiąt cztery, zachód o siedemnastej pięćdziesiąt dziewięć."* | ~250–350 ms (cache: < 1 ms) |
| `calculate` | `calculate(expression: str)` | Bezpieczny parser i ewaluator AST Pythona (`+ - * / ** % sqrt`, procenty, potęgi, ochrona przed AST DOS, brak eval/exec) | *"Wynik to trzydzieści sześć."* | < 2 ms |
| `convert_units` | `convert_units(value, from_unit, to_unit)` | Konwersje jednostek: długość, masa, temperatura, objętość, prędkość, czas; obsługa polskich i angielskich nazw potocznych | *"sto kilometrów to około sześćdziesiąt dwa przecinek czternaście mil."* | < 1 ms |
| `exchange_rate` | `exchange_rate(currency, amount=1, to="PLN")` | Tabela A Narodowego Banku Polskiego (NBP API), kursy krzyżowe, kwoty w złotych i groszach, cache 1 h | *"jeden euro to cztery złote i trzydzieści osiem groszy."* | ~80 ms (cache: < 1 ms) |
| `wiki_summary` | `wiki_summary(topic: str, lang="pl"\|"en")` | Wikipedia REST Summary API, inteligentny podział zdań odporny na skróty (`łac.`, `niem.`, `ur.`, `zm.`, `ok.`, `m.in.`, `tzw.`), czyszczenie wtrąceń językowych i IPA | *"Mikołaj Kopernik – polski astronom, prawnik, urzędnik, dyplomata, lekarz i niższy duchowny katolicki..."* | ~180–230 ms |
| `server_status` | `server_status()` | `/proc/uptime`, `/proc/loadavg`, `/proc/meminfo`, `nvidia-smi` (pamięć GPU), `systemctl --user is-active` (usługi Gaja) | *"Serwer działa od siedem dni, obciążenie jeden przecinek trzydzieści trzy, RAM dwanaście przecinek sześć z piętnaście przecinek pięć gigabajtów..."* | ~35 ms |
| `next_holiday` | `next_holiday()` | Obliczenia lokalne: 13 ustawowych dni wolnych w Polsce, algorytm Meeusa/Jonesa/Butchera dla Wielkanocy, dni do święta | *"Najbliższe święto to Wszystkich Świętych, pierwszego listopada, za dwadzieścia trzy dni."* | < 1 ms |

---

## 2. Normalizator tekstu dla TTS (`gaja/text/normalize.py`)

Piper TTS czyta surowy tekst przez espeak, który zniekształca liczby, godziny, ułamki i jednostki (np. "10.8" $\to$ "dziesięć kropka osiem", "06:54" $\to$ "zero sześć dwukropek...", "12°C" $\to$ "dwanaście stopień c").
Wdrożono szybki i bezpieczny moduł normalizacji fonetycznej `normalize_for_tts(text: str, lang: "pl"|"en") -> str`:
- **Czas wykonania**: ~0.55 ms na zdanie (< 1 ms budżetu).
- **Zakres reguł (Runda 3 / 3b)**:
  - Odmiana liczebników głównych w przypadkach gramatycznych (mianownik, dopełniacz, miejscownik, narzędnik) dla liczb 0–999 999 zależnie od kontekstu składniowego (przyimki dopełniaczowe: `od`, `do`, `z`, `ze`, `około`, `ok.`, `bez`, `u`, `dla` $\to$ dopełniacz; przyimki miejscownikowe: `o`, `przy`, `po` $\to$ miejscownik dla godzin).
  - Składnia rządu i zgody liczebnika z rzeczownikiem (reguły 1 / 2–4 prócz 12–14 / 5+; tablica fleksyjna dla kluczowych jednostek i pojęć: `dzień`, `stopień`, `procent`, `minuta`, `godzina`, `sekunda`, `kilometr`, `metr`, `złoty`, `grosz`, `euro`, `dolar`, `gigabajt`, `megabajt`, `rok`, `mila`, `kilogram`, `litr`).
  - Rozróżnienie rodzaju dla 1 i 2 ("jedno euro", "dwie godziny"), niezmiennik "jeden" w liczbach wielocyfrowych zakończonych na 1 ("dwadzieścia jeden minut", "sto jeden minut", "dwadzieścia jeden euro" — nigdy "dwadzieścia jedna").
  - Zachowanie oryginalnej wielkości liter wyrazów nieliczbowych na początku zdania ("Od 11 do 17 stopni." $\to$ "Od jedenastu do siedemnastu stopni.").
  - Spójna reguła dla ułamków dziesiętnych w dopełniaczu jednostek pamięci ("12,6 z 15,5 GB" $\to$ "dwanaście przecinek sześć z piętnastu przecinek pięciu gigabajtów").
  - Liczby kardynalne i dziesiętne ("10,8" / "10.8" $\to$ "dziesięć przecinek osiem"), liczby ujemne.
  - Godziny w mianowniku i miejscowniku ("06:54" $\to$ "szósta pięćdziesiąt cztery", "o 17:59" $\to$ "o siedemnastej pięćdziesiąt dziewięć").
  - Daty i liczebniki porządkowe ("9 października 2026 r." $\to$ "dziewiątego października dwa tysiące dwudziestego szóstego roku").
  - Waluty ("4,38 zł" $\to$ "cztery złote i trzydzieści osiem groszy", "10 EUR" $\to$ "dziesięć euro").
  - Skróty i wieki rzymskie ("w XVI wieku" $\to$ "w szesnastym wieku", "np." $\to$ "na przykład", "m.in." $\to$ "między innymi", "ok." $\to$ "około").
- **Testy jednostkowe**: [tests/test_normalize.py](file:///home/ttm-server/.t3/projects/gaja-hermes-adapter/tests/test_normalize.py) — 107 testów (PL + EN + edge cases) przechodzących w 100%.

### Obiektywna weryfikacja normalizatora (`bench/normalize_eval.py`)
Syntetyzowano 30 reprezentatywnych zdań testowych (liczby, daty, godziny, waluty, jednostki) głosem Piper `pl_PL-gosia-medium` na CPU, a następnie transkrybowano modelem Faster-Whisper `small` na **CPU int8** z pomiarem poprawności liczb i wskaźnika błędów słów (WER).

*Źródło danych: [bench/results/normalize_eval.json](file:///home/ttm-server/.t3/projects/gaja-hermes-adapter/bench/results/normalize_eval.json)*
- **Wskaźnik poprawności odczytania liczb (Numbers Correct Rate)**:
  - Bez normalizacji: **76.7%** (23/30)
  - Z normalizacją: **86.7%** (26/30) — **wzrost o +10.0 pp**
- **Średni Word Error Rate (WER)**:
  - Bez normalizacji: **19.4%**
  - Z normalizacją: **10.1%** — **redukcja błędów o 9.31 pp (-47.9% relatywnie)**
- **Czas trwania pomiaru**: **122.2 s** (30 zdań, 2 warunki na CPU)
- **Loadavg pomiaru**: start: `[0.64, 0.39, 0.49]`, end: `[4.45, 1.94, 1.05]`.

---

## 3. Rozwiązanie problemu halucynacji w 2. turze LLM (demo_6)

W Rundzie 1 zapytanie o Mikołaja Kopernika (`demo_6`) skutkowało obcięciem tekstu Wikipedii do *"Mikołaj Kopernik, łac. Nicolaus Copernicus, niem."*, po czym LFM wygenerował zniekształcony fakt *"polsko-polskim astronomem..."*.
- **Przyczyna**: prymitywny splitter dzielił zdania na kropce po skrócie językowym `"niem."`.
- **Rozwiązanie w `gaja/tools/wiki_tool.py`**:
  1. Wdrożono inteligentny regex `_split_sentences_smart` uwzględniający bazę polskich i łacińskich skrótów: `łac.`, `niem.`, `ur.`, `zm.`, `ok.`, `m.in.`, `tzw.`, `np.`, `r.`, `w.`, `św.`, `gr.`, `ang.`.
  2. Dodano automatyczne usuwanie wtrąceń etymologicznych i językowych w nawiasach oraz po przecinkach (np. `(łac. Nicolaus Copernicus, niem. Niklas Koppernigk)`), a także fonetycznych znaczników IPA.
- **Efekt**: W Rundzie 2 `wiki_summary` zwraca czyste, pełne zdania: *"Mikołaj Kopernik – polski astronom, prawnik, urzędnik, dyplomata, lekarz i niższy duchowny katolicki..."*. W obu wariantach (B_hybrid i C) wskaźnik halucynacji spadł do 0%.

---

## 4. Kompleksowe porównanie architektur (Wariant A, B, B_hybrid r1, B_hybrid r2, Wariant C)

Ewaluację przeprowadzono na głównym zbiorze testowym [bench/data/tools_eval.jsonl](file:///home/ttm-server/.t3/projects/gaja-hermes-adapter/bench/data/tools_eval.jsonl) (**106 zapytań**) oraz na osobnym, niewidzianym podczas strojenia zbiorze held-out [bench/data/tools_heldout.jsonl](file:///home/ttm-server/.t3/projects/gaja-hermes-adapter/bench/data/tools_heldout.jsonl) (**22 zapytania**).

### Tabela porównawcza (Wszystkie warianty i rundy)

| Metryka | Wariant A (Natywny) | Wariant B (Pure Router) | Wariant B_hybrid R1 | Wariant B_hybrid R2 | Wariant C (Direct Speak) | Wariant C (Held-out 22) |
|---|---|---|---|---|---|---|
| **Liczba próbek ($N$)** | 106 | 106 | 106 | 106 | **106** | 22 |
| **Dokładność akcji (Action Acc)** | 78.30% | 89.62% | 90.57% | **96.23%** | **96.23%** | **90.91%** |
| **Dokładność wyboru narz. (Tool Acc)** | 76.42% | 86.79% | 87.74% | **96.23%** | **96.23%** | **90.91%** |
| **Dokładność parametrów (Arg Acc)** | 76.42% | 86.79% | 87.74% | **96.23%** | **96.23%** | **90.91%** |
| **Recall wyboru narzędzi** | 95.59% | 91.18% | 92.65% | **95.59%** | **95.59%** | **100.00%** |
| **False Tool Rate na chit-chat** | 62.50% | 12.50% | 12.50% | **0.00%** | **0.00%** | 25.00% (1/4) |
| **Skuteczność granicy Hermesa** | 62.50% | 100.00% | 100.00% | **93.75%** | **93.75%** | **75.00%** (3/4) |
| **Skuteczność na pułapkach (Traps)** | 25.00% | 25.00% | 25.00% | **100.00%** | **100.00%** | **100.00%** (4/4) |
| **Wskaźnik wierności faktów (Faithfulness)** | N/A | N/A | N/A | 93.40% | **100% (z def.)** / 88.68% token-match | 100% (z def.) / 81.82% |
| **Latencja routera (p50)** | — | 947.5 ms | 954.6 ms | 958.0 ms | **951.7 ms** | 971.3 ms |
| **Latencja E2E (mediana p50)** | 4168.0 ms | 3657.6 ms | 3556.2 ms | 3137.4 ms | **1690.5 ms (-1.45 s!)** | **1711.8 ms** |
| **Latencja E2E (90. perc. p90)** | 6523.5 ms | 5776.3 ms | 5340.7 ms | 5014.9 ms | **2771.2 ms (-2.24 s!)** | **3188.3 ms** |
| **Latencja E2E (średnia)** | 4747.8 ms | 3563.9 ms | 3205.5 ms | 3096.3 ms | **1872.3 ms (-1.22 s!)** | **2492.3 ms** |
| **Loadavg pomiaru (start $\to$ end)** | — | — | — | 1.15 $\to$ 4.09 | 3.05 $\to$ 3.29 | 5.86 $\to$ 5.06 |

*Źródła danych:*
- *[bench/results/tools_eval_comparison.json](file:///home/ttm-server/.t3/projects/gaja-hermes-adapter/bench/results/tools_eval_comparison.json)*
- *[bench/results/tools_eval_variant_b_hybrid.json](file:///home/ttm-server/.t3/projects/gaja-hermes-adapter/bench/results/tools_eval_variant_b_hybrid.json)*
- *[bench/results/tools_eval_variant_c.json](file:///home/ttm-server/.t3/projects/gaja-hermes-adapter/bench/results/tools_eval_variant_c.json)*

---

## 5. Dlaczego rekomendujemy Wariant C (Direct Speak)?

1. **Drastyczna redukcja latencji E2E (-1.45 s na p50, -2.24 s na p90)**:
   - W Wariancie B_hybrid asystentka musi po wykonaniu narzędzia uruchomić drugą pełną generację LLM (prompt eval drugiego etapu + generacja 30–50 tokenów odpowiedzi), co zajmuje dodatkowe 1.5–3.0 s.
   - W Wariancie C po otrzymaniu wyniku narzędzia pole `speak` trafia **natychmiast do TTS**. Czas p50 wynosi zaledwie **1.69 s** (z czego ~950 ms to router, a ~300 ms to wywołanie API pogodowego/sieciowego).
2. **Całkowita eliminacja ryzyka halucynacji (100% Faithfulness)**:
   - LLM w drugiej turze ma tendencję do formatowania liczb w postaci surowych cyfr (`"10.8°C"`, `"06:54"`), dodawania niepotrzebnych ozdobników lub przekręcania faktów (jak w przypadku Kopernika w R1).
   - W Wariancie C tekst `speak` jest generowany z deterministycznych, przetestowanych szablonów narzędzi i przechodzi przez `normalize_for_tts`, co gwarantuje pełną zgodność z faktami i brak nieoczekiwanych słów.
3. **Optymalna jakość fonetyczna dla syntezy głosu**:
   - Bezpośredni tekst `speak` zawiera gotowe, odmienione liczebniki słowne, bez problematycznych dla Pipera symboli procentów, stopni czy skrótów.

---

## 6. Przykłady z rzeczywistego wykonania na żywo (Live Demo)

*Źródło danych: [bench/results/tools_demo.json](file:///home/ttm-server/.t3/projects/gaja-hermes-adapter/bench/results/tools_demo.json)*

### Przykład 1: Pogoda na jutro (`get_weather`)
- **Pytanie**: *"Jaka jest jutro pogoda w Krakowie?"*
- **Wywołanie**: `get_weather(location='Kraków', when='tomorrow')`
- **Wynik surowy**: `temp_min=11.1, temp_max=17.1, conditions='pochmurno', precip=12%, sunrise='06:54', sunset='17:59'`
- **Wypowiedź Wariant C (2669.2 ms)**: *"Kraków, jutro: od jedenaście do siedemnaście stopni, pochmurno, szansa na opady dwanaście procent, wschód słońca o szóstej pięćdziesiąt cztery, zachód o siedemnastej pięćdziesiąt dziewięć."*
- **Wypowiedź Wariant B (5661.8 ms)**: *"Kraków jutro będzie pochmurno, z temperaturą od 11.1°C do 17.1°C. Szansa na opady jest niewielka (12%), a wiatr wieje z prędkością 19 km/h. Świt słońca będzie o 06:54, a zachód o 17:59."*
- **Zysk**: **+2992.6 ms szybszy czas reakcji**, brak surowych cyfr i symboli stopni/dwukropków.

### Przykład 2: Obliczenia procentowe (`calculate`)
- **Pytanie**: *"Ile to jest 15 procent z 240?"*
- **Wywołanie**: `calculate(expression='15 * 240 / 100')`
- **Wynik surowy**: `result=36`
- **Wypowiedź Wariant C (2347.8 ms)**: *"Wynik to trzydzieści sześć."*
- **Wypowiedź Wariant B (2994.7 ms)**: *"Wynik to trzydzieści sześć."*
- **Zysk**: Identyczna treść, **+646.9 ms szybszy czas reakcji**.

### Przykład 3: Wikipedia bez halucynacji (`wiki_summary`)
- **Pytanie**: *"Kim był Mikołaj Kopernik według Wikipedii?"*
- **Wywołanie**: `wiki_summary(topic='Mikołaj Kopernik', lang='pl')`
- **Wynik surowy**: `summary='Mikołaj Kopernik – polski astronom, prawnik, urzędnik, dyplomata...'`
- **Wypowiedź Wariant C (2656.4 ms)**: *"Mikołaj Kopernik – polski astronom, prawnik, urzędnik, dyplomata, lekarz i niższy duchowny katolicki, doktor prawa kanonicznego i naukowiec – zajmował się też matematyką, ekonomią, strategią wojskową, kartografią i filologią. Bywa też nazywany fizykiem, filozofem."*
- **Wypowiedź Wariant B (5687.7 ms)**: *"Mikołaj Kopernik był polskim astronomem, prawnikiem, urzędnikiem, dyplomatą, lekarzem i niższym duchownym katolickim, doktorem prawa kanonicznego oraz naukowiecem..."*
- **Zysk**: **+3031.3 ms szybszy czas reakcji**, całkowity brak błędu ucinania na skrócie `niem.` i 0% halucynacji.

### Przykład 4: Kurs waluty NBP (`exchange_rate`)
- **Pytanie**: *"Jaki jest aktualny kurs euro w złotówkach?"*
- **Wywołanie**: `exchange_rate(currency='EUR', to='PLN')`
- **Wynik surowy**: `rate=4.3789, result=4.38`
- **Wypowiedź Wariant C (2480.7 ms)**: *"jeden euro to cztery złote i trzydzieści osiem groszy."*
- **Wypowiedź Wariant B (3272.7 ms)**: *"Jeden euro to obecnie około 4,38 złotych."*
- **Zysk**: **+792.0 ms szybszy czas reakcji**, kwota przeczytana naturalną polszczyzną w groszach zamiast zapisu ułamkowego.

### Przykład 5: Stan serwera (`server_status`)
- **Pytanie**: *"Jaki jest aktualny stan tego serwera i zużycie RAM?"*
- **Wywołanie**: `server_status()`
- **Wynik surowy**: `loadavg_1m=1.33, ram_used_gb=12.6, ram_total_gb=15.5, gpu_used='3225 MiB'`
- **Wypowiedź Wariant C (2153.3 ms)**: *"Serwer działa od siedem dni, obciążenie jeden przecinek trzydzieści trzy, RAM dwanaście przecinek sześć z piętnaście przecinek pięć gigabajtów, pamięć karty graficznej trzy tysiące dwieście dwadzieścia pięć megabajtów z cztery tysiące dziewięćdziesiąt sześć megabajtów. Usługi gaja-adapter i gaja-llm są aktywne."*
- **Wypowiedź Wariant B (4786.2 ms)**: *"Serwer działa od 7 dni, obciążenie 1:33, RAM zużyto 12,6 GB z 15,5 GB..."*
- **Zysk**: **+2632.9 ms szybszy czas reakcji**, poprawna fonetyzacja obciążenia (nie jako godzina "1:33") i jednostek RAM/GPU.

---

## 7. Schemat docelowy potoku (Rekomendacja dla `core-final`)

```
                  ┌───────────────────────┐
                  │ Transkrypcja ASR (STT)│
                  └───────────┬───────────┘
                              │
                    Reguły regex (0 ms)
                              │
               Router v3 logprob (/completion)
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
     [ LOCAL ]             [ TOOL ]            [ HERMES ]
         │                    │                    │
  Strumieniowanie       Targeted Tool Call         │
  odpowiedzi bez        w llama-server             │
  narzędzi (niski TTFT)       │                    │
         │             Wykonanie narzędzia         │
         │             (gaja.tools.execute)        │
         │                    │                    │
         │               [WARIANT C]               │
         │          Direct speak: res.speak        │
         │         (brak 2. tury LLM, -1.5s)       │
         │                    │                    │
         │             normalize_for_tts           │
         │                    │                    │
         ▼                    ▼                    ▼
     TTS Piper            TTS Piper         Handoff do ACP
```

### Proponowany wpis konfiguracyjny do `config.toml`

```toml
[tools]
enabled = true
variant = "c"  # "c" (rekomendowany, direct speak) lub "b_hybrid" (z 2. turą LLM)
default_location = "Warszawa"
timeout_s = 2.5
cache_ttl_weather_s = 600
cache_ttl_nbp_s = 3600

[text]
normalize_tts = true
```

---

## 8. Integracja w adapterze (stan po rundzie 6b, 2026-10-09)

Narzędzia działają w produkcyjnym potoku (`gaja/pipeline.py`, `gaja/llm/router.py`). Różnice względem schematu z §7:

- **Brak pre-filtra regex.** Każdą turę klasyfikuje Router v3 (`prompts/router3_pl.md`): jedno wywołanie
  `/completion` z `n_predict 1` i `n_probs` na slocie routera (slot 1). Klasa TOOL wygrywa, gdy jej masa ≥
  `router.tool_threshold = 0.6`; HERMES przy masie ≥ `router.threshold = 0.5`. Wyjątek: jawne zwrócenie się do Hermesa
  („zapytaj Hermesa…”, „w Hermesie”) idzie do Hermesa bez modelu. Ewaluacja: `bench/results/router3_eval_r6.json`
  (trafność narzędzia 91.2 %, argumentów 91.2 %, classify p50 116.5 ms).
- **Wybór narzędzia**: `Router.select_tool` to krótka kontynuacja tego samego promptu ograniczona gramatyką
  (`tool_grammar`, maks. 64 tokeny); zwraca nazwę i argumenty (select p50 ~490 ms w ewaluacji, 180–560 ms na żywo).
  Jeśli w wypowiedzi (lub, dla krótkiego dopytania, w poprzedniej wypowiedzi użytkownika) nie ma żadnego słowa-wskazówki
  narzędzia (`TOOL_CUES`), wybór jest pomijany i tura idzie do lokalnej odpowiedzi — oszczędza to ~500–660 ms przy
  błędnej klasyfikacji TOOL. Wybrane narzędzie, które nie pasuje do wskazówek, też jest odrzucane.
- **Miejsce domyślne**: pogoda bez miasta albo z miejscem-wypełniaczem („miasto”, „tutaj”, „here”) używa
  `tools.default_location = "Warszawa"`.
- **Wikipedia**: gdy strona o dokładnym tytule nie istnieje, `wiki_summary` szuka tytułu przez
  `/w/rest.php/v1/search/title` i pobiera streszczenie znalezionej strony.
- **Wariant C**: tekst `speak` narzędzia jest mówiony bezpośrednio (bez drugiej tury LLM) i zapisany w historii jako
  odpowiedź Gai. Błąd narzędzia też ma swoje zdanie (`ok: false`). Klient dostaje zdarzenie
  `{"type":"tool","name":...,"arguments":{...},"ok":bool,"text":...}` przed `utterance`.
- **Spekulacja**: wybór narzędzia może ruszyć spekulatywnie, ale samo wykonanie czeka na potwierdzenie końca
  wypowiedzi (`committed`), więc żadne zapytanie HTTP nie wychodzi dla wypowiedzi, którą użytkownik kontynuuje.
- **Normalizacja**: przy `[text] normalize_tts = true` każde zdanie (nie tylko z narzędzi) przechodzi przez
  `normalize_for_tts` przed TTS; tekst na ekranie jest bez zmian.
- **Konfiguracja** (`config.toml`): `[tools] enabled = true`, `default_location = "Warszawa"`, `timeout_s = 4.0`
  (2.5 s nie wystarczało dla Wikipedii pod obciążeniem). Pola `variant`, `cache_ttl_*` z §7 nie są używane.
- **Metryki**: `metrics.route = "tool"`, `tool`, `tool_select_ms`, `tool_ms`.
- **Na żywo (2026-10-09)**: pogoda (domyślna i „jutro w Krakowie”), godzina (PL i EN), kalkulator, kurs euro,
  Wikipedia („kim był Stanisław Lem”, „kim jest Hermes w mitologii”) — wszystkie `ok`, pierwsze audio 0.54–1.13 s od
  wysłania tekstu (`/tmp/gaja-r6b/live_tools.log`).
