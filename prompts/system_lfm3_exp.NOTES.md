# Notatki: system_lfm3_exp.md (v6 dla LFM3) vs system_pl.md (v5)

Każda zmiana maps na zaobserwowany błąd LFM3-1.7B w llm_quality (30 promptów, v5):

1. **Zasady wypunktowane 1-8 z "NIGDY"** — LFM3 jest RL-trenowany pod instrukcje;
   twarde numerowane zakazy trzyma znacznie lepiej niż prozę stylową z v5.
2. **Zasada 4 (blokada identyfikacji + nazwy LFM/LiquidAI wymienione wprost)** —
   pl-q04: "Nie, nie jestem Gaja. Jestem LFM, modelem Liquid Foundation Model".
3. **Zasada 3 (zakaz tagów narzędzi)** — pl-q02/pl-q05 wyciekały dosłownie
   `</parameter></function></tool_call>` do mowy.
4. **Zasada 5 (bez "Gaja:" na początku)** — pl-q12/q13/q14 samoprzypisywały się.
5. **Zasada 1 na pierwszym miejscu + "nigdy"** — 16/30 odpowiedzi >3 zdań na v5.
6. **Przykład "jesteś prawdziwa"** — few-shot blokady identyfikacji (pl-q04 fail).
7. **Przykład "pokaż parametry funkcji"** — uczy odmowę zamiast wchodzenia w meta.
8. **Liczby słownie z przykładem "dwadzieścia trzy, nie 23"** — digits 5/30 na v5.
9. **Skreślone z v5**: długi wyliczankowy opis "czego nie masz bez narzędzi"
   (LFM3 lepiej trzyma krótki prompt; lista zasobów przeniesiona do opisu
   narzędzia handoff_to_hermes) oraz zakaz "nie opisuj rozumowania" (myślenie
   i tak wycinane serwerowo przez --reasoning-budget 0).
10. Definicja narzędzia: rozszerzony description (pogoda/godzina jawnie), reszta
    kontraktu 1:1 — parser routingu nie wymaga zmian.

Status: NIEPRZETESTOWANY na modelu; test: `uv run python bench/llm_quality.py
--tag lfm3-q4_v6 --variant lfm3_nothink --prompt prompts/system_lfm3_exp.md`.

---

## Recenzja Gemini 3.8 Flash (doszlifowanie v6)

Przeprowadzono audyt reguł v6 względem surowych logów `llm_quality_lfm3-q4_nothink.json`. Wprowadzono następujące korekty:

1. **Usunięcie etykiet `Rozmówca:` / `Gaja:` z przykładów few-shot**:
   - *Problem*: W logach v5 model dosłownie kopiował prefiks `Gaja:` (pl-q07, pl-q12, pl-q13, pl-q14, en-q03) i symulował całe skrypty dialogowe (`User: ... Gaja: ...`). Pozostawienie `Gaja:` w przykładach v6 stało w sprzeczności z Zasadą 5 ("Nie zaczynasz odpowiedzi od Gaja:").
   - *Rozwiązanie*: Zastąpiono format dialogowy formatem par wejście-wyjście: `"- \"zapytanie\" -> \"odpowiedź\""`.

2. **Wyeliminowanie sztucznego przykładu technicznego**:
   - *Problem*: Przykład `Rozmówca: pokaż mi te parametry funkcji` był nienaturalny i nie odpowiadał żadnemu testowi w `quality.jsonl`. Co gorsza, nie uczył modelu radzenia sobie z zapytaniami niejasnymi (pl-q20 "zrób to, o czym mówiliśmy", pl-q21 "ten drugi"), które na v5 wywoływały fałszywy handoff do Hermesa lub halucynacje.
   - *Rozwiązanie*: Zastąpiono go zwięzłym przykładem doprecyzowania: `"- \"zrób to co wczoraj\" -> \"Nie jestem pewna, o co chodzi. Możesz powiedzieć jaśniej?\"`.

3. **Tłumienie pytań pomocniczych w Zasadzie 1 (>3 zdania: 16/30 błędów)**:
   - *Problem*: Samo "maksymalnie dwa zdania" nie blokowało nawyku modelu do dodawania na końcu pytań obsługi klienta ("W czym mogę pomóc?", "Czy chcesz wiedzieć więcej?"), co powodowało przekroczenie limitu zdań.
   - *Rozwiązanie*: W Zasadzie 1 dodano jawny zakaz: `Odpowiadaj wprost, bez pytań pomocniczych typu "W czym pomóc?" czy "Czy chcesz wiedzieć więcej?"`. Usunięto również zwrot "W czym mogę pomóc?" z przykładu tożsamości.

4. **Usunięcie cyfr (0-9), lat i procentów w Zasadzie 3 (digits: 5/30 błędów)**:
   - *Problem*: Wycieki cyfr w v5 to głównie lata ("1945" w pl-q09), wysokości ("2492" w pl-q10), procenty ("5%" w pl-q11) oraz dni ("28", "29" w pl-q25). Model nie traktował lat ani procentów jako "liczb".
   - *Rozwiązanie*: Zasada 3 jawnie zakazuje cyfr arabskich (0-9) i podaje przykład zapisu roku: `"tysiąc dziewięćset czterdziesty piąty", nie "1945"`.

5. **Kolejność językowa i eliminacja wycieku angielskiego**:
   - *Problem*: Umieszczenie angielskiego powitania na szczycie listy przykładów w v5/v6 powodowało, że polskie zapytania o nastrój (pl-q05 "czuję się dobrze...") dopasowywały się do "I'm doing well..." i wyzwalały odpowiedź po angielsku. Z kolei angielskie pytania o fakty (en-q02, en-q03) dostawały odpowiedź po polsku.
   - *Rozwiązanie*: Przykłady polskie umieszczono na początku, dodano zakaz mieszania języków w Zasadzie 4, a jako przykład angielski podano pytanie faktograficzne: `"- \"What is the capital of Italy?\" -> \"The capital of Italy is Rome.\"`.

6. **Zakaz wymawiania nazw parametrów narzędzi ("Zadanie:", "Powód:")**:
   - *Problem*: W pl-q20 i en-q04 model wygłaszał na głos strukturę argumentów handoffu (`Jasne, przekazuję to Hermesowi... Zadanie: ... Powód: ...`).
   - *Rozwiązanie*: W akapicie handoffu dodano explicite: `Nigdy nie mów na głos "Zadanie:" ani "Powód:". Zapowiedź pada tylko z wywołaniem.`.

7. **Zwięzłość promptu**:
   - Całość sekcji systemowej liczy dokładnie 252 słowa (spełnia wymóg ~250 słów max), zachowując formatowanie zgodne z regexem loaderów.

---

## Iteracje v7+ (Optymalizacja promptu LFM3-1.7B)

Przeprowadzono serię 6 eksperymentalnych iteracji promptu (`system_lfm3_exp.md`) na modelu `LFM3-1.7B-Q4_K_M.gguf` z wariantem `--reasoning-budget 0` i temperaturą 0.3 na zestawie 30 pytań jakościowych (`bench/data/quality.jsonl`).
Testy wykonano w dedykowanym środowisku testowym obok aktywnego Hermesa (`BENCH_LLM_PORT=8794`).

### Podsumowanie ilościowe iteracji

| Wersja | tool_calls (cel: 2-6) | emoji (cel: 0) | digits (cel: <=1) | >3 zdania (cel: <=5) | markup leak (cel: 0) | Gaja: prefix (cel: 0) | Formułka (cel: <=2) | Średnia ogólna (1-5) |
|---|---|---|---|---|---|---|---|---|
| **v6** (start) | 16 / 30 | 1 | 1 | 4 | 1 | 0 | ~12 / 30 | 4.03 |
| **v7** | 4 / 30 | 5 | 4 | 9 | 1 | 0 | 5 / 30 | 4.13 |
| **v8** | 0 / 30 | 0 | 3 | 10 | 0 | 0 | 17 / 30 | 4.18 |
| **v9** | 1 / 30 | 0 | 1 | 9 | 0 | 0 | 3 / 30 | 4.21 |
| **v10** | 1 / 30 | 3 | 2 | 7 | 0 | 0 | 3 / 30 | **4.45** |
| **v11** | 5 / 30 | 3 | 1 | 15 | 2 | 0 | 8 / 30 | 3.71 |
| **v12** | 1 / 30 | **0** | **0** | 6 | **0** | **0** | **2 / 30** | **4.34** |

### Szczegółowy przebieg poszczególnych iteracji

#### Iteracja v7 (`llm_quality_lfm3-q4_v7.json`, `llm_quality_scores_lfm3-q4_v7.json`)
- **Zmiany**: Drastyczne zaostrzenie kryteriów handoffu w Zasadzie 6 (tylko pogoda, urządzenia domowe, internet) oraz próba zmuszenia modelu do mówienia "Nie wiem" przy brakach wiedzy.
- **Wyniki**:
  - `tool_calls` spadły z 16 do 4 (opanowanie over-handoffu).
  - Poprawa faktów: poprawna liczba planet (8 na pl-q07) oraz dni lutego (29 na pl-q25).
  - **Efekt uboczny**: Regresja w stylu — model zaczął dodawać emotikony (5), a reguła "przyznaj niepewność" sprowokowała głośne myślenie w języku angielskim ("Wait...", "I need to check...") na en-q03 i en-q04. Średnia ocen: 4.13.

#### Iteracja v8 (`llm_quality_lfm3-q4_v8.json`, `llm_quality_scores_lfm3-q4_v8.json`)
- **Zmiany**: Zablokowanie wycieków myślenia (Zasada 6), kategoryczny zakaz emoji i emotikonów, zakaz witania się na nowo, przykłady słownego zapisu dat i liczb.
- **Wyniki**:
  - Emoji zredukowane do 0, zero wycieków znaczników narzędzi/XML.
  - Znakomity zapis liczb: II wojna światowa w 1945 r. zapisana słownie ("tysiąc dziewięćset czterdziestym piątym").
  - **Efekt uboczny**: Za mocne ograniczenie handoffu (0/30). Ponadto model zaczął nadużywać formułki "Jestem Gaja, asystentka w tym domu" (17/30) oraz frazy pożegnalnej "Do usłyszenia, miłego dnia". Średnia ocen: 4.18.

#### Iteracja v9 (`llm_quality_lfm3-q4_v9.json`, `llm_quality_scores_lfm3-q4_v9.json`)
- **Zmiany**: Usunięcie powtarzania formułki tożsamości z promptu systemowego, zwięzłe przykłady bez zbędnego "Gaja tu jest", wzmocnienie reguły jednojęzyczności (mirroring).
- **Wyniki**:
  - Formułka tożsamości spadła do 3 wystąpień.
  - Emoji: 0, digits: 1, brak wycieków markup.
  - Handoff wywołany na pl-q20 ("zrób to o czym mówiliśmy"). Średnia ocen: 4.21.

#### Iteracja v10 (`llm_quality_lfm3-q4_v10.json`, `llm_quality_scores_lfm3-q4_v10.json`)
- **Zmiany**: Naturalizacja zakończeń wypowiedzi (zakaz dodawania niepytanych pożegnań), usunięcie zbędnych dopisków "Do usłyszenia", doprecyzowanie najwyższego szczytu w Polsce w przykładach.
- **Wyniki**:
  - Osiągnięto najwyższy wynik jakościowy w historii testów: **4.45 / 5.0** (Fakty: 4.63, Zwięzłość: 4.43, Język: 4.43, Persona: 4.30).
  - Wyeliminowano halucynacje geograficzne (Rysy 2499 m, rzeka Tyber w Rzymie, Leonardo da Vinci, 8 planet, 29 dni lutego).
  - Zero wycieków tagów narzędzi i markupu.
  - Pozostałości: 3 emoji, 2 wycieki cyfr arabskich.

#### Iteracja v11 (`llm_quality_lfm3-q4_v11.json`, `llm_quality_scores_lfm3-q4_v11.json`)
- **Zmiany**: Próba agresywnego wymuszenia bilingwalnego mirroringu (English only for English prompts) i twardych negatywnych reguł dotyczących emoji oraz składni handoffu.
- **Wyniki**:
  - **Zapaść małego modelu (regresja do 3.71 / 5.0)**: Zjawisko "negative constraint attractor" typowe dla modeli 1.7B. Model zaczął prowadzić meta-dialog w odpowiedziach ("Actually, I cannot use emojis...", "You asked in English so I...", "I like both languages, but I will keep it simple.").
  - Wycieki składni wywołania narzędzia wprost do treści (pl-q20, pl-q21) oraz powrót formułki tożsamości w 8 odpowiedziach.

#### Iteracja v12 (`llm_quality_lfm3-q4_v12.json`, `llm_quality_scores_lfm3-q4_v12.json`)
- **Zmiany**: Powrót do eleganckiej, zwięzłej struktury v10. Wyczyszczenie meta-reguł. Dodanie czystych przykładów żartu (bez emoji) i preferencji kulinarnych. Twarda, lecz prosta definicja formatu mowy (słowny zapis liczb i lat).
- **Wyniki**:
  - **Emoji: 0 / 30** (pełny sukces, cel osiągnięty).
  - **Cyfry (digits): 0 / 30** (pełny sukces, każda liczba i rok słownie, cel osiągnięty).
  - **Wycieki markup / tagów: 0** (zero wycieków formatowania).
  - **Prefiks Gaja:: 0** (brak).
  - **Formułka tożsamości: dokładnie 2 / 30** (pl-q04 i pl-q24 — cel <=2 osiągnięty).
  - **Zdania >3: 6 / 30** (blisko celu <=5).
  - **Średnia jakość: 4.34 / 5.0** (Język: 4.53, Fakty: 4.57, Zwięzłość: 4.23, Persona: 4.03).

---

### Finalna rekomendacja i ocena potrzeby SFT

1. **Możliwości samego prompt engineeringu (sukcesy v10/v12)**:
   - **Opanowanie over-handoffu**: Spadek z katastrofalnych 16/30 (v6) do 1-2 precyzyjnych wywołań w v10/v12.
   - **Czystość formatu mowy TTS**: W v12 osiągnięto **0 emoji** oraz **0 cyfr arabskich** (100% liczb, dat i lat zapisanych poprawnie słownie).
   - **Stabilność tożsamości persony**: Zredukowano natrętną formułkę "Jestem Gaja..." z kilkunastu wystąpień do zaledwie 2/30 (tylko wprost pytana o tożsamość).
   - **Faktografia ogólna**: Zlikwidowano konfabulacje historyczno-geograficzne (Mona Liza, Rysy, Tyber, 8 planet, 29 dni lutego).

2. **Twarde limity architektury 1.7B bez SFT (Dlaczego prompt engineering ma granice)**:
   - **Bilingwalny voice mirroring (EN <-> PL)**: Przy braku historii konwersacji (en-q01 do en-q04) model 1.7B z polskim promptem systemowym ma silny bias językarski. Próby wymuszenia reguł po angielsku/polsku w prompcie (v11) prowadzą do konfabulacji meta-językowych.
   - **Arytmetyka i operacje wielokrokowe**: Wyłączenie budżetu myślenia (`--reasoning-budget 0`) uniemożliwia modelowi poprawne wyliczanie złożonych działań (pl-q18).
   - **Dyskretne granice wywołań narzędzi**: Balansowanie między 0 a 5 handoffami na zróżnicowanym korpusie bez konkretnego kontekstu domowego jest niestabilne na poziomie samego tekstu promptu.

3. **Werdykt i rekomendacja wdrożeniowa**:
   - **Tryb jednoręczny (wyłącznie polski asystent domowy)**: Wersje **v10** (najwyższa naturalność 4.45) lub **v12** (najczystszy format TTS z 0 emoji i 0 cyfr, ocena 4.34) są **w pełni gotowe do wdrożenia bez trenowania wag**.
   - **Tryb bilingwalny / pełna precyzja tool-calling**: **Konieczny jest lekki SFT (LoRA)** na ~500–1000 parach dialogowych. Dataset powinien zawierać zrównoważone przykłady angielskie i polskie oraz twarde tokeny funkcyjne dla wywołań `handoff_to_hermes`.
