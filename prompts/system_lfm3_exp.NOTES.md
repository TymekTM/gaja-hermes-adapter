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
