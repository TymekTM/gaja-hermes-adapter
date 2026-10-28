# Wyniki ewaluacji LFM3-1.7B dla Gaji i Hermesa

Dokument decyzyjny worktree `t3/71e485c3` (2026-10-27). Ocenia rodzinę **LiquidAI LFM3**
(premiera 27.10.2026) pod trzema kątami:

1. **Zamiennik LFM2.5-2.6B w pętli głosowej Gaji** (jakość PL, persona, TTFT),
2. **Mały model dla agenta Hermes** (proste, powtarzalne zadania),
3. **Dłuższy kontekst** (needle-in-a-haystack po polsku do 32K) oraz **vLLM**.

Sprzęt i silnik: GTX 1050 Ti 4 GB (Pascal sm_61), llama.cpp master `338fce6`
(wsparcie LFM3 z PR #31252/#31257, ten sam dzień), CUDA 12.9 (arch 61; CUDA 13.x
wycina Pascala), host g++-13 jak w buildzie produkcyjnym.

---

## 1. Modele i warianty

| Zmienna | LFM2.5-2.6B (produkcja) | LFM3-1.7B | LFM3-Audio-2B | LFM3-Omni-2B | d1-2B |
|---|---|---|---|---|---|
| Parametry | 2.6 B | 1.73 B | 2 B | 2 B | 2 B |
| Kontekst | 32 K | **256 K** | 32 K | 32 K | 32 K |
| Q4_K_M | 1.67 GB | **1.05 GB** | 1.10 GB (+mmproj 0.32 GB) | 1.10 GB (+mmproj 0.52 GB) | 1.10 GB |
| Polski | poza listą | **oficjalnie (1/16)** | b.d. | b.d. | b.d. |
| Rola | chat głosowy | agenty/narzędzia | audio↔tekst | omni | reasoning |

Oficjalne benchmarki LiquidAI (LFM3-1.7B vs LFM2.5-2.6B): średnia **49.37 vs 31.55**;
AutomationBench 41.29 vs 0.94; BFCL V4 71.39 vs 55.89; RULER 128K 65.56 vs 1.15.
Pełne porównanie: `bench/results/lfm25_vs_lfm3_official.html`.

## 2. Kluczowa konfiguracja llama.cpp dla LFM3

- GGUF **nie ma osadzonego szablonu czatu** — trzeba podać
  `--chat-template-file models/templates/LFM3.jinja` (upstream llama.cpp).
- Wycięcie rozumowania: **`--reasoning-budget 0`** działa niezawodnie.
  Wariant "pusty `<think></think>` w szablonie" **nie wystarcza** — model RL-trenowany
  do myślenia otwiera nowy blok `<think>` sam z siebie (widoczne w smoke teście).
- Natywne tool calling: format `<tool_call><function=…>` działa (logi w §4),
  ale bywa wyciekany do treści jako surowy markup.

## 3. Jakość po polsku (quality.jsonl, 30 promptów, prompt v5)

Auto-checki (im mniej tym lepiej; temp 0.3):

| Metryka | LFM2.5 v4 (prod, ocena ręczna) | LFM3 nothink | LFM3 think | LFM3 nothink (sampling z karty) |
|---|---|---|---|---|
| emoji | 1 | **10** | **10** | **10** |
| markdown | 0 | 0 | 0 | 1 |
| >3 zdania | 7–8 | 16 | 18 | 16 |
| think_leak | 0 | 0 | 0 | 0 |
| tool_calls | 0 | 1 | 2 | 3 |
| empty | 0 | 0 | 0 | 0 |

Oceny 1-5 (rubryka jak v4; oceniający: agent GLM-5.3, plik
`bench/results/llm_quality_scores_lfm3-q4_nothink.json`):

| Oś | LFM2.5-2.6B (ręcznie, v4) | LFM3-1.7B |
|---|---|---|
| lang | 3.97 | 3.63 |
| brevity | 3.67 | 3.63 |
| persona | 4.13 | **3.47** |
| fact | 3.97 | 3.97 |
| **overall** | **3.93** | **3.67** |

Główze problemy LFM3 w roli głosowej: emoji w 1/3 odpowiedzi, wyciek identyfikacji
("Jestem LFM od Liquid AI"), wyciek markupu tool-calli do treści, samoprzypis
"Gaja:", sporadyczne odpowiedzi po angielsku na polskie prompty (i odwrotnie),
garbled fakty ("jedna planeta Ziemia"). Sampling z karty (temp 0.1/top_k 50/rep 1.1)
**nie** zmienia emoji/gadatliwości.

**Wniosek: LFM3-1.7B NIE jest drop-in zamiennikiem LFM2.5-2.6B w pętli głosowej.**
Potrzebowałby dostrojenia persony (prompt v6 / SFT). Do ról agentowych — patrz niżej.

## 4. Tool calling (rola agentowa)

`handoff_to_hermes` wywoływany poprawnie składniowo (1-3/30 promptów), ale recall
niski — ten sam problem, który LFM2.5 miał w Strategii A (dlatego Router v3 używa
logprobów). **Router v3 byłby do retunowania** przy jakiejkolwiek zmianie modelu.

## 5. Crony → JSON (zaparkowane na życzenie: nowa funkcjonalność)

Zero-shot: JSON 8/10, poprawne wyrażenia cron **1/10** (systematyczne mieszanie
GG:MM — "17:45" → `5 17`). Few-shot (2 przykłady): JSON 10/10, cron ~2-4/10.
Rekomendacja na przyszłość: intencja z LLM + deterministyczny generator składni
(reguły, jak CHEAP_RULES w routerze), albo SFT pod format.

## 6. Szybkość (GTX 1050 Ti, llama.cpp 338fce6)

Wszystkie konfiguracje: Q4_K_M, `-ngl 99`, prompt v5 + tools, 5 promptów × 3 powtórzenia,
ta sama binarka i metoda dla obu modeli (`bench/llm_bench.py`).

| Konfiguracja | TTFT ciepły | TTFT zimny | 1. zdanie | tok/s | VRAM Δ |
|---|---|---|---|---|---|
| **LFM3-1.7B** baseline (c4096, fa) | 0.701 s | **0.989 s** | 0.846 s | **44.6** | **1220 MiB** |
| LFM3 fa_off | 0.781 s | 1.221 s | 0.948 s | 40.3 | 1316 MiB |
| LFM3 ub128 | 0.999 s | 1.059 s | 1.147 s | 43.7 | 1170 MiB |
| LFM3 ub1024 | 1.008 s | 1.118 s | 1.158 s | 43.2 | 1294 MiB |
| LFM3 kv_q8 | 0.724 s | 1.102 s | 0.885 s | 43.0 | 1194 MiB |
| LFM3 ctx8192 | 0.710 s | 1.095 s | 0.859 s | 44.1 | 1272 MiB |
| LFM3 think | 0.663 s* | 1.036 s | n/d** | 44.0 | 1220 MiB |
| LFM3 + no-reasoning-preserve | 0.711 s | 1.091 s | 0.859 s | 43.9 | 1220 MiB |
| **LFM3 + `--swa-full`** | **0.131 s** | 1.007 s | **0.278 s** | 42.5 | 1254 MiB |
| **LFM2.5-2.6B** baseline_budget0 | **0.133 s** | 1.552 s | 0.664 s | 32.3 | 1794 MiB |

\* pierwszy token rozumowania (nie treści). \** 120 tokenów nie wystarczało na wyjście
z trybu think przy pomiarze pierwszego zdania.

Wnioski:

- **Dekodowanie: LFM3 +38%** (44.6 vs 32.3 tok/s) przy **32% mniejszym VRAM** (1220 vs 1794 MiB).
- **Zimny TTFT: LFM3 +36% szybszy** (0.99 s vs 1.55 s).
- **Ciepły TTFT: rozwiązany flagą `--swa-full`** (0.131 s ≈ LFM2.5!). Pierwotna diagnoza
  "bug cache" była błędna co do natury: LFM3 ma 5 warstw SWA, które domyślnie trzymają
  tylko okno KV → prefix cache nie mógł reuse'ować promptu → pełny re-prefill
  (0.701 s). `--swa-full` (pełne KV dla SWA, koszt +34 MiB) przywraca trafienia:
  **TTFT ciepły 0.131 s, pierwsze zdanie 0.278 s — 2.4× szybciej niż LFM2.5 (0.664 s)**.
  `--no-reasoning-preserve` nic nie zmienia (0.711 s) — to nie reasoning był przyczyną.
- Skala referencyjna: 3060 Ti (Ampere, ~448 GB/s) robi 130–160 tok/s na tym modelu —
  nasz 1050 Ti (112 GB/s) przy 44 tok/s działa ~proporcjonalnie do pasma, bez anomalii.
- **Konfiguracja produkcyjna LFM3 (gdyby wdrażać): `-fa on --swa-full` + LFM3.jinja
  + `--reasoning-budget 0`** — pełen paritet latencji z LFM2.5 przy dekodowaniu +38%.

Dane: `bench/results/llm_latency_lfm3_*.json`, `llm_latency_lfm25_newbuild.json`.

## 7. Needle-in-a-haystack po polsku (do 32K)

`bench/needle_in_haystack.py` — siatka 4 konteksty × 7 głębokości (0–100%) × 3 typy
igieł (kod XK-0000, godzina zadania crona, hasło), haystack = prawdziwa polska proza
(Pan Tadeusz, Quo Vadis; Wolne Lektury, domena publiczna). Q4_K_M, KV f16, 84 requesty.

| Metryka | Wynik |
|---|---|
| **Trafność ogólna** | **98.8%** (83/84) |
| igła "kod" | 100% |
| igła "hasło" | 100% |
| igła "godzina crona" | 96.4% (1 miss) |
| Prefill 32K | 352–696 tok/s (spada z długością; jednorazowy koszt ~60 s) |

Heatmapa: `bench/results/niah_lfm3_q4.html`, dane: `niah_lfm3_q4.json` / `.jsonl`.

**Wniosek: kontekst można wydłużyć z obecnych 8K do 16K–32K bez ryzyka utraty
trafności retrievalu.** KV dla 32K zmieściło się w VRAM gdy serwer był jedynym
procesem na GPU; w produkcji trzeba by zwolnić miejsce. Pełne 256K nieprzetestowane
(limit 4 GB VRAM).

## 8. LFM3-Audio-2B jako kandydat na STT/TTS (smoke, 8 klipów FLEURS PL)

Serwer: `-m LFM3-Audio-2B-Q4_K_M.gguf --mmproj mmproj-Q8_0` (łącznie ~1.4 GB VRAM).

**STT** przez `/v1/audio/transcriptions` (kompatybilne z API Whisper) z `language=pl`.
Kluczowe odkrycie: własny prompt tekstowy NADPISUJE wbudowany preset ASR llama.cpp
i psuje transkrypcję (ucięcia, streszczenia); bez własnego promptu działa wzorowo.

| | LFM3-Audio-2B (Q4+mmproj Q8) | faster-whisper small (produkcja) |
|---|---|---|
| WER PL (8 klipów FLEURS) | **13.66%** | 18.69% |
| Latencja / klip 7–12 s | 0.9–2.8 s | 0.395 s p50 (krótsze klipy) |

**TTS** przez `POST /tts` (input=tekst, language=pl) — zwraca `audio/wav`
(mono 24 kHz 16-bit); próbki: `bench/results/lfm3_audio_tts_pl*.wav`.

**Wniosek:** jeden model 2B realnie łączy STT+TTS po polsku (teza potwierdzona na
próbce smoke). Do decyzji: pełna ewaluacja 44 klipów, RTF TTS, odsłuchowa ocena vs
Piper. Błędy fonetyczne na trudnych słowach ("spoemierzone" ↔ "sprzymierzone").

## 8.1 Eksperymentalny prompt v6→v12 (pętla: Gemini 3.8 Flash, 7 iteracji)

`prompts/system_lfm3_exp.md` — finalna wersja **v12** po automatycznej pętli
edycja→test→ocena (delegat Gemini 3.8 Flash; szczegóły w `.NOTES.md`, wyniki
per iteracja w `bench/results/llm_quality_lfm3-q4_v*.json` + `llm_quality_scores_*.json`).

| Wersja | tool_calls | emoji | digits | >3 zdania | markup | formułka | ocena 1-5 |
|---|---|---|---|---|---|---|---|
| v6 (start) | 16 | 1 | 1 | 4 | 1 | ~12 | 4.03 |
| v7 | 4 | 5 | 4 | 9 | 1 | 5 | 4.13 |
| v8 | 0 | 0 | 3 | 10 | 0 | 17 | 4.18 |
| v9 | 1 | 0 | 1 | 9 | 0 | 3 | 4.21 |
| v10 | 1 | 3 | 2 | 7 | 0 | 3 | **4.45** |
| v11 | 5 | 3 | 1 | 15 | 2 | 8 | 3.71 (regresja) |
| **v12** | 1 | **0** | **0** | 6 | **0** | **2** | **4.34** |

v12 (rubryka: lang 4.53, brevity 4.23, persona 4.03, fact 4.57):
- **format TTS bezbłędny**: zero emoji, zero cyfr, zero wycieków markupu/prefiksów;
- **formułka identyczności tylko tam, gdzie pytana** (2/30);
- **over-handoff rozwiązany** (16→1-2); **faktografia podstawowa naprawiona**
  (osiem planet, dwadzieścia dziewięć dni, Leonardo);
- odkrycie z v11: *"Negative Constraint Attractor"* — agresywne meta-reguły
  językowe u 1.7B wywołują meta-dialog i wycieki składni (3.71); wycofane w v12.

**Werdykt delegata:** dla wariantu **PL prompt wystarcza do wdrożenia**.
Do **SFT (LoRA, ~500-1000 par)** kwalifikują się: bezkontekstowy mirroring EN↔PL
(zero-shot angielski przy polskim promptcie systemowym) i wielokrokowa arytmetyka
bez łańcucha myślenia (pl-q18). Większość odpowiedzi po angielsku w historii
rozmowy działa poprawnie bez SFT.

## 9. vLLM na GTX 1050 Ti

**Weryfikacja empiryczna (vLLM 0.32.0, torch 2.13.0+cu130): niedziałające.**

1. `NVIDIA GeForce GTX 1050 Ti with CUDA capability sm_61 is not compatible with
   the current PyTorch installation. The current PyTorch install supports CUDA
   capabilities sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.` — ostrzeżenie przy imporcie.
2. Próba startu silnika (`LLM(...)`, dummy load): `CUDA error: no kernel image is
   available for execution on the device` (`cudaErrorNoKernelImageForDevice`).

To nie jest kwestia konfiguracji: dystrybuowane przez pip vLLM linkuje PyTorch
zbudowany tylko pod sm_75+ (Turing i nowsze); NVIDIA usunęło Pascala z CUDA 13.
Pełny log: `bench/results/vllm_pascal_proof.log`.

Konsekwencje dla celu "vLLM dla Hermesa":
- **Na tym GPU vLLM nie wstanie** (niezależnie od modelu; SGLang ma te same wymagania).
- Pozostaje **llama.cpp** — i dla LFM3 jest to bardzo dobry silnik (tri-hybrid:
  13 warstw konwolucyjnych nie ma KV; prefill/decode szybszy niż LFM2.5 — patrz §6).
- vLLM miałby sens dopiero na sprzęcie sm_75+ (np. used T4 16 GB) — wtedy
  `vllm serve LiquidAI/LFM3-1.7B` (BF16) działa natywnie wg karty modelu.

## 8.2 Router v3 na LFM3 (retune) + kontekst 32K

Retune przeprowadzony na żywym serwerze testowym (`-c 32768 --swa-full`, port 8794,
`GAJA_LLM__BASE_URL` + `GAJA_LLM__ROUTER_SLOT=0`; dane:
`bench/results/router3_eval_lfm3.json` + `..._raw_lfm3.jsonl`).

**Wynik: prompt `prompts/router3_pl.md` działa na LFM3 bez żadnej zmiany**
(ChatML i BOS `<|startoftext|>` wspólne z LFM2.5; tokeny ` H`/` LOCAL`/` TO`
emisjonowane naturalnie — smoke: "wyślij maila do Kasi" → ` H` P≈1.0).
Zmienia się tylko punkt pracy: **τ* = 0.5** (LFM2.5: 0.97) — LFM3 ma inny,
bardziej zachowawczy rozkład prawdopodobieństw (przy τ=0.97 recall spada do ~81%).

| Zbiór | R strict | Precision | FHR | Uwagi |
|---|---|---|---|---|
| **routing_holdout (80 ślepych)** | **96.2%** | **100%** | **0.0%** | LFM2.5 prod: 95.0/95.0/5.0 |
| routing (tuning, 128) | 85.4% | 95.6% | 3.3% | |
| wszystkie (208) | 89.2% | 97.1% | 2.0% | τ=0.5, tool_τ=0.6 |

Pogoda/godzina/światła trafiają prawidłowo w kategorię TOOL (ścieżka lokalnych
narzędzi) — nie jest to FN, tylko poprawne pokrycie architektury v3.
Latencja klasyfikacji: p50 **82 ms** / p90 89 ms (LFM2.5: 54 ms; budżet 150 ms).

**Kontekst 32K zweryfikowany produkcyjnie**: serwer LFM3 Q4_K_M z `-c 32768 -fa on
--swa-full` zajmuje **1931 MiB** (model+KV+compute) — mieści się obok Whispera
adaptera (378 MiB) i dashboardu Hermesa (~1 GB) z ~0.7 GB zapasu. 64K (~2.4 GB
szac.) wchodziłoby już tylko solo. KV rośnie symbolicznie dzięki architekturze
hybrydowej (4K→32K to +364 MiB). **Rekomendacja: 32K przy wdrożeniu LFM3.**

Konfiguracja produkcyjna LFM3 (głos, gotowa do swap-in):
```bash
llama-server -m LFM3-1.7B-Q4_K_M.gguf --host 127.0.0.1 --port 8791 \
  -ngl 99 -c 32768 -np 2 --no-kv-unified --jinja \
  --chat-template-file bench/templates/LFM3.jinja --reasoning-budget 0 \
  -fa on --swa-full
# + router: threshold = 0.5 (config [router]); persona: prompts/system_lfm3_exp.md (v12)
```

## 9.1 Badanie: zamiana whisper/parakeet na LFM3-Audio-2B (STT)

Pełna ewaluacja repo-metodologią (jiwer + normalize_text; `bench/lfm3_audio_stt_eval.py`,
dane: `bench/results/lfm3_audio_stt_eval.json`; serwer: Q4_K_M + mmproj Q8_0,
`/v1/audio/transcriptions`).

| Metryka | **LFM3-Audio-2B** | parakeet-TDT-0.6B (prod) | faster-whisper small (GPU) |
|---|---|---|---|
| PL WER (44 FLEURS) | **12.68%** | 12.6% | 18.69% |
| mixed PL/EN WER | 29.3% (auto) / 33.8% (pl) | **19.0%** | b/d |
| Latencja p50 / p90 (2-5 s) | 596 / 773 ms | ~300-500 ms (CPU) | 395 / 436 ms |
| RTF | 6.06 | b/d | 7.70 |
| VRAM | ~1.4 GB (z mmproj) | 0 (CPU) | 0.38 GB |

**Werdykt STT:** na czystej polszczyźnie LFM3-Audio **dorównuje parakeet** (remis
statystyczny 12.7 vs 12.6%) i bije whisper o 6 p.p. Słabość: **code-switching
PL/EN (29-34% vs 19%)** — parakeet zachowuje przewagę na słownictwie technicznym.
Latencja p50 ~200 ms wyższa niż whisper (nad budżetem 300 ms, ale RTF porównywalny).

## 9.2 System docelowy: jeden model audio (STT+chat+TTS) — potencjał potwierdzony

Zgodnie z decyzją właściciela, sprawdzono wszystkie trzy role w LFM3-Audio-2B:

| Rola | Wynik | Uwagi |
|---|---|---|
| **STT** | ✓ 12.68% PL / RTF 6.06 | §9.1; mixed PL/EN do doszlifowania |
| **Chat/persona** | ✓ format idealny | v12: emoji 0, cyfry 0, zdania>3: 0, wycieki 0; **EN mirroring lepszy niż 1.7B tekst**; ale over-handoff 12/30 + puste zapowiedzi handoffu |
| **TTS** | ✓ RTF **1.86** | 8.5× lepszy RTF niż Piper (15.74); 24 kHz WAV przez `POST /tts`; próbki w bench/results |
| Speech→speech 1 request | ✗ ten build | chat zwraca echo instrukcji; kandydat: upstream `/v1/realtime` (PR #31269, poza naszymym buildem) |

**VRAM całej pętli w jednym modelu: ~1.5 GB** (vs ~2.2 GB dziś: whisper+LFM2.5,
vs ~3.4 GB w wariantach LFM3-1.7B+Audio).

**Rekomendowana adopcja etapowa:**
1. **Etap 1 (możliwy dziś):** LFM3-Audio jako ujednolicony serwis STT+TTS obok
   LFM3-1.7B (persona v12, router τ=0.5) — usuwa whisper+parakeet+Piper z GPU;
   parakeet zostaje jako fallback CPU dla wypowiedzi code-switchingowych (router
   STT po confidencie) albo akceptujemy lukę.
2. **Etap 2 (po przebudowie llama.cpp na master):** `/v1/realtime` — streaming
   audio w obrębie jednego modelu; wtedy pełna pętla single-model.
3. **Do domknięcia:** prompt v13 (over-handoff + zapowiedzi handoffu na audio),
   mixed-PL/EN (ew. SFT), streaming latency budget.

## 9.3 E2E /v1/realtime (upstream master e486a4b, PR #31269) — pełna pętla single-model

llama.cpp master przebudowany (0.7.0-dev, inkrementalnie) i przetestowany pełnym
E2E przez WebSocket (`bench/lfm3_realtime_e2e.py`, dane:
`bench/results/lfm3_realtime_e2e.json`; audio streamowane chunkami 200 ms jak
z mikrofonu; usługi zatrzymane na czas testu, przywrócone po).

**Tryb transkrypcyjny (44 FLEURS PL):**

| Metryka | realtime (streaming) | batch (/v1/audio/transcriptions) |
|---|---|---|
| WER | 14.85% | **12.68%** |
| commit→transkrypcja p50 / p90 | 923 / 1302 ms | ~600 ms (p50 całego requestu) |

Streaming kosztuje ~2.2 p.p. WER względem batch (chunking audio w locie) —
akceptowalny trade-off za brak oczekiwania na koniec wypowiedzi.

**Tryb konwersacyjny (audio-in → audio+text-out, jeden model):**

| Metryka | Wynik |
|---|---|
| **Time-to-first-audio (TTFA)** | **0.28–0.32 s** od commita |
| Pełny round-trip (commit→response.done) | 4.0–7.3 s (echo całej wypowiedzi; krótsza odpowiedź = proporcjonalnie szybciej) |
| Audio-out | 5.2–10.6 s mowy 24 kHz/klip, WAV w bench/results |

Uwaga: sesja "realtime" **wymaga jawnego `response.create`** po commicie
(odpowiedź nie startuje automatycznie — inaczej niż w OpenAI API); eventy:
`response.output_audio.delta` (PCM16 b64), `response.output_text.delta`,
`response.done`. Transkrypcja wejścia nie przychodzi osobno w trybie realtime
(audio trafia prosto do konwersacji jako wiadomość użytkownika). Model na klipach
FLEURS (oznajmienia) echoował treść zamiast odpowiadać — kwestia instrukcji
sesji ("odpowiadaj użytkownikowi"), nie mechaniki; latencja i generacja audio
działają poprawnie.

**Werdykt E2E:** pętla głosowa single-model **działa mechanicznie na naszym
GPU** — STT+chat+TTS w jednym procesie, TTFA ~0.3 s. Do poziomu produkcyjnego:
instrukcje sesji pod odpowiedzi (nie echo), skrócenie round-tripu (limit
outputu), i WER streamingowy do poprawy (ver. upstream / ew. SFT).

## 10. Rekomendacje

> **[2026-10-28 13:35] MIGRACJA WYKONANA.** Produkcja działa na LFM3-1.7B Q4_K_M
> (32K, 2x16K slotów, --swa-full, budget 0, LFM3.jinja, persona v12, router τ=0.5,
> VRAM 1899 MiB). Walidacja E2E: lokalny LLM TTFT 135 ms / E2E 510 ms; tool 1.03 s;
> fast-path 147 ms. Rollback: /tmp/opencode/gaja-llm.service.backup (unit),
> backup starej binarki w /tmp/opencode/llama_build_backup_*, przywrócić ścieżkę
> GGUF w bench/logs/lfm_gguf_path.txt na LFM2.5 i sekcje [router]/[llm] w
> config.local.toml.

1. **Głos (pętla Gaji): LFM2.5 zostaje na dziś, ale LFM3 zrównał latencję** (§6:
   `--swa-full` → TTFT 0.131 s, pierwsze zdanie 0.278 s = 2.4× lepiej niż LFM2.5)
   **i persona jest gotowa po polsku** (§8.1: v12, format TTS bezbłędny, ocena 4.34,
   faktografia naprawiona) **i Router v3 działa bez zmian promptu** (§8.2: holdout
   R 96.2% / P 100% przy τ=0.5). Migracja głosu na LFM3 jest gotowa do przeprowadki:
   konfiguracja swap-in w §8.2 (32K, --swa-full, v12, router τ=0.5). Pozostaje
   decyzja o EN mirroringu (SFT albo akceptacja polskojęzycznej inercji).
   Zysk: −32% VRAM, +38% dekodowania, pierwsze zdanie 2.4× szybciej, kontekst 4×.
2. **Hermes / proste zadania: wdrożyć LFM3-1.7B jako drugi, mały model** (llama.cpp,
   Q4_K_M, szablon `LFM3.jinja` + `--reasoning-budget 0`; `--swa-full` gdy istotne
   wielokrotne requesty z tym samym prefixem). Profil: 44.6 tok/s, 1.2 GB VRAM,
   zimny TTFT ~1 s, polski oficjalnie, JSON niezawodny.
   Crony: intencja z LFM + deterministyczny generator składni (SFT docelowo).
3. **vLLM: na tym GPU niemożliwe** (sm_61 < sm_75, §9). Opcja wraca przy sprzęcie
   Ampere+ (na 3060 Ti LFM3 robi 130–160 tok/s).
4. **Kontekst: podnieść do 16K** (ew. 32K) przy wdrożeniu LFM3 — NIAH 98.8% (§7).
5. **Audio-2B: kontynuować ewaluację** — realny kandydat na scalenie STT+TTS w jeden
   model po stronie llama-server (§8); pełny test WER + RTF TTS + porównanie z Piper.
6. **Prompt v7** wg §8.1; Router v3 retunować dopiero przy zmianie modelu w głosie.
7. **Upstream watch**: `/v1/realtime` (PR #31269, w teście §9.3) i video-input LFM3
   (PR #31283, otwarty) — istotne dla workstreamu audio. d1-2B usunięty z dysku
   (decyzja właściciela: brak zastosowania w naszym setupie).

### Zmiany w repo (worktree t3/71e485c3)

- **Nowe narzędzia**: `bench/needle_in_haystack.py` (NIAH PL), `bench/cron_eval.py`,
  `bench/lfm3_audio_stt_smoke.py`, `bench/lfm3_audio_e2e.py` (STT/TTS llama-server)
- **Rozszerzone**: `bench/llm_server.py` (warianty lfm3/lfm3_nothink), `bench/llm_bench.py`
  (--set/--only/--out, configi LFM3, TTFT w trybie think), `bench/llm_quality.py`
  (top-k/repeat-penalty)
- **Szablony**: `bench/templates/LFM3.jinja` + `lfm3_nothink.jinja` (upstream PR #31252)
- **Dane**: `bench/data/niah/` — korpus haystack (domena publiczna, ~1.7 MB)
- **Prompty**: `prompts/system_lfm3_exp.md` + `.NOTES.md` (v6 eksperymentalny)
- **Dokument**: niniejszy plik; surowe wyniki w `bench/results/` (gitignored)
