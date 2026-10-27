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
| **LFM2.5-2.6B** baseline_budget0 | **0.133 s** | 1.552 s | 0.664 s | 32.3 | 1794 MiB |

\* pierwszy token rozumowania (nie treści). \** 120 tokenów nie wystarczało na wyjście
z trybu think przy pomiarze pierwszego zdania.

Wnioski:

- **Dekodowanie: LFM3 +38%** (44.6 vs 32.3 tok/s) przy **32% mniejszym VRAM** (1220 vs 1794 MiB).
- **Zimny TTFT: LFM3 +36% szybszy** (0.99 s vs 1.55 s).
- **Ciepły TTFT: LFM3 5× GORSZY (0.70 s vs 0.13 s)** — i to jest aktualnie bloker dla
  pętli głosowej. Przyczyna zdiagnozowana w logach slotów: LFM2.5 po prime'owaniu
  trafia w prefix cache i liczy tylko ~14 nowych tokenów, natomiast **LFM3 za każdym
  razem przelicza cały prompt (~750–1050 tokenów)**. `--no-reasoning-preserve` nie
  zmienia nic (0.711 s). To day-0 implementacja tri-hybrydy (PR #31252 z dziś,
  w tym "recurrent state rollback dla LFM3") — kandydat do zgłoszenia/obserwacji
  upstream; dla zadań cronowych bez znaczenia (jednorazowe requesty, zimny TTFT lepszy).
- Skala referencyjna: 3060 Ti (Ampere, ~448 GB/s) robi 130–160 tok/s na tym modelu —
  nasz 1050 Ti (112 GB/s) przy 44 tok/s działa ~proporcjonalnie do pasma, bez anomalii.

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

## 8.1 Eksperymentalny prompt v6 (nadzór: Gemini 3.8 Flash)

`prompts/system_lfm3_exp.md` — twarde reguły + few-shot bez etykiet ról; recenzja
Gemini 3.8 Flash w `.NOTES.md` (m.in. usunięcie etykiet "Gaja:" z przykładów,
rozszerzenie zakazu cyfr, negatywne przykłady handoffu). Wyniki na quality.jsonl:

| Metryka | LFM3 v5 | **LFM3 v6** | LFM2.5 v5 (świeży) |
|---|---|---|---|
| emoji | 10 | **1** | 1 |
| >3 zdania | 16 | **4** | 14 |
| digits | 5 | **1** | 2 |
| tool_calls | 1 | 16 | 2 |
| **ocena 1-5 (agent)** | 3.67 | **4.03** | b/d (ręczne v4: 3.93) |

v6 naprawił: identyczność (pl-q04 ✓), emoji, gadatliwość (lepsza niż LFM2.5!), cyfry.
Pozostało: formułka "Jestem Gaja, asystentka w tym domu" w ~40% odpowiedzi,
**over-handoff 16/30** (również powitania), słaba faktografia ("sześć planet",
"luty ma czterdzieści dwa dni"), sporadyczne mieszanie EN/PL, 1 wyciek markupu.
Kierunek v7: przykład identyczności wyeliminować (reguła wystarczy), negatywne
przykłady handoffu, docelowo SFT persony.

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

## 10. Rekomendacje

1. **Głos (pętla Gaji): zostać przy LFM2.5-2.6B.** LFM3 przegrywa personą i ciepłym
   TTFT (bug cache day-0 w llama.cpp; śledzić upstream, powtórzyć test po fixie).
2. **Hermes / proste zadania: wdrożyć LFM3-1.7B jako drugi, mały model** (llama.cpp,
   Q4_K_M, szablon `LFM3.jinja` + `--reasoning-budget 0` dla natychmiastowych odpowiedzi).
   Profil: 44.6 tok/s, 1.2 GB VRAM, zimny TTFT ~1 s, polski oficjalnie, JSON niezawodny.
   Crony: intencja z LFM + deterministyczny generator składni (SFT docelowo).
3. **vLLM: na tym GPU niemożliwe** (sm_61 < sm_75, §9). Opcja wraca przy sprzęcie
   Ampere+ (na 3060 Ti LFM3 robi 130–160 tok/s).
4. **Kontekst: podnieść do 16K** (ew. 32K) przy wdrożeniu LFM3 — NIAH 98.8% (§7).
5. **Audio-2B: kontynuować ewaluację** — realny kandydat na scalenie STT+TTS w jeden
   model po stronie llama-server (§8); pełny test WER + RTF TTS + porównanie z Piper.
6. **Prompt v7** wg §8.1; Router v3 retunować dopiero przy zmianie modelu w głosie.

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
