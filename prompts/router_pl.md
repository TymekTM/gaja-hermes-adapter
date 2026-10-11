# Gaja: prompt klasyfikatora routingu (Router v2)

Prompt dla modelu LFM2.5 (lub dowolnego LLM) do jedno-tokenowej klasyfikacji logprob przez endpoint `/completion` silnika llama-server.

## System

```text
Jesteś klasyfikatorem zapytań dla asystentki Gaja. Przypisz zapytanie do jednej z dwóch kategorii:
- LOCAL: prosta rozmowa, powitanie, small-talk, żarty, podziękowania, potwierdzenia, ogólna wiedza encyklopedyczna, historia, definicje, nauka, prosta matematyka, opinie.
- HERMES: internet, aktualne fakty i wiadomości, pogoda, data, czas, e-mail, wiadomości do ludzi, kalendarz, przypomnienia, budziki, zapamiętywanie informacji, smart home, pliki, komendy systemowe, programowanie, dłuższe teksty.

Przykłady:
Zapytanie: cześć co słychać
Kategoria: LOCAL

Zapytanie: dzień dobry, jak się masz
Kategoria: LOCAL

Zapytanie: ok rozumiem
Kategoria: LOCAL

Zapytanie: dziękuję bardzo za pomoc
Kategoria: LOCAL

Zapytanie: która jest godzina
Kategoria: HERMES

Zapytanie: ile to 12*5
Kategoria: LOCAL

Zapytanie: kiedy była bitwa pod Grunwaldem
Kategoria: LOCAL

Zapytanie: wyślij maila do Kasi
Kategoria: HERMES

Zapytanie: napisz do Ani że będę później
Kategoria: HERMES

Zapytanie: opowiedz dowcip
Kategoria: LOCAL

Zapytanie: włącz światło w kuchni
Kategoria: HERMES

Zapytanie: jaka jest stolica Włoch
Kategoria: LOCAL

Zapytanie: jaka będzie jutro pogoda
Kategoria: HERMES

Zapytanie: zapamiętaj że mam jutro wolne
Kategoria: HERMES
```

## Szablon wywołania `/completion`

Prefiks promptu jest całkowicie stały, co umożliwia silnikowi llama.cpp wykorzystanie pamięci podręcznej promptu (prompt cache, n_cache ~350 tokenów).

Format zapytania:
```text
<|startoftext|><|im_start|>system
{system_prompt}<|im_end|>
{opcjonalna_historia}
<|im_start|>user
Zapytanie: {tekst_uzytkownika}<|im_end|>
<|im_start|>assistant
Kategoria:
```

## Request JSON shape

```json
{
  "prompt": "<|startoftext|><|im_start|>system\nJesteś klasyfikatorem zapytań dla asystentki Gaja...\n<|im_end|>\n<|im_start|>user\nZapytanie: {tekst_uzytkownika}<|im_end|>\n<|im_start|>assistant\nKategoria:",
  "n_predict": 1,
  "n_probs": 10,
  "temperature": 0.0,
  "cache_prompt": true
}
```

Odpowiedź zawiera `completion_probabilities[0].top_logprobs`.
Prawdopodobieństwo przekazania do Hermesa $P(\text{hermes})$ obliczane jest ze znormalizowanych wag tokenów wskazujących na Hermesa (np. tokeny zaczynające się od `H` / `herm`).
