import json
import statistics
import subprocess
import time
import requests

URL = "http://127.0.0.1:8791/completion"
H_SYSTEM = (
    "Jesteś klasyfikatorem zapytań dla asystentki Gaja. Przypisz zapytanie do jednej z dwóch kategorii:\n"
    "- LOCAL: prosta rozmowa, powitanie, small-talk, żarty, podziękowania, potwierdzenia, ogólna wiedza encyklopedyczna, historia, definicje, nauka, prosta matematyka, opinie.\n"
    "- HERMES: internet, aktualne fakty i wiadomości, pogoda, data, czas, e-mail, wiadomości do ludzi, kalendarz, przypomnienia, budziki, zapamiętywanie informacji, smart home, pliki, komendy systemowe, programowanie, dłuższe teksty.\n\n"
    "Przykłady:\n"
    "Zapytanie: cześć co słychać\nKategoria: LOCAL\n\n"
    "Zapytanie: dzień dobry, jak się masz\nKategoria: LOCAL\n\n"
    "Zapytanie: ok rozumiem\nKategoria: LOCAL\n\n"
    "Zapytanie: dziękuję bardzo za pomoc\nKategoria: LOCAL\n\n"
    "Zapytanie: która jest godzina\nKategoria: HERMES\n\n"
    "Zapytanie: ile to 12*5\nKategoria: LOCAL\n\n"
    "Zapytanie: kiedy była bitwa pod Grunwaldem\nKategoria: LOCAL\n\n"
    "Zapytanie: wyślij maila do Kasi\nKategoria: HERMES\n\n"
    "Zapytanie: napisz do Ani że będę później\nKategoria: HERMES\n\n"
    "Zapytanie: opowiedz dowcip\nKategoria: LOCAL\n\n"
    "Zapytanie: włącz światło w kuchni\nKategoria: HERMES\n\n"
    "Zapytanie: jaka jest stolica Włoch\nKategoria: LOCAL\n\n"
    "Zapytanie: jaka będzie jutro pogoda\nKategoria: HERMES\n\n"
    "Zapytanie: zapamiętaj że mam jutro wolne\nKategoria: HERMES"
)

prompt = (
    "<|startoftext|><|im_start|>system\n" + H_SYSTEM + "<|im_end|>\n"
    "<|im_start|>user\nZapytanie: jaka jest dzisiaj pogoda w Lublinie<|im_end|>\n"
    "<|im_start|>assistant\nKategoria:"
)

body = {
    "prompt": prompt,
    "n_predict": 1,
    "n_probs": 5,
    "temperature": 0.0,
    "cache_prompt": True,
}

# Warm up prompt cache once
requests.post(URL, json=body, timeout=10)

dts = []
for _ in range(20):
    t0 = time.time()
    r = requests.post(URL, json=body, timeout=10)
    r.raise_for_status()
    dts.append((time.time() - t0) * 1000)

load = subprocess.check_output(["cat", "/proc/loadavg"]).decode().strip()
sorted_dts = sorted(dts)
res = {
    "dts_ms": [round(x, 2) for x in dts],
    "mean_ms": round(statistics.mean(dts), 2),
    "median_ms": round(statistics.median(dts), 2),
    "p95_ms": round(sorted_dts[int(len(dts) * 0.95)], 2),
    "min_ms": round(min(dts), 2),
    "max_ms": round(max(dts), 2),
    "loadavg": load,
}
print(json.dumps(res))
