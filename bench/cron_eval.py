#!/usr/bin/env python3
"""Mini-eval 'cron dla Hermesa': LFM3-1.7B tłumaczy polski opis zadania na JSON z harmonogramem.

10 przypadków typowych prostych cronów domowych. Metryka: poprawne pole `schedule`
(wyrażenie cron 5-polowe) + poprawne parametry akcji. Uruchamiane na istniejącym
serwerze bench (port 8794) — startowanym przez llm_server.start().

  flock /tmp/gaja-bench.lock uv run python bench/cron_eval.py --tag lfm3-q4 --variant lfm3_nothink
"""
import argparse
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_server as S  # noqa: E402

RESULTS = S.ROOT / "bench/results"
URL = S.URL + "/v1/chat/completions"

SYSTEM = (
    "Jesteś planistą zadań agenta Hermes. Przekształcasz polski opis powtarzalnego zadania "
    "na dokładnie jeden obiekt JSON bez żadnego komentarza, w formacie:\n"
    '{"schedule": "<5-polowe wyrażenie cron>", "action": "<krótka nazwa_akcji>", '
    '"params": {<parametry z opisu>}}\n'
    "Nazwa_dnia w polu dnia tygodnia zapisywana małą literą po angielsku (mon, tue, ...). "
    "Godziny w formacie 24h. Jeśli opis nie mówi inaczej, dzień miesiąca/miesiąc to *."
)

# (opis, oczekiwany cron, wymagane pary klucz=>wartość w params/expect)
CASES = [
    ("Codziennie o siódmej trzydzieści rano sprawdź pogodę w Krakowie i gdy ma padać, dopisz przypomnienie o parasolu.",
     "30 7 * * *", {"miasto": "kraków"}),
    ("W każdy poniedziałek o 9:00 zrób kopię zapasową bazy danych i zapisz ją na dysku sieciowym.",
     "0 9 * * mon", {"zadanie": "kopia"}),
    ("Pierwszego dnia każdego miesiąca o północy wygeneruj raport zużycia energii za poprzedni miesiąc.",
     "0 0 1 * *", {"zadanie": "raport"}),
    ("Od poniedziałku do piątku o 17:45 sprawdź korki na trasie dom-praca.",
     "45 17 * * mon-fri", {"zadanie": "korki"}),
    ("Każdej soboty i niedzieli o 10:15 uruchom odkurzacz robota w salonie.",
     "15 10 * * sat,sun", {"pokój": "salon"}),
    ("Codziennie o 22:00 sprawdź, czy drzwi garażowe są zamknięte; jeśli nie, wyślij powiadomienie.",
     "0 22 * * *", {"zadanie": "drzwi"}),
    ("Trzydziestego listopada o 12:00 przypomnij o urodzinach Anny.",
     "0 12 30 11 *", {"kto": "anna"}),
    ("Co godzinę zapisz temperaturę z czujnika na zewnątrz do pliku dziennika.",
     "0 * * * *", {"zadanie": "temperatura"}),
    ("W każdą środę o 6:30 rano i w piątek o 18:00 uruchom nawadnianie ogrodu na 15 minut.",
     None, {"zadanie": "nawadnianie"}),  # dwa terminy -> akceptujemy dowolny z nich
    ("Codziennie o 20:30 podsumuj zużycie danych mobilnych w tym miesiącu.",
     "30 20 * * *", {"zadanie": "zużycie"}),
]


def extract_json(text):
    text = text.split("</think>")[-1]
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def cron_norm(c):
    return " ".join(str(x).lower().strip() for x in (c or "").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", default="lfm3_nothink", choices=list(S.VARIANTS))
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--fewshot", action="store_true",
                    help="doklej 2 przykłady z pełnym cronem do promptu systemowego")
    args = ap.parse_args()

    system = SYSTEM
    if args.fewshot:
        system += ("\n\nPrzykłady:\n"
                   'Opis: "Codziennie o 18:45 wyłącz światła w salonie." → '
                   '{"schedule": "45 18 * * *", "action": "lights_off", "params": {"pokoj": "salon"}}\n'
                   'Opis: "W każdy piątek o 8:15 wyślij podsumowanie tygodnia mailem." → '
                   '{"schedule": "15 8 * * fri", "action": "send_weekly_summary", "params": {"kanal": "mail"}}')

    proc = S.start(S.VARIANTS[args.variant], f"cron_server_{args.tag}.log")
    rows = []
    try:
        for i, (desc, want_cron, must) in enumerate(CASES):
            body = {"messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": desc}],
                    "temperature": args.temp, "seed": 42, "max_tokens": 200, "stream": False}
            resp = requests.post(URL, json=body, timeout=300)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content") or ""
            parsed = None
            err = ""
            try:
                parsed = extract_json(content)
            except Exception as e:  # noqa: BLE001
                err = str(e)
            ok_json = parsed is not None
            got_cron = cron_norm(parsed.get("schedule", "")) if ok_json else ""
            if want_cron is None:
                cron_ok = None  # przypadek wieloharmonogramowy: sprawdzamy tylko params
            else:
                cron_ok = got_cron == cron_norm(want_cron)
            params_txt = json.dumps(parsed.get("params", {}), ensure_ascii=False).lower() if ok_json else ""
            params_ok = all(k.lower() in params_txt and str(v).lower() in params_txt for k, v in must.items())
            rows.append({"i": i, "desc": desc, "want_cron": want_cron, "got": parsed,
                         "raw": content[:300], "json_ok": ok_json, "cron_ok": cron_ok,
                         "params_ok": params_ok, "error": err})
            print(f"cron[{i}] json={ok_json} cron={cron_ok} params={params_ok} -> {content[:90]!r}", flush=True)
    finally:
        S.stop(proc)

    n = len(rows)
    summary = {"n": n,
               "json_ok": sum(1 for r in rows if r["json_ok"]),
               "cron_ok": sum(1 for r in rows if r["cron_ok"]),
               "params_ok": sum(1 for r in rows if r["params_ok"]),
               "both_ok": sum(1 for r in rows if r["cron_ok"] and r["params_ok"])}
    out = {"tag": args.tag, "variant": args.variant, "temp": args.temp,
           "gguf": S.GGUF.name, "summary": summary, "rows": rows}
    (RESULTS / f"cron_eval_{args.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
