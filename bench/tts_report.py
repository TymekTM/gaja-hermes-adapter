#!/usr/bin/env python3
"""Render the TTS comparison tables into docs/TTS_RESULTS.md from bench/results.

Reads bench/results/tts_<name>.json (+ _probe), bench/results/tts_quality_proxy.json and, for the
old runs, the same tts_*.json files without loadavg fields. Only the block between the TTS_TABLE
markers is rewritten. Each row names its source JSON. A row is CONTAMINATED when either run's
1-min loadavg at start was above QUIET_LOAD (or the JSON predates the loadavg/contaminated fields).
"""
import json
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
DOC = ROOT / "docs" / "TTS_RESULTS.md"
PROXY = RESULTS / "tts_quality_proxy.json"
BEGIN, END = "<!-- TTS_TABLE:BEGIN -->", "<!-- TTS_TABLE:END -->"

PIPER_PROSODY = "style -> Piper length/noise, WSOLA pitch, pauses, soft clip (apply_post)"
SUPER_PROSODY = "style -> speed (rate), WSOLA pitch, pauses, soft clip (apply_post)"
ENERGY_ONLY = "energy only (no pitch or pauses)"

PL_ROWS = [
    ("Piper pl_PL-darkman-medium", "piper_final_darkman", "alternative", "piper_final_darkman", PIPER_PROSODY),
    ("Piper pl_PL-gosia-medium", "piper_final_gosia", "production default", "piper_final_gosia", PIPER_PROSODY),
    ("Piper pl_PL-mc_speech-medium", "piper_final_mc_speech", "candidate", "piper_final_mc_speech", PIPER_PROSODY),
    ("Piper pl_PL-bass-high", "piper_final_bass-high", "candidate", "piper_final_bass-high", PIPER_PROSODY),
    ("Supertonic-3 M4, 6 steps, 22.05 kHz", "supertonic_final", "config in config.toml", "supertonic_final", SUPER_PROSODY),
    ("Supertonic-3 M4, 6 steps, native 44.1 kHz", "supertonic_native_M4", "reference only", "supertonic_native_M4", SUPER_PROSODY),
    ("MMS mms-tts-pol, CPU (CC-BY-NC)", "mms_final_cpu", "candidate, licence-limited", "mms_final_cpu", ENERGY_ONLY),
]
EN_ROWS = [
    ("Paradee-8M, 1 thread", "paradee_final_en", "production EN default", None, SUPER_PROSODY),
]
NOT_REMEASURED = [
    ("Paradee-8M, 1 thread", "paradee_t1"),
    ("Paradee-8M, 2 threads", "paradee_t2"),
    ("Paradee-8M, 4 threads", "paradee_t4"),
    ("Paradee-8M, Polish via espeak-ng IPA", "paradee_pl_espeak"),
    ("Chatterbox-turbo fp32 (GPU)", "chatterbox-turbo_fp32"),
    ("Chatterbox-turbo fp32, cfm steps 1 (GPU)", "chatterbox-turbo_fp32_cfm1"),
    ("Chatterbox-turbo fp32, expandable segments (GPU)", "chatterbox-turbo_fp32_expseg"),
    ("Chatterbox-turbo fp16 (GPU)", "chatterbox-turbo_fp16"),
]

MEASURED_HEADER = (
    "| Engine / config | Role | SR (Hz) | Device | Threads | RTF median (min-max) | TTFA median (short) | "
    "Cold load (s) | Streaming | Prosody controls | Whistle WER mean / median | faster-whisper WER mean | "
    "Peak dBFS / clip frac | Max pause (s) | Load 1-min start -> end | Status | Source |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)
NOT_HEADER = (
    "| Config | Lang | Device | SR (Hz) | RTF median (min-max) | TTFA median | Whistle WER mean | "
    "Cold load (s) | Failed | Status | Source |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def _load(name):
    path = RESULTS / f"tts_{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _proxy():
    return json.loads(PROXY.read_text(encoding="utf-8"))["voices"] if PROXY.exists() else {}


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def _f(x, digits=2):
    return "n/a" if x is None else f"{x:.{digits}f}"


def _range(xs, digits=2):
    if not xs:
        return "n/a"
    return f"{_f(_median(xs), digits)} ({_f(min(xs), digits)}-{_f(max(xs), digits)})"


def _ok(items):
    return [r for r in items if "error" not in r]


def _sr(doc, lang):
    for r in _ok(doc["langs"][lang]["items"]):
        return str(sf.info(ROOT / r["wav_path"]).samplerate)
    return "n/a"


def _threads(doc):
    a, e = doc["args"], doc["engine"]
    if e == "piper":
        return f"{a['piper_threads']}"
    if e == "supertonic":
        return f"{a['supertonic_threads']}"
    if e == "paradee":
        return f"{a['threads']}"
    return "torch default"


def _status(docs):
    names = ["main", "probe"]
    bad = [f"{names[i]} (start load {d['loadavg_start'][0]:.2f})" if "loadavg_start" in d else f"{names[i]} (no loadavg)"
           for i, d in enumerate(docs) if d.get("contaminated", True)]
    if not bad:
        return "quiet (start load <= 2.0)"
    if len(bad) < len(docs):
        return "timing from quiet run only; contaminated: " + ", ".join(bad)
    return "CONTAMINATED: " + ", ".join(bad)


def _measured_row(label, role, name, proxy_key, prosody, lang, proxy):
    main, probe = _load(name), _load(f"{name}_probe")
    if main is None:
        return f"| {label} | {role} | missing `{name}.json` |"
    docs = [d for d in (main, probe) if d is not None]
    quiet = [d for d in docs if not d.get("contaminated", True)]
    timing_docs = quiet or docs
    items = [r for d in timing_docs for r in _ok(d["langs"][lang]["items"])]
    rtf = [r["rtf"] for r in items if "rtf" in r]
    ttfa = [r["ttfa_s"] for r in items]
    short = [r["ttfa_s"] for r in items if r["cat"] == "short"]
    summ = main["langs"][lang]["summary"]
    wer = f"{_f(summ['wer_mean'], 3)} / {_f(summ['wer']['median'] if summ['wer'] else None, 3)}"
    fw = proxy.get(proxy_key) if proxy_key else None
    fw_mean = _f(fw["wer_fw_mean"], 3) if fw else "n/a"
    sig = (f"{_f(fw['peak_dbfs_max'], 1)} / {fw['clip_frac_999_max']:.4f}" if fw else "n/a")
    pause = _f(fw["max_internal_pause_s_max"], 2) if fw else "n/a"
    loads = " ; ".join(
        f"{d['loadavg_start'][0]:.2f} -> {d['loadavg_end'][0]:.2f}" if "loadavg_start" in d else "unknown"
        for d in docs
    )
    files = ", ".join(f"`tts_{n}.json`" for n in [name, f"{name}_probe"][: len(docs)])
    return (f"| {label} | {role} | {_sr(main, lang)} | {main['device']} | {_threads(main)} | "
            f"{_range(rtf)} | {_f(_median(ttfa))} ({_f(_median(short))}) | {_f(main['load_s'])} | "
            f"{main['streaming']} | {prosody} | {wer} | {fw_mean} | {sig} | {pause} | {loads} | "
            f"{_status(docs)} | {files} |")


def _not_remeasured_rows():
    rows = []
    for label, name in NOT_REMEASURED:
        doc = _load(name)
        if doc is None:
            rows.append(f"| {label} | - | missing `tts_{name}.json` |")
            continue
        for lang, block in doc["langs"].items():
            items = _ok(block["items"])
            rtf = [r["rtf"] for r in items if "rtf" in r]
            ttfa = [r["ttfa_s"] for r in items]
            summ = block["summary"]
            wer = _f(summ["wer_mean"], 3)
            rows.append(
                f"| {label} | {lang} | {doc['device']} | {_sr(doc, lang)} | {_range(rtf)} | {_f(_median(ttfa))} | "
                f"{wer} | {_f(doc.get('load_s'))} | {sum('error' in r for r in block['items'])} | "
                f"NOT RE-MEASURED; no loadavg recorded | `tts_{name}.json` |"
            )
    return rows


def build_block():
    proxy = _proxy()
    parts = ["### Polish (PL)", "",
             "Main set (12 sentences) and probe set (12 sentences) per voice. RTF and TTFA come only from runs "
             "whose start 1-min loadavg was <= 2.0 when one exists. Whistle WER uses the main set. "
             "faster-whisper `small` (CPU, int8, beam 5) runs on both sets; WER does not depend on load.", "",
             MEASURED_HEADER]
    parts += [_measured_row(label, role, name, key, prosody, "pl", proxy) for label, name, role, key, prosody in PL_ROWS]
    parts += ["", "### English (EN)", "",
              "Whistle WER on the EN main set; faster-whisper proxy is PL-only, so it is n/a here.", "", MEASURED_HEADER]
    parts += [_measured_row(label, role, name, key, prosody, "en", proxy) for label, name, role, key, prosody in EN_ROWS]
    parts += ["", "### Not re-measured (old runs, no loadavg; do not compare with the tables above)", "", NOT_HEADER]
    parts += _not_remeasured_rows()
    return "\n".join(parts)


def main():
    block = build_block()
    text = DOC.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"markers {BEGIN} / {END} missing in {DOC}")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    DOC.write_text(f"{head}{BEGIN}\n{block}\n{END}{tail}", encoding="utf-8")
    print(block)


if __name__ == "__main__":
    main()
