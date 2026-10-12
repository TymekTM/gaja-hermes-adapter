#!/usr/bin/env python3
"""False-positive evaluation for Gaja STT vocabulary post-correction.

Evaluates correct() over all local Polish sentences:
- bench/data/stt/pl/refs.tsv
- bench/data/tools_eval.jsonl
- bench/data/tools_heldout.jsonl
- bench/sentences_pl.txt

Lists every changed token and checks that zero ordinary Polish words are changed.
Outputs results to bench/results/vocab_fp_eval.json.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gaja.stt.vocab import VocabularyManager


def collect_sentences(repo_root: Path) -> list[dict[str, str]]:
    corpus: list[dict[str, str]] = []

    # 1. refs.tsv
    refs_p = repo_root / "bench/data/stt/pl/refs.tsv"
    if refs_p.exists():
        with open(refs_p, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("ref"):
                    corpus.append({"source": "bench/data/stt/pl/refs.tsv:ref", "text": row["ref"]})
                if row.get("raw_ref"):
                    corpus.append({"source": "bench/data/stt/pl/refs.tsv:raw_ref", "text": row["raw_ref"]})

    # 2. tools_eval.jsonl
    te_p = repo_root / "bench/data/tools_eval.jsonl"
    if te_p.exists():
        with open(te_p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                for k in ("text", "prompt", "user", "utterance"):
                    if k in d and isinstance(d[k], str) and d[k].strip():
                        corpus.append({"source": "bench/data/tools_eval.jsonl", "text": d[k].strip()})
                        break

    # 3. tools_heldout.jsonl
    th_p = repo_root / "bench/data/tools_heldout.jsonl"
    if th_p.exists():
        with open(th_p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                for k in ("text", "prompt", "user", "utterance"):
                    if k in d and isinstance(d[k], str) and d[k].strip():
                        corpus.append({"source": "bench/data/tools_heldout.jsonl", "text": d[k].strip()})
                        break

    # 4. bench/sentences_pl.txt
    sp_p = repo_root / "bench/sentences_pl.txt"
    if sp_p.exists():
        with open(sp_p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    corpus.append({"source": "bench/sentences_pl.txt", "text": line.strip()})

    return corpus


def evaluate_fp(repo_root: Path) -> dict[str, Any]:
    vm = VocabularyManager(
        default_path=repo_root / "prompts/vocab_default.toml",
        user_path=repo_root / "data/vocab.toml",
        suggestions_path=repo_root / "data/vocab_suggestions.json",
        enabled=True,
        fuzzy=True,
    )

    known_stems = set()
    for t in vm.all_terms:
        known_stems.add(t.term.lower())
        for a in t.aliases:
            known_stems.add(a.lower())

    corpus = collect_sentences(repo_root)
    changed_records = []
    ordinary_word_false_positives = []

    for entry in corpus:
        orig = entry["text"]
        corr = vm.correct(orig)
        if corr != orig:
            orig_tokens = re.findall(r"\b\w+\b", orig)
            corr_tokens = re.findall(r"\b\w+\b", corr)

            diffs = []
            for ot, ct in zip(orig_tokens, corr_tokens):
                if ot != ct:
                    # Check if this change belongs to a known technical term
                    is_term_change = any(ot.lower().startswith(stem[:4]) or ct.lower().startswith(stem[:4]) for stem in known_stems)
                    diffs.append({
                        "original_token": ot,
                        "corrected_token": ct,
                        "is_term_change": is_term_change,
                    })
                    if not is_term_change:
                        ordinary_word_false_positives.append({
                            "source": entry["source"],
                            "original_token": ot,
                            "corrected_token": ct,
                            "sentence": orig,
                        })

            changed_records.append({
                "source": entry["source"],
                "original": orig,
                "corrected": corr,
                "token_changes": diffs,
            })

    summary = {
        "total_sentences": len(corpus),
        "changed_sentences": len(changed_records),
        "ordinary_word_false_positives_count": len(ordinary_word_false_positives),
        "false_positive_rate": len(ordinary_word_false_positives) / max(1, len(corpus)),
        "changes": changed_records,
        "ordinary_word_false_positives": ordinary_word_false_positives,
    }

    out_file = repo_root / "bench/results/vocab_fp_eval.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"=== Vocab False-Positive Evaluation ===")
    print(f"Total Polish sentences evaluated: {len(corpus)}")
    print(f"Sentences with vocabulary corrections: {len(changed_records)}")
    print(f"Ordinary Polish words modified (false positives): {len(ordinary_word_false_positives)}")
    print(f"Saved results to: {out_file}")
    for cr in changed_records:
        for tc in cr["token_changes"]:
            status = "KNOWN_TERM" if tc["is_term_change"] else "FALSE_POSITIVE"
            print(f"  [{status}] {tc['original_token']} -> {tc['corrected_token']} ({cr['source']})")

    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    evaluate_fp(root)
