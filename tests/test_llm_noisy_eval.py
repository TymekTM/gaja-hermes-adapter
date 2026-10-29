from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bench.llm_noisy_eval import (
    claims_unexecuted_action,
    compute_metrics,
    contains_english_sentence,
    evaluate_case_accuracy,
    is_english_sentence,
)
from gaja.config import Settings
from gaja.tools.registry import build_runtime_registry

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "bench/data/llm_pl_noisy.jsonl"


def test_english_heuristic_polish_sentences() -> None:
    polish_samples = [
        "Jutro w Katowicach będzie padać deszcz.",
        "Dzień dobry, w czym mogę Ci dzisiaj pomóc?",
        "Temperatura w salonie wynosi dwadzieścia jeden stopni.",
        "Włączam klimatyzację w sypialni.",
        "Pięć plus siedem to dwanaście.",
        "Jasne, już sprawdzam to w Wikipedii.",
        "Kiedy wypada najbliższe święto państwowe?",
        "Zrób trochę ciszej na głośniku.",
        "Zapamiętałam, że lubisz gorzką czekoladę.",
        "Ustawiono minutnik na dziesięć minut.",
        "Nie jestem pewna, o co dokładnie pytasz.",
    ]
    for sample in polish_samples:
        assert not contains_english_sentence(sample), f"False positive for Polish: {sample}"


def test_english_heuristic_english_sentences() -> None:
    english_samples = [
        "Yeah, I was asking a lot.",
        "It will be rainy tomorrow in London.",
        "I cannot answer this question right now.",
        "Sorry, I did not understand that request.",
        "Let me know if you need anything else.",
        "Please tell me what time it is.",
        "Turn off the lights in the living room.",
        "Have a nice day and see you later.",
    ]
    for sample in english_samples:
        assert contains_english_sentence(sample), f"False negative for English: {sample}"


def test_english_heuristic_mixed_content() -> None:
    # Model appends an English sentence to a Polish reply
    mixed_1 = "Jutro w Warszawie będzie 15 stopni. Let me know if you need anything else."
    assert contains_english_sentence(mixed_1)

    mixed_2 = "Jasne, włączam klimatyzację. Have a great day!"
    assert contains_english_sentence(mixed_2)

    # XML tags like <think> should be ignored when checking content
    with_tags = "<think>some internal thought</think> Jutro w Krakowie będzie słonecznie."
    # The Polish part is not English, but if thought is English it may be flagged if not stripped
    # Our contains_english_sentence strips tags
    clean_pl = "<think></think> Dzień dobry, jak mogę pomóc?"
    assert not contains_english_sentence(clean_pl)


def test_noisy_dataset_schema() -> None:
    assert DATA_PATH.exists(), f"Missing dataset file at {DATA_PATH}"

    os.environ["HASS_URL"] = "http://127.0.0.1:1"
    os.environ["HASS_TOKEN"] = "dummy"
    registry = build_runtime_registry(Settings())
    valid_tool_names = set(registry.names())

    assert len(valid_tool_names) == 15

    cases = []
    seen_ids = set()
    covered_tools = set()
    clean_count = 0
    asr_count = 0

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            cases.append(case)

            # Check required keys
            for key in ("id", "utterance", "clean", "expect", "noise"):
                assert key in case, f"Line {line_num} missing key: {key}"

            cid = case["id"]
            assert cid not in seen_ids, f"Line {line_num}: Duplicate id: {cid}"
            seen_ids.add(cid)

            noise = case["noise"]
            assert noise in ("clean", "asr"), f"Line {line_num}: Invalid noise value: {noise}"
            if noise == "clean":
                clean_count += 1
                assert case["utterance"] == case["clean"], f"Line {line_num}: clean utterance must match clean field"
            else:
                asr_count += 1

            expect = case["expect"]
            assert "kind" in expect, f"Line {line_num}: missing expect.kind"
            assert "tool" in expect, f"Line {line_num}: missing expect.tool"
            assert "args_contain" in expect, f"Line {line_num}: missing expect.args_contain"
            assert isinstance(expect["args_contain"], dict)

            kind = expect["kind"]
            assert kind in ("tool", "reply", "hermes"), f"Line {line_num}: Invalid kind: {kind}"

            tool = expect["tool"]
            if kind == "tool":
                assert tool in valid_tool_names, f"Line {line_num}: Tool {tool} not in registry {valid_tool_names}"
                covered_tools.add(tool)
            else:
                assert tool is None, f"Line {line_num}: Expected tool must be null for kind={kind}"

    # Overall dataset properties
    assert len(cases) >= 190, f"Expected around 200 cases, got {len(cases)}"
    assert abs(clean_count - asr_count) <= 5, f"Expected roughly 50/50 clean vs asr: {clean_count} vs {asr_count}"

    # Every tool in the registry must be covered by at least one case
    missing_tools = valid_tool_names - covered_tools
    assert not missing_tools, f"Registry tools not covered in dataset: {missing_tools}"


def test_evaluate_case_accuracy() -> None:
    case_tool = {
        "expect": {"kind": "tool", "tool": "get_weather", "args_contain": {"location": "Katowice"}}
    }
    res_correct = {
        "route_chosen": "tool",
        "tool_called": "get_weather",
        "tool_arguments": {"location": "Katowice", "when": "now"},
    }
    route_ok, tool_ok = evaluate_case_accuracy(case_tool, res_correct)
    assert route_ok and tool_ok

    res_wrong_args = {
        "route_chosen": "tool",
        "tool_called": "get_weather",
        "tool_arguments": {"location": "Kraków"},
    }
    route_ok, tool_ok = evaluate_case_accuracy(case_tool, res_wrong_args)
    assert route_ok and not tool_ok

    res_wrong_route = {
        "route_chosen": "reply",
        "tool_called": None,
        "tool_arguments": None,
    }
    route_ok, tool_ok = evaluate_case_accuracy(case_tool, res_wrong_route)
    assert not route_ok and not tool_ok


def test_claims_unexecuted_action() -> None:
    # The headline failure ("mówi, a nie robi"): an announcement or a done-claim with no call behind it.
    assert claims_unexecuted_action(
        "Jasne, przekazuję to Hermesowi… Wywołaj narzędzie handoff_to_hermes", executed=False
    )
    assert claims_unexecuted_action("Jasne, przekazuję to Hermesowi.", executed=False)
    assert claims_unexecuted_action("Zrobiłem to teraz.", executed=False)
    assert not claims_unexecuted_action("Dwa plus dwa to cztery.", executed=False)
    # The same text is harmless when the router or a tool call actually executed something.
    assert not claims_unexecuted_action("Zrobiłem to teraz.", executed=True)
    assert not claims_unexecuted_action("handoff_to_hermes(task=\"x\")", executed=True)


def test_compute_metrics_counts_claimed_actions() -> None:
    case = {"expect": {"kind": "reply", "tool": None, "args_contain": {}}}
    def result(claimed: bool) -> dict:
        return {
            "case": case,
            "route_chosen": "reply",
            "route_correct": True,
            "tool_called": None,
            "tool_correct": True,
            "has_english": False,
            "has_think": False,
            "finish_reason": "stop",
            "latency_ms": 100.0,
            "completion_tokens": 10,
            "claimed_action_without_call": claimed,
        }
    metrics = compute_metrics([result(True), result(False), result(False)])
    assert metrics["claimed_action_rate"] == round(1 / 3 * 100.0, 1)
    assert compute_metrics([])["claimed_action_rate"] == 0.0
