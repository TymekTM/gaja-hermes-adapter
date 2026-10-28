#!/usr/bin/env python3
"""Evaluate the production Router (gaja/llm/router.py) on the routing and tools datasets.

Runs the real Router class against llama-server :8791, exactly as the pipeline uses it (router slot,
prime before each call, history_turns from config.toml). Two configurations:

  v2  prompts/router_pl.md, 2 classes, no tools      (the router deployed before round 6)
  v3  prompts/router3_pl.md, 3 classes + tool select (round 6)

Raw class masses are stored per item, so thresholds are swept offline (``--sweep``) without
re-querying the model.

Metrics (per dataset group and overall):
  hermes_recall_strict  HERMES-labelled items whose need is NOT covered by a local tool -> routed hermes
  covered_served        HERMES-labelled items a local tool covers (time, weather, currency, server
                        status, indoor temperature / AC) -> routed tool or hermes (anything but local)
  false_handoff_local   LOCAL-labelled items routed to hermes
  hermes_precision      routed hermes items that are labelled hermes (covered ones count as correct)
  action_acc            tools datasets: route == target_action (v2: tool targets count when routed hermes)
  tool_acc / args_acc   tools datasets: right tool / right arguments among tool targets (v3)

routing_live holds the 2026-10-26 device incident utterances (media/Spotify volume turn family); it is
reported as its own group and counts in "all". Items whose route differs once the local spotify tool
is logged in carry a "label_spotify_ready" field; pass --spotify-ready to score that variant
(without it the labels describe a house where Spotify is not configured locally, i.e. Hermes).

Usage:
  flock /tmp/gaja-bench.lock .venv/bin/python bench/router3_eval.py            # run both, write results
  .venv/bin/python bench/router3_eval.py --sweep bench/results/router3_eval_raw.jsonl
"""

from __future__ import annotations

import re

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.tools_eval import check_arg_accuracy  # noqa: E402
from gaja.config import Settings, resolve  # noqa: E402
from gaja.llm.client import LLMClient  # noqa: E402
from gaja.llm.parse import ToolCall  # noqa: E402
from gaja.llm.router import (  # noqa: E402
    AT_HOME,
    HERMES_ACTION,
    MYTHOLOGY,
    SPOTIFY_DEVICE_TRANSFER,
    TOOL_CUES,
    Router,
    ToolUnavailable,
    addresses_hermes,
    clock_fast_path,
    cue_plausible,
    home_imperative_route,
    home_status_route,
    home_volume_route,
    local_tool_override,
    any_plausible,
    media_handoff,
    plausible,
    spotify_fast_path,
    volume_fast_path,
)
from gaja.tools import default_registry  # noqa: E402
from gaja.tools.weather_tool import resolve_tool_location  # noqa: E402

DATA = ROOT / "bench/data"
RESULTS = ROOT / "bench/results"
DATASETS = {
    "routing": DATA / "routing.jsonl",
    "routing_holdout": DATA / "routing_holdout.jsonl",
    "routing_live": DATA / "routing_live.jsonl",
    "routing_home": DATA / "routing_home.jsonl",
    "tools_eval": DATA / "tools_eval.jsonl",
    "tools_heldout": DATA / "tools_heldout.jsonl",
}
# HERMES-labelled routing items whose need a local tool now covers (time/date, weather incl. sunrise,
# NBP currency, server status, indoor temperature / bedroom AC). A tool or Hermes answer is right for
# these; a local LLM answer is not.
COVERED_CATEGORIES = {"time", "weather", "sun", "holiday", "timer", "memory", "spotify", "home"}
COVERED_IDS = {
    "pl-h09", "pl-hld48", "en-hld18",  # currency
    "pl-h24", "pl-h27", "en-h07", "pl-hld53",  # server status
    "pl-h45", "en-h09", "pl-hld58", "en-hld20",  # weather follow-ups
    "pl-b17", "pl-h22", "pl-hld45",  # indoor temperature / bedroom heating & AC
}


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected(item: dict[str, Any], *, spotify_ready: bool = False) -> tuple[str, bool]:
    """(label, covered): label in local/tool/hermes."""
    if "target_action" in item:
        return item["target_action"], False
    label = item["label"]
    if spotify_ready and "label_spotify_ready" in item:
        label = item["label_spotify_ready"]
    covered = item["label"] == "hermes" and (item["category"] in COVERED_CATEGORIES or item["id"] in COVERED_IDS)
    return label, covered


def postprocess_call(
    call: dict[str, Any] | None,
    item: dict[str, Any],
    default_location: str = "Katowice",
    *,
    spotify_ready: bool = True,
) -> dict[str, Any] | None:
    """Apply the adapter pipeline's post-processing (plausible + resolve_tool_location + when)."""
    if call is None:
        return None
    name = call["name"]
    arguments = dict(call.get("arguments") or {})
    text = item["text"]
    history = item.get("history") or []
    # r5 6e: plausible() gates spotify on the tool's readiness exactly like the router does.
    if not plausible(ToolCall(name, arguments), text, history[-2:], spotify_ready=spotify_ready):
        return None
    if name in ("get_weather", "get_sun"):
        arguments["location"] = resolve_tool_location(
            arguments.get("location"),
            text,
            default_location,
            history=history,
            last_location=item.get("last_location"),
        )
        if name == "get_weather":
            if re.search(r"\bjutro\b|\btomorrow\b", text, re.I):
                arguments["when"] = "tomorrow"
            elif "when" not in arguments and re.search(
                r"\bdzi[sś]\b|\bdzisiaj\b|\bw\s+weekend\b|\btoday\b", text, re.I
            ):
                arguments["when"] = "today"
    return {"name": name, "arguments": arguments}


def decide(
    row: dict[str, Any],
    version: str,
    threshold: float,
    tool_threshold: float,
    *,
    spotify_ready: bool = False,
) -> str:
    local, tool, hermes = row["masses"]
    text = row["text"]
    history = (row.get("item") or {}).get("history") or []
    if version == "v2":
        total = local + tool + hermes
        p = hermes / total if total else 0.0
        return "hermes" if p >= threshold else "local"
    if addresses_hermes(text):
        return "hermes"
    if clock_fast_path(text):
        return "tool"
    mythology = MYTHOLOGY.search(text)
    # Mirror of Router.classify: the deterministic media rule fires before the model masses do, and
    # the local spotify fast paths only when the tool is ready (token file present).
    if not mythology and media_handoff(text, history, spotify_ready=spotify_ready):
        return "hermes"
    if spotify_ready and spotify_fast_path(text, history):
        return "tool"
    if volume_fast_path(text):
        return "tool"
    # Mirror of Router.classify: a media-player volume command over a household device, an on/off
    # imperative over a household device and a read-only status question over a household-
    # configured device are served by the local home tool without the model (NEW home fast
    # paths). classify() guards only the STATUS path with MYTHOLOGY — volume/imperative fire
    # regardless — so the mirror keeps each path's own guard (F7).
    if "home" in set(default_registry.names()):
        if home_volume_route(text) or home_imperative_route(text):
            return "tool"
        if not mythology and home_status_route(text):
            return "tool"
    # Mirror of Router.classify: an unregistered tool's cue sends the turn to Hermes (H4).
    registered = set(default_registry.names())
    if not mythology and any(
        name not in registered and TOOL_CUES[name].search(text) for name in ("home", "spotify")
    ):
        return "hermes"
    if hermes >= threshold and not mythology:
        return "hermes"
    is_local_override = local_tool_override(text, set(default_registry.names()))
    if (
        not mythology
        and not is_local_override
        and (HERMES_ACTION.search(text) or AT_HOME.search(text) or SPOTIFY_DEVICE_TRANSFER.search(text))
    ):
        return "hermes"
    if tool >= tool_threshold:
        # Mirror of Router.classify (r4e): a TOOL-classified turn about spotify with no ready local
        # tool goes to Hermes ("tool-unavailable") instead of a doomed call or a local reply. The
        # cue check ignores the spotify fast path on purpose ("pauza" stays local, main parity).
        if not spotify_ready and "spotify" in registered and cue_plausible("spotify", text, history[-2:]):
            return "hermes"
        # r5 6e: mirror Router.select_tool — with no plausible tool at all it never generates,
        # so the raw call (however spotify-shaped) is not evaluated and the turn stays local.
        if not any_plausible(
            set(default_registry.names()), text, history[-2:], spotify_ready=spotify_ready
        ):
            return "local"
        if "call" in row:  # no plausible call: local, unless the model picked the unusable tool
            call = postprocess_call(row["call"], row["item"], spotify_ready=spotify_ready)
            if call is None:
                if (
                    not spotify_ready
                    and isinstance(row["call"], dict)
                    and row["call"].get("name") == "spotify"
                ):
                    return "hermes"  # select_tool raises ToolUnavailable -> Hermes handoff
                return "local"
        return "tool"
    return "local"


def metrics(
    rows: list[dict[str, Any]],
    version: str,
    threshold: float,
    tool_threshold: float,
    *,
    spotify_ready: bool = False,
) -> dict[str, Any]:
    strict_n = strict_ok = covered_n = covered_ok = local_n = local_fh = 0
    routed_h = routed_h_ok = 0
    action_n = action_ok = tool_n = tool_ok = args_ok = 0
    tool_target_local = 0
    for row in rows:
        route = decide(row, version, threshold, tool_threshold, spotify_ready=spotify_ready)
        label, covered = row["label"], row["covered"]
        if route == "hermes":
            routed_h += 1
            routed_h_ok += label == "hermes" or covered
        if label == "local":
            local_n += 1
            local_fh += route == "hermes"
        if label == "hermes" and covered:
            covered_n += 1
            covered_ok += route in ("tool", "hermes") if version == "v3" else route == "hermes"
        elif label == "hermes":
            strict_n += 1
            strict_ok += route == "hermes"
        if row["dataset"].startswith("tools"):
            action_n += 1
            if version == "v2":
                action_ok += route == label or (label == "tool" and route == "hermes")
            else:
                action_ok += route == label
            if label == "tool":
                tool_n += 1
                tool_target_local += route == "local"
                if version == "v3" and route == "tool":
                    call = postprocess_call(row.get("call"), row["item"])
                    if call and call["name"] == row["target_tool"]:
                        tool_ok += 1
                        args_ok += check_arg_accuracy(call["name"], call["arguments"], row["item"])

    def pct(a: int, b: int) -> float | None:
        return round(100 * a / b, 1) if b else None

    return {
        "threshold": threshold,
        "tool_threshold": tool_threshold,
        "hermes_recall_strict": pct(strict_ok, strict_n),
        "hermes_strict_n": strict_n,
        "covered_served": pct(covered_ok, covered_n),
        "covered_n": covered_n,
        "false_handoff_local": pct(local_fh, local_n),
        "local_n": local_n,
        "hermes_precision": pct(routed_h_ok, routed_h),
        "routed_hermes": routed_h,
        "action_acc": pct(action_ok, action_n),
        "action_n": action_n,
        "tool_acc": pct(tool_ok, tool_n) if version == "v3" else None,
        "args_acc": pct(args_ok, tool_n) if version == "v3" else None,
        "tool_targets_answered_locally": tool_target_local,
        "tool_n": tool_n,
    }


async def run(
    version: str,
    settings: Settings,
    items: list[tuple[str, dict[str, Any]]],
    v3_prompt: str,
    select: bool,
    *,
    spotify_ready: bool = False,
) -> list[dict[str, Any]]:
    llm = LLMClient(settings.llm, [])
    schemas = default_registry.schemas() if version == "v3" else None
    prompt = v3_prompt if version == "v3" else "prompts/router_pl.md"
    router = Router.from_file(
        llm,
        resolve(Path(prompt)),
        threshold=0.0,
        history_turns=settings.router.history_turns,
        timeout_s=10.0,
        slot=settings.llm.router_slot,
        tools=schemas,
        tool_threshold=0.5,
        spotify_ready=(lambda: True) if spotify_ready else None,
    )
    rows: list[dict[str, Any]] = []
    try:
        for dataset, item in items:
            history = item.get("history") or []
            await router.prime(history)
            prompt_text = router._prompt(item["text"], history)
            started = time.perf_counter()
            completion = await llm.complete(prompt_text, timeout_s=10.0, slot=settings.llm.router_slot)
            classify_ms = (time.perf_counter() - started) * 1000
            from gaja.llm.router import class_masses

            masses = class_masses(completion)
            label, covered = expected(item)
            label_ready, _ = expected(item, spotify_ready=True)
            row = {
                "version": version,
                "dataset": dataset,
                "id": item["id"],
                "text": item["text"],
                "label": label,
                "label_spotify_ready": label_ready,
                "covered": covered,
                "target_tool": item.get("target_tool"),
                "masses": [round(m, 5) for m in masses],
                "classify_ms": round(classify_ms, 1),
                "item": item,
            }
            has_fast_path = bool(
                clock_fast_path(item["text"])
                or (spotify_ready and spotify_fast_path(item["text"], history) is not None)
                or volume_fast_path(item["text"])
                # Mirror of Router.select_tool's deterministic home calls (F7): volume and
                # imperative ride unguarded, the status path keeps classify()'s MYTHOLOGY
                # guard, and all three need the home tool registered.
                or (
                    "home" in set(default_registry.names())
                    and (
                        home_volume_route(item["text"])
                        or home_imperative_route(item["text"])
                        or (not MYTHOLOGY.search(item["text"]) and home_status_route(item["text"]))
                    )
                )
            )
            if select and version == "v3" and (masses[1] >= 0.2 or label == "tool" or has_fast_path):
                started = time.perf_counter()
                try:
                    call = await router.select_tool(item["text"], history)  # None when implausible
                except ToolUnavailable:
                    # Readiness leak, r4e: the model picked the not-ready spotify tool; the pipeline
                    # hands the turn to Hermes, so record exactly that routing outcome.
                    call, tool_unavailable = None, True
                else:
                    tool_unavailable = False
                row["select_ms"] = round((time.perf_counter() - started) * 1000, 1)
                raw_call = (
                    {"name": "spotify", "arguments": {}} if tool_unavailable
                    else {"name": call.name, "arguments": call.arguments} if call else None
                )
                row["raw_call"] = raw_call
                row["call"] = postprocess_call(
                raw_call, item, settings.tools.default_location, spotify_ready=spotify_ready
            )
            rows.append(row)
    finally:
        await llm.aclose()
    return rows


def report(
    rows: list[dict[str, Any]],
    thresholds: dict[str, tuple[float, float]],
    *,
    spotify_ready: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for version, (threshold, tool_threshold) in thresholds.items():
        vrows = [r for r in rows if r["version"] == version]
        if not vrows:
            continue
        for row in vrows:
            if spotify_ready and "label_spotify_ready" in row:
                row["label"] = row["label_spotify_ready"]
        groups = {
            "routing": [r for r in vrows if r["dataset"] == "routing"],
            "routing_holdout": [r for r in vrows if r["dataset"] == "routing_holdout"],
            "routing_live": [r for r in vrows if r["dataset"] == "routing_live"],
            "routing_home": [r for r in vrows if r["dataset"] == "routing_home"],
            "tools_eval": [r for r in vrows if r["dataset"] == "tools_eval"],
            "tools_heldout": [r for r in vrows if r["dataset"] == "tools_heldout"],
            "all": vrows,
        }
        out[version] = {
            name: metrics(g, version, threshold, tool_threshold, spotify_ready=spotify_ready)
            for name, g in groups.items()
            if g
        }
        ms = [r["classify_ms"] for r in vrows]
        sel = [r["select_ms"] for r in vrows if "select_ms" in r]
        out[version]["latency_ms"] = {
            "classify_p50": round(statistics.median(ms), 1),
            "classify_p90": round(sorted(ms)[int(0.9 * (len(ms) - 1))], 1),
            "select_p50": round(statistics.median(sel), 1) if sel else None,
            "select_p90": round(sorted(sel)[int(0.9 * (len(sel) - 1))], 1) if sel else None,
        }
    return out


def errors(rows: list[dict[str, Any]], version: str, threshold: float, tool_threshold: float, spotify_ready: bool = False) -> list[str]:
    lines = []
    for r in rows:
        if r["version"] != version:
            continue
        route = decide(r, version, threshold, tool_threshold)
        label = r["label"]
        ok = route == label or (r["covered"] and route in ("tool", "hermes"))
        if version == "v3" and route == "tool" and label == "tool":
            call = postprocess_call(r.get("call"), r["item"], spotify_ready=spotify_ready) or {}
            ok = call.get("name") == r["target_tool"]
        if not ok:
            l, t, h = r["masses"]
            lines.append(f"{r['id']:12} {label:6}->{route:6} L{l:.2f} T{t:.2f} H{h:.2f} {r['text']!r} {r.get('call')}")
    return lines


def sweep(rows: list[dict[str, Any]]) -> None:
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97):
        for tool_threshold in (0.3, 0.4, 0.5, 0.6):
            m = metrics([r for r in rows if r["version"] == "v3"], "v3", threshold, tool_threshold)
            print(
                f"v3 th={threshold:.2f} tool={tool_threshold:.1f}  strict_recall={m['hermes_recall_strict']} "
                f"covered={m['covered_served']} false_local={m['false_handoff_local']} "
                f"precision={m['hermes_precision']} action={m['action_acc']} tool={m['tool_acc']}"
            )
    for threshold in (0.5, 0.8, 0.9, 0.97):
        m = metrics([r for r in rows if r["version"] == "v2"], "v2", threshold, 0.5)
        print(
            f"v2 th={threshold:.2f}  strict_recall={m['hermes_recall_strict']} covered={m['covered_served']} "
            f"false_local={m['false_handoff_local']} precision={m['hermes_precision']} action={m['action_acc']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", default="v2,v3")
    ap.add_argument("--sweep", type=Path, help="raw jsonl from a previous run; print a threshold sweep")
    ap.add_argument("--v3-threshold", type=float, default=0.5)
    ap.add_argument("--v3-tool-threshold", type=float, default=0.6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--v3-prompt", default="prompts/router3_pl.md")
    ap.add_argument("--no-select", action="store_true", help="classify only (tool/args accuracy not measured)")
    ap.add_argument("--spotify-ready", action="store_true",
                    help="score with the local spotify tool ready (logged in): fast paths serve it and "
                         "items with a label_spotify_ready field are scored against that label")
    ap.add_argument("--datasets", default="",
                    help="comma-separated subset of: routing,routing_holdout,routing_live,routing_home,"
                         "tools_eval,tools_heldout (default: all)")
    args = ap.parse_args()
    thresholds = {"v2": (0.97, 0.5), "v3": (args.v3_threshold, args.v3_tool_threshold)}

    if args.sweep:
        rows = [json.loads(line) for line in args.sweep.read_text(encoding="utf-8").splitlines()]
        sweep(rows)
        print(json.dumps(report(rows, thresholds, spotify_ready=args.spotify_ready), indent=1))
        for version, (th, tt) in thresholds.items():
            print(f"--- {version} errors at {th}/{tt}")
            print("\n".join(errors(rows, version, th, tt, spotify_ready=spotify_ready)))
        return

    # NEW-6: the generic example household stands in for the private data/home.toml, and its
    # "u <person>" cues are injected exactly as server.py does at start-up.
    from gaja.llm.router import (
        home_device_words,
        home_person_words,
        set_home_device_words,
        set_home_person_words,
    )

    settings = Settings(home={"household_file": ROOT / "docs/home.example.toml"})
    set_home_person_words(home_person_words(settings.home.aliases))
    set_home_device_words(home_device_words(settings.home.aliases))
    selected = [d for d in args.datasets.split(",") if d] or list(DATASETS)
    unknown = [d for d in selected if d not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown datasets: {unknown}; known: {', '.join(DATASETS)}")
    items = [(name, item) for name in selected for item in load(DATASETS[name])]
    load_start = os.getloadavg()
    rows: list[dict[str, Any]] = []
    for version in args.versions.split(","):
        rows += asyncio.run(
            run(version, settings, items, args.v3_prompt, not args.no_select, spotify_ready=args.spotify_ready)
        )
    load_end = os.getloadavg()
    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    raw = RESULTS / f"router3_eval_raw{suffix}.jsonl"
    raw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {
        "loadavg_start": load_start,
        "loadavg_end": load_end,
        "items": len(items),
        "v3_prompt": args.v3_prompt,
        "spotify_ready": args.spotify_ready,
        **report(rows, thresholds, spotify_ready=args.spotify_ready),
    }
    (RESULTS / f"router3_eval{suffix}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"raw: {raw}")


if __name__ == "__main__":
    main()
