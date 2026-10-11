from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

@dataclass(frozen=True)
class Style:
    emotion: str = "neutral"
    rate: float = 1.0
    pitch_semitones: float = 0.0
    energy: float = 1.0
    events: tuple[str, ...] = ()


STYLE_TAGS = ["neutral", "happy", "excited", "calm", "sad", "serious", "fast", "slow", "laugh", "sigh"]

EMOTION_TAGS = ("neutral", "happy", "excited", "calm", "sad", "serious")
RATE_TAGS = ("fast", "slow")
EVENT_TAGS = ("laugh", "sigh")

STYLE_PRESETS: dict[str, Style] = {
    "neutral": Style("neutral", rate=1.0, pitch_semitones=0.0, energy=1.0),
    "happy": Style("happy", rate=1.05, pitch_semitones=1.2, energy=1.1),
    "excited": Style("excited", rate=1.12, pitch_semitones=2.0, energy=1.2),
    "calm": Style("calm", rate=0.92, pitch_semitones=-0.6, energy=0.85),
    "sad": Style("sad", rate=0.85, pitch_semitones=-1.5, energy=0.8),
    "serious": Style("serious", rate=0.95, pitch_semitones=-1.0, energy=1.05),
    "fast": Style("neutral", rate=1.18),
    "slow": Style("neutral", rate=0.86),
    "laugh": Style("neutral", events=("laugh",)),
    "sigh": Style("neutral", events=("sigh",)),
}

RATE_RANGE = (0.8, 1.25)
PITCH_RANGE = (-2.5, 2.5)
ENERGY_RANGE = (0.6, 1.4)

_ALIASES = {
    "radośnie": "happy",
    "wesoło": "happy",
    "entuzjastycznie": "excited",
    "ekscytująco": "excited",
    "smutno": "sad",
    "spokojnie": "calm",
    "poważnie": "serious",
    "neutralnie": "neutral",
    "szybko": "fast",
    "wolno": "slow",
    "śmiech": "laugh",
    "śmieje się": "laugh",
    "westchnienie": "sigh",
    "wzdycha": "sigh",
}
_TAG_MAP = {tag: tag for tag in STYLE_TAGS} | _ALIASES

_TAG_RE = re.compile(r"\[([^\[\]\n]{0,40})\]")
_PARTIAL_TAG_RE = re.compile(r"\[[^\[\]\n]{0,40}$")
_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'»”)]*(?=\s|$)|\n")
_ABBREVIATIONS = {"dr", "mgr", "np", "itd", "itp", "tzw", "ul", "św", "prof", "inż", "godz", "tj", "tzn"}


def _canonical(raw: str) -> str | None:
    key = " ".join(unicodedata.normalize("NFC", raw).casefold().split())
    return _TAG_MAP.get(key)


def _tokenize(text: str) -> list[tuple[str, str | None]]:
    tokens: list[tuple[str, str | None]] = []
    parts = _TAG_RE.split(text)
    for i, part in enumerate(parts):
        if i % 2 == 1:
            tokens.append(("tag", _canonical(part)))
            continue
        if i == len(parts) - 1:
            partial = _PARTIAL_TAG_RE.search(part)
            if partial:
                part = part[: partial.start()]
        part = part.replace("[", "").replace("]", "")
        if part:
            tokens.append(("text", part))
    return tokens


def _is_abbreviation(chunk: str, end_index: int) -> bool:
    word = re.search(r"(\w+)$", chunk[:end_index])
    return bool(word) and word.group(1).casefold() in _ABBREVIATIONS


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def _build_style(tags: list[str], default: Style) -> Style:
    emotional = [t for t in tags if t in EMOTION_TAGS]
    rate_factor = 1.0
    for t in tags:
        if t in RATE_TAGS:
            rate_factor *= STYLE_PRESETS[t].rate
    events = tuple(dict.fromkeys(t for t in tags if t in EVENT_TAGS))
    base = STYLE_PRESETS[emotional[-1]] if emotional else default
    return Style(
        emotion=base.emotion,
        rate=_clamp(base.rate * rate_factor, RATE_RANGE),
        pitch_semitones=_clamp(base.pitch_semitones, PITCH_RANGE),
        energy=_clamp(base.energy, ENERGY_RANGE),
        events=events,
    )


def parse_styled(text: str, default: Style) -> list[tuple[Style, str]]:
    segments: list[tuple[Style, str]] = []
    tags: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        text_out = " ".join(buf.split())
        if text_out:
            segments.append((_build_style(tags, default), text_out))
        buf = ""

    for kind, value in _tokenize(text):
        if kind == "tag":
            if value is None:
                continue
            if buf.strip():
                flush()
                tags.clear()
            tags.append(value)
            continue
        pos = 0
        for m in _SENTENCE_END_RE.finditer(value):
            if m.group(0).startswith(".") and _is_abbreviation(value, m.start()):
                continue
            buf += value[pos:m.end()]
            flush()
            tags.clear()
            pos = m.end()
        buf += value[pos:]
    flush()
    return segments
