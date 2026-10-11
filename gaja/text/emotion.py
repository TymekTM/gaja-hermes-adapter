from __future__ import annotations

import re

from gaja.text.style import STYLE_PRESETS, Style

_EXPLICIT_TAG = re.compile(r"\[[^\[\]\n]{0,40}\]")


# Stems match inflected Polish (leading \b only); short whole words keep both \b ("super" is not
# "supermarket"). Calm, sad and serious are checked before happy; "dobrze" and a lone "!" are not
# emotions (a plain "Dobrze, zaraz sprawdzę." stays neutral).
def _rule(stems: str, words: str = "", extra: str = "") -> re.Pattern[str]:
    parts = [rf"\b(?:{stems})"]
    if words:
        parts.append(rf"\b(?:{words})\b")
    if extra:
        parts.append(extra)
    return re.compile("|".join(parts), re.I)


_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("excited", _rule("hurra|wygra[lł]|sukces|niesamowi|fantastyczn|wreszcie|szok", "ekstra|super|mega", r"!{2,}")),
    ("sad", _rule("przykro|smut|niestety|za[lł]ama|chor(?:y|a|e|ob)|martwi|obla[lł]|pora[zż]k|bólu?\\b|strasznie")),
    ("calm", _rule("spok[oó]j|powoli|odpocz|relaks|wycisz|oddech|oddych|wypocz|cisz[aeyę]")),
    ("serious", _rule("powa[zż]n|ryzyk|kryzys|bezpiecze[nń]stw|ostrze[zż]|obowi[aą]z|uwaga\\b")),
    ("happy", _rule("ciesz|rado[sś]|gratul|urodzin|[sś]wietn", "mi[lł]o|fajnie")),
)


def has_explicit_tag(text: str) -> bool:
    return _EXPLICIT_TAG.search(text) is not None


def classify_emotion(text: str) -> str:
    for emotion, pattern in _RULES:
        if pattern.search(text):
            return emotion
    return "neutral"


def auto_style(text: str) -> Style:
    preset = STYLE_PRESETS[classify_emotion(text)]
    return Style(emotion=preset.emotion, rate=preset.rate, pitch_semitones=preset.pitch_semitones, energy=preset.energy)
