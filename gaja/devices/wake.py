"""Fuzzy match of the spoken wake phrase ("Hej Gaja") at the start of a transcript.

The device already detected the phrase on-device; this is the server-side second opinion, run on a Whisper
transcript. Whisper spells the phrase in many ways ("Hej, Gaja.", "Hey Gaia", "Ej Gajo", "Hejgaja"), so the
greeting and the name are matched against variant sets (diacritics and case ignored), with one edit of
tolerance on the name only when it starts with "g" (so "maja", "kaja" never match).

Policy (documented in docs/DEVICE.md):
- greeting + name, within the first ``MAX_LEAD`` words, name directly after the greeting: match;
- a bare name ("Gaja, jaka jest pogoda") as the very FIRST word: match when ``accept_bare_name`` (default), because
  the device has already heard the full phrase; the name anywhere else without a greeting does not match.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_LEAD = 2  # filler words allowed before the greeting ("no hej Gaja")
GREETINGS = frozenset({"hej", "hey", "ej", "ey", "hei", "hay", "hejj", "hejze", "haj", "he", "heja"})
NAMES = frozenset({"gaja", "gajo", "gaju", "gaje", "gaia", "gaya", "gaiya", "gajah", "gajia", "gaija", "gayah"})
FUZZY_NAME_ROOTS = ("gaja", "gajo", "gaju")
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_LEADING_PUNCT = " \t\r\n,.;:!?-–—…\"'()"


def normalize(word: str) -> str:
    """Lower case, no Polish (or other) diacritics: 'Gaję' -> 'gaje', 'łódź' -> 'lodz'."""
    word = word.casefold().replace("ł", "l")
    decomposed = unicodedata.normalize("NFKD", word)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class WakeMatch:
    matched: bool
    rest: str = ""  # the transcript after the wake phrase (leading punctuation removed); the full text if no match
    phrase: str = ""  # the words that were taken as the wake phrase
    bare: bool = False  # matched through the bare-name rule (no greeting)


class WakeMatcher:
    def __init__(
        self,
        *,
        accept_bare_name: bool = True,
        extra_names: tuple[str, ...] | list[str] = (),
        extra_greetings: tuple[str, ...] | list[str] = (),
    ) -> None:
        self.accept_bare_name = accept_bare_name
        self._names = NAMES | {normalize(n) for n in extra_names}
        self._greetings = GREETINGS | {normalize(g) for g in extra_greetings}

    def is_name(self, token: str, *, fuzzy: bool) -> bool:
        if token in self._names:
            return True
        return fuzzy and len(token) >= 3 and token[0] == "g" and any(edit_distance(token, r) <= 1 for r in FUZZY_NAME_ROOTS)

    def _split_merged(self, token: str) -> bool:
        """'hejgaja' -> greeting + name in one token."""
        return any(token.startswith(g) and self.is_name(token[len(g) :], fuzzy=False) for g in self._greetings)

    def match(self, text: str) -> WakeMatch:
        spans = [(m.start(), m.end(), normalize(m.group())) for m in _TOKEN.finditer(text)]
        if not spans:
            return WakeMatch(False, text.strip())
        for i, (_, end, token) in enumerate(spans[: MAX_LEAD + 1]):
            if token in self._greetings and i + 1 < len(spans) and self.is_name(spans[i + 1][2], fuzzy=True):
                last = spans[i + 1][1]
                return self._found(text, spans[0][0], last, bare=False)
            if self._split_merged(token):
                return self._found(text, spans[0][0], end, bare=False)
        if self.accept_bare_name and self.is_name(spans[0][2], fuzzy=False):
            return self._found(text, spans[0][0], spans[0][1], bare=True)
        return WakeMatch(False, text.strip())

    @staticmethod
    def _found(text: str, start: int, end: int, *, bare: bool) -> WakeMatch:
        return WakeMatch(True, text[end:].lstrip(_LEADING_PUNCT).strip(), text[start:end], bare)


DEFAULT_MATCHER = WakeMatcher()


def match_wake_phrase(text: str, *, accept_bare_name: bool = True) -> WakeMatch:
    return (DEFAULT_MATCHER if accept_bare_name else WakeMatcher(accept_bare_name=False)).match(text)
