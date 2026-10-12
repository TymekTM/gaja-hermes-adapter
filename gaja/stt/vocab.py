"""Custom vocabulary, post-STT correction and approval-based auto-learning for Gaja."""
from __future__ import annotations

import json
import logging
import os
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

log = logging.getLogger(__name__)

# Common Polish words that fuzzy matching must NEVER modify.
COMMON_PL_WORDS = {
    "a", "aby", "ach", "acz", "aczkolwiek", "albo", "ale", "ani", "aż",
    "bardziej", "bardzo", "bez", "bo", "będzie", "będą", "będziemy", "będziecie", "będę",
    "by", "być", "był", "była", "było", "byli", "były", "byłem", "byłam", "byłeś", "byłaś",
    "cały", "cała", "całe", "choć", "chociaż", "chcę", "chcesz", "chce", "chcemy", "chcecie", "chcą",
    "ci", "ciebie", "cię", "co", "coś", "czas", "czasu", "czasem", "czy",
    "dla", "do", "dobra", "dobre", "dobry", "dobrze", "dość", "dwa", "dwie", "dwaj", "dwóch",
    "dzień", "dzisiaj", "dziś",
    "gdy", "gdyby", "gdzie", "go",
    "i", "ich", "ile", "im", "inny", "inna", "inne", "iż",
    "ja", "jak", "jaka", "jaki", "jakie", "jako", "jakoś", "je", "jeden", "jedna", "jedno",
    "jednak", "jego", "jej", "jemu", "jest", "jestem", "jesteś", "jesteśmy", "jesteście", "są",
    "jeszcze", "jeśli", "jeżeli", "już",
    "kiedy", "kilka", "kto", "ktoś", "który", "która", "które", "których", "którym", "ku",
    "lub", "ma", "mają", "mamy", "macie", "mam", "masz", "mało", "mały", "mała", "małe",
    "mnie", "mi", "mój", "moja", "moje", "może", "można", "musi", "muszę", "musisz",
    "na", "nad", "nam", "nami", "nas", "nawet", "nic", "nie", "niego", "niej", "niemu",
    "nigdy", "nim", "nimi", "niż", "no", "nowy", "nowa", "nowe",
    "o", "obok", "od", "około", "on", "ona", "one", "oni", "ono", "oraz", "oto",
    "pan", "pani", "państwo", "po", "pod", "podczas", "pomimo", "ponad", "ponieważ",
    "potem", "powinien", "powinna", "poza", "przed", "przede", "przez", "przy",
    "raz", "razem", "również",
    "sam", "sama", "samo", "się", "skąd", "sobie", "sobą", "sposób", "stąd",
    "ta", "tak", "taki", "taka", "takie", "także", "tam", "tamten", "tamta", "tamto",
    "te", "tego", "tej", "temu", "ten", "teraz", "też", "to", "tobie", "tobą", "trochę", "tu", "tutaj", "twoja", "twoje", "twój", "ty", "tylko", "tysiąc",
    "u",
    "w", "wam", "wami", "was", "wasi", "we", "według", "więc", "wiele", "wielu",
    "właśnie", "wszak", "wszyscy", "wszystko", "wszystkie", "wszystkich", "wtedy",
    "z", "za", "zawsze", "zaraz", "ze", "znowu", "znów", "został", "została", "zostało",
    "że", "żeby",
}

# Polish noun case endings for inflected loanwords.
POLISH_SUFFIXES = (
    "ami", "ach", "owi", "em", "ie", "ów", "om", "ze", "a", "u", "e", "y", "i", "ą", "ę",
)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Fast pure-Python Levenshtein distance."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr[j + 1] = min(insertions, deletions, substitutions)
        prev = curr
    return prev[len(s2)]


@dataclass
class VocabTerm:
    term: str
    aliases: list[str] = field(default_factory=list)
    lang: str | None = None
    builtin: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "term": self.term,
            "aliases": list(self.aliases),
        }
        if self.lang is not None:
            d["lang"] = self.lang
        if self.builtin:
            d["builtin"] = True
        return d


@dataclass
class VocabSuggestion:
    id: str
    term: str
    heard: list[str] = field(default_factory=list)
    count: int = 1
    reason: str = "near-miss"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "term": self.term,
            "heard": list(self.heard),
            "count": self.count,
            "reason": self.reason,
        }


def _dump_vocab_toml(terms: list[VocabTerm]) -> str:
    lines = ["# Gaja custom user vocabulary (auto-generated)\n"]
    for t in terms:
        lines.append("[[terms]]")
        lines.append(f"term = {json.dumps(t.term, ensure_ascii=False)}")
        lines.append(f"aliases = {json.dumps(t.aliases, ensure_ascii=False)}")
        if t.lang:
            lines.append(f"lang = {json.dumps(t.lang, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


class VocabularyManager:
    def __init__(
        self,
        default_path: Path | str | None = None,
        user_path: Path | str | None = None,
        suggestions_path: Path | str | None = None,
        *,
        max_hotwords: int = 50,
        fuzzy: bool = True,
        enabled: bool = True,
    ) -> None:
        self.default_path = Path(default_path) if default_path else Path("prompts/vocab_default.toml")
        self.user_path = Path(user_path) if user_path else Path("data/vocab.toml")
        self.suggestions_path = Path(suggestions_path) if suggestions_path else Path("data/vocab_suggestions.json")
        self.max_hotwords = max_hotwords
        self.fuzzy = fuzzy
        self.enabled = enabled

        self._default_terms: list[VocabTerm] = []
        self._user_terms: list[VocabTerm] = []
        self._suggestions: list[VocabSuggestion] = []
        self.load()

    def load(self) -> None:
        self._default_terms = []
        if self.default_path.exists():
            try:
                with open(self.default_path, "rb") as f:
                    data = tomllib.load(f)
                for entry in data.get("terms", []):
                    self._default_terms.append(
                        VocabTerm(
                            term=entry["term"],
                            aliases=list(entry.get("aliases", [])),
                            lang=entry.get("lang"),
                            builtin=True,
                        )
                    )
            except Exception as exc:
                log.warning("failed to load default vocab from %s: %s", self.default_path, exc)

        self._user_terms = []
        if self.user_path.exists():
            try:
                with open(self.user_path, "rb") as f:
                    data = tomllib.load(f)
                for entry in data.get("terms", []):
                    self._user_terms.append(
                        VocabTerm(
                            term=entry["term"],
                            aliases=list(entry.get("aliases", [])),
                            lang=entry.get("lang"),
                            builtin=False,
                        )
                    )
            except Exception as exc:
                log.warning("failed to load user vocab from %s: %s", self.user_path, exc)

        self._suggestions = []
        if self.suggestions_path.exists():
            try:
                with open(self.suggestions_path, encoding="utf-8") as f:
                    raw = json.load(f)
                for s in raw:
                    self._suggestions.append(
                        VocabSuggestion(
                            id=str(s["id"]),
                            term=s["term"],
                            heard=list(s.get("heard", [])),
                            count=int(s.get("count", 1)),
                            reason=s.get("reason", "near-miss"),
                        )
                    )
            except Exception as exc:
                log.warning("failed to load vocab suggestions from %s: %s", self.suggestions_path, exc)

    def save_user_terms(self) -> None:
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        content = _dump_vocab_toml(self._user_terms)
        tmp = self.user_path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.user_path)

    def save_suggestions(self) -> None:
        self.suggestions_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps([s.to_dict() for s in self._suggestions], indent=2, ensure_ascii=False)
        tmp = self.suggestions_path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.suggestions_path)

    @property
    def all_terms(self) -> list[VocabTerm]:
        return self._default_terms + self._user_terms

    def hotwords(self, max_terms: int | None = None) -> str:
        """Returns hotwords string for faster-whisper transcribe."""
        if not self.enabled:
            return ""
        cap = max_terms or self.max_hotwords
        picked: list[str] = []
        for t in self.all_terms:
            if t.term not in picked:
                picked.append(t.term)
            if len(picked) >= cap:
                break
        return ", ".join(picked)

    def correct(self, text: str) -> str:
        """Post-STT correction: alias replacement, inflection preservation, conservative fuzzy matching."""
        if not self.enabled or not text.strip():
            return text

        result = text

        # 1. Multi-word terms (e.g. "home assistant", "pull request")
        for t in self.all_terms:
            patterns = [t.term] + t.aliases
            for pat in patterns:
                if " " in pat:
                    # Match multi-word with optional inflection on last word
                    words = pat.split()
                    last = words[-1]
                    prefix = " ".join(words[:-1])
                    # Try matching prefix + last with possible case endings
                    pattern_re = rf"(?i)\b{re.escape(prefix)}\s+{re.escape(last)}(?:a|u|y|ów|ach|em|ie|e|owi)?\b"
                    def _replace_multi(m: re.Match[str]) -> str:
                        matched = m.group(0)
                        matched_words = matched.split()
                        last_matched = matched_words[-1]
                        suffix = last_matched[len(last):] if last_matched.lower().startswith(last.lower()) else ""
                        return f"{t.term}{suffix}"
                    result = re.sub(pattern_re, _replace_multi, result)

        # 2. Tokenize by words and non-words to preserve spacing and punctuation
        tokens = re.split(r"(\w+)", result)
        out_tokens: list[str] = []

        for token in tokens:
            if not token.isalnum():
                out_tokens.append(token)
                continue

            cleaned = token
            lower = cleaned.lower()

            # Rule: NEVER touch common Polish words or very short words (< 3 chars unless exact technical term)
            if lower in COMMON_PL_WORDS:
                out_tokens.append(token)
                continue

            replaced = self._correct_single_word(cleaned)
            out_tokens.append(replaced)

        return "".join(out_tokens)

    def _correct_single_word(self, word: str) -> str:
        lower = word.lower()

        # 1. Exact term match
        for t in self.all_terms:
            if lower == t.term.lower():
                return t.term

        # 2. Polish inflection on terms (e.g. Kubernetesa, Dockera, Grafanie)
        for t in self.all_terms:
            t_base = t.term
            if len(t_base) > 3 and t_base[-1].lower() in ("a", "e", "o", "y"):
                t_stem = t_base[:-1].lower()
                stem_base = t_base[:-1]
            else:
                t_stem = t_base.lower()
                stem_base = t_base

            for suff in ("ami", "ach", "owi", "em", "ie", "ów", "om", "ze", "o", "a", "u", "e", "y", "i", "ą", "ę"):
                if lower == t_stem + suff:
                    return f"{stem_base}{suff}"

        # 3. Inflection on aliases (e.g. kubernetisa -> Kubernetesa, dokera -> Dockera)
        for t in self.all_terms:
            t_base = t.term
            if len(t_base) > 3 and t_base[-1].lower() in ("a", "e", "o", "y"):
                stem_base = t_base[:-1]
            else:
                stem_base = t_base

            for alias in t.aliases:
                if len(alias) > 3 and alias[-1].lower() in ("a", "e", "o", "y"):
                    a_stem = alias[:-1].lower()
                else:
                    a_stem = alias.lower()

                for suff in ("ami", "ach", "owi", "em", "ie", "ów", "om", "ze", "o", "a", "u", "e", "y", "i", "ą", "ę"):
                    if lower == a_stem + suff:
                        return f"{stem_base}{suff}"

        # 4. Exact alias match
        for t in self.all_terms:
            for alias in t.aliases:
                if lower == alias.lower():
                    return t.term

        # Conservative fuzzy matching
        if self.fuzzy and len(lower) >= 5 and lower not in COMMON_PL_WORDS:
            for t in self.all_terms:
                t_base = t.term
                t_stem = t_base.lower()
                stem_base = t_base
                if len(t_base) > 4 and t_base[-1].lower() in ("a", "e", "o", "y"):
                    t_stem = t_base[:-1].lower()
                    stem_base = t_base[:-1]

                # Check fuzzy match against stem + possible suffix
                for suff in ("", *POLISH_SUFFIXES):
                    expected = t_stem + suff
                    # Length check: don't compare vastly different lengths
                    if abs(len(lower) - len(expected)) > 2:
                        continue
                    dist = levenshtein_distance(lower, expected)
                    # Threshold: max 1 edit for len 5-7, max 2 edits for len 8+
                    max_allowed = 1 if len(expected) <= 7 else 2
                    ratio = 1.0 - (dist / max(len(lower), len(expected)))
                    if dist <= max_allowed and ratio >= 0.82:
                        # Record near-miss suggestion
                        self._add_suggestion(t.term, lower, "near-miss")
                        return f"{stem_base}{suff}" if suff else t.term

        return word

    def _add_suggestion(self, term: str, heard: str, reason: str) -> None:
        # Check if suggestion for term exists
        for s in self._suggestions:
            if s.term.lower() == term.lower():
                s.count += 1
                if heard not in s.heard:
                    s.heard.append(heard)
                self.save_suggestions()
                return

        new_id = f"s{len(self._suggestions) + 1}"
        s = VocabSuggestion(id=new_id, term=term, heard=[heard], count=1, reason=reason)
        self._suggestions.append(s)
        self.save_suggestions()

    def observe(self, heard_text: str, previous_turn_text: str | None = None) -> None:
        """Collect suggestions from user repetitions/corrections and unknown technical terms."""
        if not self.enabled or not heard_text.strip():
            return

        # 1. User correction / repetition detection
        if previous_turn_text and previous_turn_text.strip():
            prev_words = re.findall(r"\b\w+\b", previous_turn_text.lower())
            curr_words = re.findall(r"\b\w+\b", heard_text.lower())
            # If current utterance repeats previous with one word changed, or phrases like "chodziło o X"
            m = re.search(r"(?:chodziło o|to znaczy|miało być|miałem na myśli|popraw na)\s+([A-Za-z0-9_-]+)", heard_text, re.IGNORECASE)
            if m:
                term = m.group(1)
                # find differing word in previous turn
                differing = [w for w in prev_words if w not in curr_words and w not in COMMON_PL_WORDS]
                heard_w = differing[0] if differing else term.lower()
                self._add_suggestion(term, heard_w, "user-correction")
                return

            if len(prev_words) >= 2 and len(curr_words) >= 2:
                # Compare word-by-word
                diff_prev = [w for w in prev_words if w not in curr_words and w not in COMMON_PL_WORDS]
                diff_curr = [w for w in curr_words if w not in prev_words and w not in COMMON_PL_WORDS]
                if len(diff_prev) == 1 and len(diff_curr) == 1:
                    w_prev = diff_prev[0]
                    w_curr = diff_curr[0]
                    if len(w_curr) >= 4:
                        # User corrected w_prev to w_curr
                        self._add_suggestion(w_curr.capitalize(), w_prev, "user-correction")

        # 2. Unknown capitalized tokens in input that might be terms
        raw_tokens = re.findall(r"\b[A-Z][a-zA-Z0-9_-]{3,}\b", heard_text)
        known = {t.term.lower() for t in self.all_terms}
        for token in raw_tokens:
            if token.lower() not in known and token.lower() not in COMMON_PL_WORDS:
                self._add_suggestion(token, token.lower(), "unknown-token")

    def get_api_payload(self) -> dict[str, Any]:
        return {
            "terms": [t.to_dict() for t in self.all_terms],
            "suggestions": [s.to_dict() for s in self._suggestions],
        }

    def put_terms(self, new_terms: list[dict[str, Any]]) -> dict[str, Any]:
        """Replaces user terms with new terms. Default terms remain read-only and builtin."""
        builtins = {t.term.lower() for t in self._default_terms}
        cleaned_user_terms: list[VocabTerm] = []
        for entry in new_terms:
            term = str(entry.get("term", "")).strip()
            if not term or term.lower() in builtins:
                continue
            aliases = [str(a).strip() for a in entry.get("aliases", []) if str(a).strip()]
            lang = entry.get("lang")
            cleaned_user_terms.append(VocabTerm(term=term, aliases=aliases, lang=lang, builtin=False))

        self._user_terms = cleaned_user_terms
        self.save_user_terms()
        return self.get_api_payload()

    def approve_suggestion(self, suggestion_id: str) -> dict[str, Any]:
        idx = -1
        for i, s in enumerate(self._suggestions):
            if s.id == suggestion_id:
                idx = i
                break
        if idx == -1:
            raise HTTPException(404, detail="suggestion not found")

        s = self._suggestions.pop(idx)
        # Add to user terms if not already present
        exists = any(t.term.lower() == s.term.lower() for t in self.all_terms)
        if not exists:
            aliases = [h for h in s.heard if h.lower() != s.term.lower()]
            self._user_terms.append(VocabTerm(term=s.term, aliases=aliases, builtin=False))
            self.save_user_terms()
        self.save_suggestions()
        return self.get_api_payload()

    def reject_suggestion(self, suggestion_id: str) -> dict[str, Any]:
        idx = -1
        for i, s in enumerate(self._suggestions):
            if s.id == suggestion_id:
                idx = i
                break
        if idx == -1:
            raise HTTPException(404, detail="suggestion not found")

        self._suggestions.pop(idx)
        self.save_suggestions()
        return self.get_api_payload()


def create_vocab_router(
    settings_or_manager: Any,
    origin_ok: Callable[[str | None], bool],
    manager: VocabularyManager | None = None,
) -> APIRouter:
    if isinstance(settings_or_manager, VocabularyManager):
        mgr = settings_or_manager
    elif manager is not None:
        mgr = manager
    else:
        # settings_or_manager is Settings
        vocab_cfg = getattr(settings_or_manager, "vocab", None)
        if vocab_cfg:
            from gaja.config import resolve
            mgr = VocabularyManager(
                default_path=resolve(vocab_cfg.default_path),
                user_path=resolve(vocab_cfg.user_path),
                suggestions_path=resolve(vocab_cfg.suggestions_path),
                max_hotwords=vocab_cfg.max_hotwords,
                fuzzy=vocab_cfg.fuzzy,
                enabled=vocab_cfg.enabled,
            )
        else:
            mgr = VocabularyManager()

    def check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if not origin_ok(origin):
            log.warning("rejected vocab request from origin %r", origin)
            raise HTTPException(403, detail="origin not allowed")

    def _get_mgr(request: Request) -> VocabularyManager:
        services = getattr(getattr(request.app, "state", None), "services", None)
        return getattr(services, "vocab", None) or mgr

    router = APIRouter(prefix="/api/vocab", dependencies=[Depends(check_origin)])

    @router.get("")
    async def get_vocab(request: Request) -> dict[str, Any]:
        return _get_mgr(request).get_api_payload()

    @router.put("")
    async def put_vocab(request: Request) -> dict[str, Any]:
        body = await request.json()
        terms = body.get("terms", [])
        return _get_mgr(request).put_terms(terms)

    @router.post("/suggestions/{suggestion_id}/approve")
    async def approve_suggestion(suggestion_id: str, request: Request) -> dict[str, Any]:
        return _get_mgr(request).approve_suggestion(suggestion_id)

    @router.post("/suggestions/{suggestion_id}/reject")
    async def reject_suggestion(suggestion_id: str, request: Request) -> dict[str, Any]:
        return _get_mgr(request).reject_suggestion(suggestion_id)

    return router
