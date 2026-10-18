from __future__ import annotations

import re

_STYLE_TAG = re.compile(r"\[[A-Za-z_]{1,20}\]")
_STYLE_TAG_PREFIX = re.compile(r"\[[A-Za-z_]{0,20}")
_CLOSERS = "\"'”’»)"
_PUNCT = ".!?…"
_WORD = re.compile(r"\S+")
# Phrase-boundary punctuation (besides sentence ends) where an early first-chunk cut sounds natural.
_PHRASE_PUNCT = ",;:–—"
# Subordinating conjunctions / relative pronouns: a clause starts here, so a cut BEFORE one keeps
# both halves prosodically whole. Coordinating "i", "a", "oraz" are left out on purpose: they join
# nouns as often as clauses ("między Krakowem i Warszawą").
_CLAUSE_STARTERS = frozenset(
    {
        "że", "żeby", "aby", "gdy", "kiedy", "jeśli", "jeżeli", "ponieważ", "bo", "ale", "lecz", "więc",
        "dlatego", "chociaż", "choć", "czyli", "gdzie", "jednak", "natomiast", "który", "która", "które",
        "którego", "której", "których", "którym", "którymi", "którą",
    }
)
_LAST_WORD = re.compile(r"([^\s.!?…\"'”’»()\[\]]+)$")
_ABBREVIATIONS = frozenset(
    {
        "np", "godz", "ul", "dr", "mgr", "inż", "tzn", "tj", "itd", "itp", "prof", "nr", "al", "wg", "tel",
        "św", "zob", "hab", "ks", "gen", "płk", "ur", "os", "pn", "wt", "śr", "czw", "pt", "sob", "niedz",
        "mld", "mln", "tys", "in", "etc", "vs",
    }
)


def strip_style_tags(text: str) -> str:
    return _STYLE_TAG.sub("", text)


def speakable(text: str) -> str:
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_#`>|]+", " ", text)
    return " ".join(text.split())


def split_sentences(text: str, *, min_words: int = 3, max_chars: int = 220) -> list[str]:
    chunker = SentenceChunker(first_min_words=min_words, min_words=min_words, max_chars=max_chars)
    return chunker.feed(text) + chunker.flush()


class SentenceChunker:
    """Splits a streamed reply into speakable sentences, holding text until a sentence end is certain.

    The first sentence is also cut at a phrase boundary (comma, semicolon, colon, dash) once it has
    `first_min_words` words. Past `first_max_words` words without one (0 = no cap), it is cut before a clause-starting
    conjunction, and only past twice the cap at a plain word boundary. Piper renders every chunk as
    its own utterance (falling intonation, trailing silence), so a cut in the middle of a phrase is
    heard as a stutter: the early cut buys first-audio latency only where a pause is natural.
    """

    def __init__(self, *, first_min_words: int, min_words: int, max_chars: int, first_max_words: int = 0) -> None:
        self._first_min_words = first_min_words
        self._min_words = min_words
        self._max_chars = max_chars
        self._first_max_words = first_max_words
        self._buf = ""
        self._pending = ""
        self._emitted = False

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        return self._take(final=False)

    def flush(self) -> list[str]:
        out = self._take(final=True)
        if self._buf.strip():
            self._accept(self._buf, out, final=True)
            self._buf = ""
        if self._pending:
            out.append(self._pending)
            self._pending = ""
        return out

    def _take(self, final: bool) -> list[str]:
        out: list[str] = []
        while (cut := self._find_cut(final)) is not None:
            sentence, self._buf = self._buf[:cut], self._buf[cut:].lstrip()
            self._accept(sentence, out, final=False)
        return out

    def _accept(self, sentence: str, out: list[str], *, final: bool) -> None:
        text = sentence.strip()
        if not text:
            return
        combined = f"{self._pending} {text}".strip()
        if not final and _word_count(combined) < self._min_words:
            self._pending = combined
            return
        self._pending = ""
        out.append(combined)
        self._emitted = True

    def _find_cut(self, final: bool) -> int | None:
        buf = self._buf
        for i, ch in enumerate(buf):
            if ch == "\n":
                return i
            if ch in _PUNCT:
                end = self._sentence_end(i, final)
                if end is None:
                    return None
                if end > 0:
                    return end
            elif ch in _PHRASE_PUNCT and not self._emitted and _word_count(buf[:i]) >= self._first_min_words:
                if i + 1 >= len(buf):
                    if final:
                        return i + 1
                    return None
                if buf[i + 1].isspace():
                    return i + 1
        if not self._emitted and self._first_max_words > 0 and _word_count(buf) > self._first_max_words:
            cut = _phrase_cut(buf, self._first_min_words)
            if cut is None:
                cut = _word_boundary(buf, 2 * self._first_max_words)
            if cut is not None:
                return cut
        if len(buf) >= self._max_chars:
            space = buf.rfind(" ", 0, self._max_chars)
            if space > 0:
                return space
        return None

    def _sentence_end(self, i: int, final: bool) -> int | None:
        """Index just past the sentence end; -1 if this punctuation is not a boundary; None if undecidable yet."""
        buf = self._buf
        n = len(buf)
        j = i + 1
        if buf[i] == ".":
            while j < n and buf[j] == ".":
                j += 1
        k = j
        while k < n and buf[k] in _CLOSERS:
            k += 1
        if k >= n and not final:
            return None
        if k < n and not (buf[k].isspace() or buf[k] == "["):
            return -1
        if buf[i] == "." and j - i == 1:
            word = _last_word(buf[:i])
            if word is not None and (word.lower() in _ABBREVIATIONS or (len(word) == 1 and word.isupper())):
                return -1
            if word is not None and word.isdigit() and len(word) <= 2:
                following = buf[k:].lstrip()
                if not following and not final:
                    return None
                if following and following[0].islower():
                    return -1
        return k


def _word_boundary(buf: str, words: int) -> int | None:
    count = 0
    for match in _WORD.finditer(buf):
        if _STYLE_TAG.fullmatch(match.group()):
            continue
        count += 1
        if count == words:
            return match.end() if match.end() < len(buf) else None
    return None


def _phrase_cut(buf: str, min_words: int) -> int | None:
    """Early cut in an over-long first chunk: before the first complete clause-starting word with
    `min_words` or more words before it (phrase punctuation past `min_words` is cut earlier, in
    `_find_cut`; a shorter chunk would only be merged into the next one). None when there is none."""
    count = 0
    for match in _WORD.finditer(buf):
        if _STYLE_TAG.fullmatch(match.group()):
            continue
        if match.end() < len(buf) and count >= min_words and match.group().lower() in _CLAUSE_STARTERS:
            return match.start()
        count += 1
    return None


def _last_word(before: str) -> str | None:
    match = _LAST_WORD.search(before)
    return match.group(1) if match else None


def _word_count(text: str) -> int:
    return len(strip_style_tags(text).split())


_SPEAKER_LABELS = ("gaja", "assistant", "asystent", "asystentka")
_SPEAKER_PREFIX = re.compile(r"\s*(?:\*\*)?(?:gaja|assistant|asystent(?:ka)?)(?:\*\*)?\s*:\s*", re.IGNORECASE)
_SPEAKER_PARTIAL = re.compile(r"\s*(?:\*\*)?([^\W\d_]*)(?:\*\*?)?\s*")


class SpeakerPrefixStripper:
    """Drops a leading "Gaja:" speaker label the model sometimes writes at the start of a reply."""

    def __init__(self) -> None:
        self._head: str | None = ""

    def feed(self, delta: str) -> str:
        if self._head is None:
            return delta
        head = self._head + delta
        match = _SPEAKER_PREFIX.match(head)
        if match and match.end() < len(head):
            self._head = None
            return head[match.end() :]
        if match:
            self._head = head
            return ""
        partial = _SPEAKER_PARTIAL.fullmatch(head)
        if partial and any(label.startswith(partial.group(1).lower()) for label in _SPEAKER_LABELS):
            self._head = head
            return ""
        self._head = None
        return head

    def flush(self) -> str:
        head, self._head = self._head or "", None
        match = _SPEAKER_PREFIX.match(head)
        return head[match.end() :] if match else head


class StyleTagStripper:
    """Removes [tag] markers from streamed text, holding back a tag that may continue in the next delta."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, delta: str) -> str:
        data = self._pending + delta
        self._pending = ""
        out: list[str] = []
        i = 0
        while True:
            start = data.find("[", i)
            if start < 0:
                out.append(data[i:])
                break
            out.append(data[i:start])
            close = data.find("]", start)
            if close < 0:
                if _STYLE_TAG_PREFIX.fullmatch(data[start:]):
                    self._pending = data[start:]
                else:
                    out.append(data[start:])
                break
            if _STYLE_TAG.fullmatch(data[start : close + 1]):
                i = close + 1
            else:
                out.append("[")
                i = start + 1
        return "".join(out)

    def flush(self) -> str:
        pending, self._pending = self._pending, ""
        return pending
