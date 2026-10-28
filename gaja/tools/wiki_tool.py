from __future__ import annotations

import re
import urllib.parse
from typing import Any

import httpx

from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import ToolResult, clean_speakable

ABBREVIATIONS_SPLIT = {
    "łac", "niem", "ur", "zm", "ok", "m.in", "tzw", "np", "r", "w", "św", "gr", "ang",
    "dr", "prof", "hab", "ul", "al", "pl", "tzn", "itp", "itd", "jw", "pt", "vs", "art",
    "ust", "par", "mr", "mrs", "ms", "jr", "sr", "etc", "i.e", "e.g", "approx", "b", "d", "c", "ca",
}

RE_INLINE_ETYMOLOGY = re.compile(
    r"(?:,\s*(?:łac\.|niem\.|gr\.|ang\.|franc\.|fr\.|ros\.|wł\.|hebr\.|arab\.|starogr\.|sanskryt)\s+[^,\–\—\(\)]+)+",
    re.I,
)
RE_PARENTHETICAL = re.compile(r"\s*\([^\)]*?\)")
RE_BRACKETS_IPA = re.compile(r"\[[^\]]*?\]|/[^/]{2,}?/")
RE_DASHES = re.compile(r"\s*([–—])\s*")


def _clean_wiki_extract(text: str) -> str:
    cleaned = text.replace("\xa0", " ")
    # Remove IPA and bracketed pronunciation
    cleaned = RE_BRACKETS_IPA.sub("", cleaned)
    # Remove parenthesized clauses (dates, pronunciation, etymology)
    cleaned = RE_PARENTHETICAL.sub("", cleaned)
    # Remove inline etymology clauses e.g. ", łac. Nicolaus Copernicus, niem. Nikolaus Kopernikus"
    cleaned = RE_INLINE_ETYMOLOGY.sub("", cleaned)
    # Format dashes cleanly
    cleaned = RE_DASHES.sub(r" \1 ", cleaned)
    cleaned = clean_speakable(cleaned)
    return cleaned


def _split_sentences_smart(text: str) -> list[str]:
    tokens = text.split()
    sentences = []
    curr: list[str] = []
    for i, tok in enumerate(tokens):
        curr.append(tok)
        if re.search(r"[.!?]$", tok):
            clean_word = re.sub(r"^[^\w]+|[^\w.]+$", "", tok).lower().rstrip(".")
            if clean_word in ABBREVIATIONS_SPLIT:
                continue
            if i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                if next_tok and (next_tok[0].isupper() or next_tok[0] in '"„«'):
                    sentences.append(" ".join(curr))
                    curr = []
            else:
                sentences.append(" ".join(curr))
                curr = []
    if curr:
        sentences.append(" ".join(curr))
    return [s.strip() for s in sentences if s.strip()]


def _extract_sentences(text: str, max_sentences: int = 2) -> str:
    cleaned = _clean_wiki_extract(text)
    sentences = _split_sentences_smart(cleaned)
    selected = sentences[:max_sentences]
    return " ".join(selected).strip()


async def _search_title(client: httpx.AsyncClient, wiki_lang: str, topic: str) -> str | None:
    query = re.sub(r"\s*\([^)]*\)", "", topic).strip() or topic
    try:
        resp = await client.get(
            f"https://{wiki_lang}.wikipedia.org/w/rest.php/v1/search/title",
            params={"q": query, "limit": 1},
        )
        if resp.status_code != 200:
            return None
        pages = resp.json().get("pages") or []
        return pages[0].get("key") if pages else None
    except (httpx.HTTPError, ValueError):
        return None


async def wiki_summary(
    topic: str,
    lang: str = "pl",
    *,
    client: httpx.AsyncClient | None = None,
) -> ToolResult:
    """Fetch short 1-2 sentence Wikipedia summary for a topic."""
    if not topic or not topic.strip():
        err = "Brak tematu do wyszukania w Wikipedii"
        speak = "Nie podano tematu." if lang == "pl" else "No topic given."
        return ToolResult(data={"error": err}, speak=speak, error=err)

    clean_topic = topic.strip().replace(" ", "_")
    wiki_lang = "pl" if lang != "en" else "en"

    own_client = False
    if client is None:
        client = httpx.AsyncClient(
            timeout=2.5,
            headers={"User-Agent": "Gaja-voice-assistant/1.0 (https://github.com/TymekTM/gaja-hermes-adapter)"},
        )
        own_client = True

    try:
        url = f"https://{wiki_lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(clean_topic)}"
        resp = await client.get(url)

        if resp.status_code == 404:
            # The selector often produces an inflected or decorated title ("Lalke", "Hermes (mitologia)");
            # let Wikipedia's title search resolve it to the real article.
            found = await _search_title(client, wiki_lang, topic)
            if found:
                resp = await client.get(
                    f"https://{wiki_lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(found.replace(' ', '_'))}"
                )

        if resp.status_code == 404:
            err = f"Brak artykułu dla '{topic}' w Wikipedii"
            speak = f"Nie znalazłam artykułu o {topic} w Wikipedii." if wiki_lang == "pl" else f"Could not find Wikipedia article for {topic}."
            return ToolResult(data={"error": err, "topic": topic}, speak=speak, error=err)

        if resp.status_code != 200:
            err = f"Wikipedia API HTTP {resp.status_code}"
            speak = (
                "Nie udało się pobrać informacji z Wikipedii."
                if lang == "pl"
                else "Could not fetch information from Wikipedia."
            )
            return ToolResult(data={"error": err}, speak=speak, error=err)

        data = resp.json()
        title = data.get("title", topic)
        extract = data.get("extract", "")
        if not extract:
            err = f"Brak treści podsumowania dla '{topic}'"
            speak = f"Artykuł o {topic} nie zawiera podsumowania." if wiki_lang == "pl" else f"No summary for {topic}."
            return ToolResult(data={"error": err, "title": title}, speak=speak, error=err)

        short_summary = _extract_sentences(extract, max_sentences=2)
        spoken_summary = normalize_for_tts(short_summary, wiki_lang)
        res_data = {
            "title": title,
            "summary": short_summary,
            "lang": wiki_lang,
        }
        return ToolResult(data=res_data, speak=spoken_summary)
    except Exception as exc:
        err = f"Błąd Wikipedii: {exc}"
        speak = "Nie udało się sprawdzić informacji w Wikipedii." if wiki_lang == "pl" else "Could not fetch Wikipedia information."
        return ToolResult(data={"error": err}, speak=speak, error=err)
    finally:
        if own_client:
            await client.aclose()
