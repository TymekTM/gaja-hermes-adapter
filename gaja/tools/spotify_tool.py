"""Household Spotify Web API tool.

Controls playback on household Spotify Connect devices. Two token sources:

- ``token_source = "file"`` (default): Gaja's own OAuth2 + PKCE token file
  (``data/spotify_token.json``, mode 0600, created by ``scripts/spotify_login.py``).
- ``token_source = "hermes"``: borrow a short-lived access token from Hermes via
  ``scripts/hermes_spotify_token.py`` (Hermes's code does every refresh under its own lock;
  the refresh token never reaches Gaja — see gaja/tools/spotify_hermes.py).

When unconfigured, when no device is available, or when a query is generic/unmatched,
sets ``handoff=True`` so the pipeline hands the turn to Hermes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import secrets
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from gaja.config import Settings, resolve
from gaja.text.normalize import normalize_for_tts, polish_declension
from gaja.tools.base import ToolResult, clean_speakable, current_session_context
from gaja.tools.spotify_hermes import (
    HermesSpotifyToken,
    hermes_access_token,
    hermes_home_for,
    hermes_spotify_ready,
    profile_for_current_turn,
    reset_hermes_spotify_caches,
)

log = logging.getLogger(__name__)

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"
SPOTIFY_SCOPES = "user-read-playback-state user-modify-playback-state user-read-currently-playing"
TOKEN_EXPIRY_SKEW_S = 60.0
PLAYING_STATE_TTL_S = 600.0
INVALID_GRANT_TTL_S = 600.0
DEVICE_CACHE_TTL_S = 60.0

_REFRESH_LOCK = asyncio.Lock()
_LAST_PLAYING_BY_SCOPE: dict[str, tuple[bool, float]] = {}
_TOKEN_CACHE: dict[Path, tuple[int, dict[str, Any]]] = {}
# Token files that failed to parse (mtime-keyed): is_spotify_ready re-reads only on mtime change.
_CORRUPT_TOKEN_CACHE: dict[Path, int] = {}
_PENDING_PERSIST: dict[Path, dict[str, Any]] = {}
_INVALID_GRANT_CACHE: dict[Path, tuple[float, int]] = {}
_DEVICE_CACHE: dict[tuple[Path, str], tuple[dict[str, Any], float]] = {}

_QUERY_FILLER_WORDS: frozenset[str] = frozenset(
    {
        "pusc",
        "wlacz",
        "zagraj",
        "odtworz",
        "graj",
        "na",
        "w",
        "ze",
        "z",
        "spotify",
        "muzyke",
        "muzyka",
        "muzyki",
        "piosenke",
        "piosenka",
        "piosenki",
        "utwor",
        "utwory",
        "artyste",
        "wykonawce",
        "zespol",
        "playliste",
        "playlista",
        "album",
        "prosze",
        "mi",
        "play",
        "on",
        "the",
        "a",
        "an",
        "some",
        "music",
        "song",
        "track",
        "artist",
        "playlist",
    }
)

_GENERIC_RESUME_QUERIES: frozenset[str] = frozenset(
    {
        "",
        "muzyka",
        "muzyke",
        "spotify",
        "odtwarzanie",
        "dalej",
        "music",
        "playback",
    }
)

_GENERIC_INDEFINITE_RE = re.compile(
    r"\b(?:co[sś]|cokolwiek|jak[aą][sś]|jaki[sś]|jakie[sś]|jakiej[sś]|troch[eę]\s+muzyk\w*|something|anything|some\s+music)\b",
    re.IGNORECASE,
)

_GENERIC_PURPOSE_RE = re.compile(
    r"\b(?:do|na)\s+(?:pracy|snu|spania|biegania|treningu|nauki|relaksu|czytania|jazdy|imprezy|impreze|wiecz[oó]r|poranek|t[lł]a)\b"
    r"|\bw\s+tle\b",
    re.IGNORECASE,
)

_MOOD_OR_GENRE_WORDS: frozenset[str] = frozenset(
    {
        "spokojna",
        "spokojnego",
        "spokojne",
        "spokojnej",
        "relaksujaca",
        "relaksujacego",
        "relaksujace",
        "relaksacyjna",
        "relaksacyjnego",
        "relaks",
        "energetyczna",
        "energetycznego",
        "energetyczne",
        "energiczna",
        "energicznego",
        "smutna",
        "smutnego",
        "smutne",
        "wesola",
        "wesolego",
        "wesole",
        "romantyczna",
        "romantycznego",
        "romantyczne",
        "nastrojowa",
        "nastrojowego",
        "cicha",
        "cichego",
        "delikatna",
        "delikatnego",
        "klasyczna",
        "klasycznej",
        "klasyke",
        "klasyka",
        "jazz",
        "jazzu",
        "jazzowa",
        "rock",
        "rocka",
        "rockowa",
        "pop",
        "popowa",
        "metal",
        "metalu",
        "hiphop",
        "rap",
        "techno",
        "house",
        "elektro",
        "elektroniczna",
        "elektronicznej",
        "chill",
        "chillout",
        "lofi",
        "ambient",
        "blues",
        "bluesa",
        "reggae",
        "funk",
        "soul",
        "disco",
        "punk",
        "indie",
        "folk",
        "country",
    }
)


def _playing_scope(scope: str | None = None) -> str:
    if scope:
        return scope
    ctx = current_session_context.get()
    if ctx:
        dev = str(ctx.get("device_id") or "").strip()
        if dev:
            return f"dev:{dev}"
        sess = str(ctx.get("session_id") or "").strip()
        if sess:
            return f"sess:{sess}"
    return "default"


def set_spotify_playing(playing: bool, *, scope: str | None = None) -> None:
    """Record the last known playback state for the current session/device scope."""
    key = _playing_scope(scope)
    if not playing:
        _LAST_PLAYING_BY_SCOPE.pop(key, None)
        return
    _LAST_PLAYING_BY_SCOPE[key] = (True, time.monotonic())


def is_spotify_playing(*, scope: str | None = None) -> bool:
    """Whether Spotify was recently observed or commanded to be playing in this scope."""
    key = _playing_scope(scope)
    entry = _LAST_PLAYING_BY_SCOPE.get(key)
    if entry is None:
        return False
    playing, ts = entry
    if not playing:
        return False
    if (time.monotonic() - ts) > PLAYING_STATE_TTL_S:
        _LAST_PLAYING_BY_SCOPE.pop(key, None)
        return False
    return True


def reset_spotify_caches() -> None:
    """Clear in-memory Spotify caches (used by unit tests for clean isolation)."""
    _LAST_PLAYING_BY_SCOPE.clear()
    _TOKEN_CACHE.clear()
    _PENDING_PERSIST.clear()
    _INVALID_GRANT_CACHE.clear()
    _CORRUPT_TOKEN_CACHE.clear()
    _DEVICE_CACHE.clear()
    reset_hermes_spotify_caches()


def _stat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


def is_spotify_ready(settings: Settings | None = None) -> bool:
    """Return True when Spotify is enabled and a local token source looks usable.

    ``token_source = "file"``: the local token file exists, parses and isn't known-expired.
    A file that fails to parse (corrupt JSON / not an object) counts as not configured (NEW-8):
    media turns then hand off to Hermes instead of failing mid-tool with a generic error. The
    parse result is cached by mtime, so the readiness probe stays a stat + small read.

    ``token_source = "hermes"``: cache-only answer, never a process spawn (this runs on every
    turn); the background maintenance task keeps the cache warm.
    """
    cfg = settings if settings is not None else Settings()
    if not cfg.spotify.enabled:
        return False
    if cfg.spotify.token_source == "hermes":
        return hermes_spotify_ready(cfg.spotify)
    token_path = resolve(cfg.spotify.token_path)
    if token_path in _PENDING_PERSIST:
        return True
    st = _stat_or_none(token_path)
    if st is None:
        return False
    bad = _INVALID_GRANT_CACHE.get(token_path)
    if bad is not None:
        until, bad_mtime = bad
        if time.monotonic() < until and st.st_mtime_ns == bad_mtime:
            return False
    cached = _TOKEN_CACHE.get(token_path)
    if cached is not None and cached[0] == st.st_mtime_ns:
        return True  # parsed fine before and unchanged
    if _CORRUPT_TOKEN_CACHE.get(token_path) == st.st_mtime_ns:
        return False
    try:
        _read_token_file(token_path)
    except (OSError, ValueError):
        log.warning("Spotify token file %s is unreadable; treating Spotify as not configured", token_path)
        _CORRUPT_TOKEN_CACHE[token_path] = st.st_mtime_ns
        return False
    return True


def _fold_ascii(text: str) -> str:
    text = text.lower().replace("ł", "l")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _tokenize(text: str) -> list[str]:
    folded = _fold_ascii(text)
    return [tok for tok in re.findall(r"[a-z0-9]+", folded) if tok]


def _nominative_candidates(word: str) -> set[str]:
    """Generate strict Polish nominative base forms that ``word`` can be an inflected form of.

    Never performs arbitrary prefix truncation (so ``muse`` never matches ``museum``, ``rock`` never
    matches ``rocket``, and ``queen`` never matches ``queens``).
    """
    forms = {word}
    if len(word) < 4:
        return forms

    # Multi-character Polish case endings (genitive/dative/instrumental/locative)
    for suf in ("owi", "ami", "ach", "iem", "ego", "emu", "iej", "ie", "em", "om", "ow", "iu", "ia"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            stem = word[: -len(suf)]
            forms.add(stem)
            forms.add(stem + "a")
            forms.add(stem + "o")
            if suf == "ie" and stem.endswith("c") and len(stem) >= 4:
                # e.g. Metallice -> Metallica, Monice -> Monika
                forms.add(stem[:-1] + "ca")
                forms.add(stem[:-1] + "ka")

    # Fleeting 'e' in Polish masculine names (Marka -> Marek, Dudka -> Dudek)
    if word.endswith(("ka", "kowi", "kiem")):
        base = word[:-2] if word.endswith("ka") else word[:-4]
        if len(base) >= 3:
            forms.add(base + "ek")
            forms.add(base + "ka")

    # Single-character Polish case endings
    if word.endswith("a") and len(word) >= 4:
        # Masculine/animate genitive/accusative: Dawida -> Dawid, Korteza -> Kortez, Chopina -> Chopin
        stem = word[:-1]
        forms.add(stem)
        forms.add(stem + "o")
    elif word.endswith("e") and len(word) >= 4:
        # Feminine accusative (-ę folded to -e): Nirvanę -> Nirvana, Metallicę -> Metallica
        stem = word[:-1]
        forms.add(stem + "a")
        if stem.endswith("c") and len(stem) >= 4:
            forms.add(stem[:-1] + "ca")
            forms.add(stem[:-1] + "ka")
    elif word.endswith(("i", "y")) and len(word) >= 5:
        # Feminine genitive: Metalliki -> Metallica, Sanah/Kory -> Kora
        stem = word[:-1]
        forms.add(stem + "a")
        if stem.endswith("k") and len(stem) >= 4:
            forms.add(stem[:-1] + "ca")
    elif word.endswith("u") and len(word) >= 5:
        # Inanimate masculine genitive: Dżemu -> Dżem, Kultu -> Kult
        forms.add(word[:-1])

    return forms


def _tokens_match(q_tok: str, c_tok: str, *, single_token_query: bool = False) -> bool:
    if q_tok == c_tok:
        return True
    if len(q_tok) < 3 or len(c_tok) < 3:
        return False
    # Check whether the query token is a Polish case inflection of the candidate token
    if c_tok in _nominative_candidates(q_tok):
        return True
    if not single_token_query and q_tok in _nominative_candidates(c_tok):
        return True
    return False


def is_generic_or_mood_query(query: str, utterance: str = "") -> bool:
    """Return True when a play request asks for a mood, genre, activity, or indefinite 'coś' rather than a concrete title/artist."""
    combined = f"{query} {utterance}".strip()
    if not combined:
        return False
    if _GENERIC_INDEFINITE_RE.search(combined):
        return True
    if _GENERIC_PURPOSE_RE.search(combined):
        return True
    cleaned = clean_search_query(query) or clean_search_query(utterance)
    tokens = [t for t in _tokenize(cleaned) if t not in _QUERY_FILLER_WORDS]
    if not tokens:
        return False
    if all(t in _MOOD_OR_GENRE_WORDS for t in tokens):
        return True
    return False


def clean_search_query(query: str) -> str:
    """Strip command filler words ('puść ... na Spotify') while preserving artist/track names."""
    raw = query.strip()
    raw = re.sub(
        r"^\s*(?:hej\s+gaja\s*[,:!]?\s*|gaja\s*[,:!]?\s*)?"
        r"(?:pu[sś][cć]|w[lł][aą]cz|zagraj|odtw[oó]rz|graj|play)\s+(?:mi\s+)?",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(r"\s+(?:na|w|ze|z|on)\s+spotify\s*[?!.]*$", "", raw, flags=re.IGNORECASE)
    raw = re.sub(
        r"^\s*(?:piosenk[eę]|utw[oó]r|wykonawc[eę]|artyst[eę]|zesp[oó][lł]|playlist[eę]|muzyk[eę])\s+",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = raw.strip(" ?!.,\"'«»„”")
    if _fold_ascii(raw) in _GENERIC_RESUME_QUERIES:
        return ""
    return raw


def extract_play_query_from_utterance(utterance: str) -> str:
    """Extract a search query from the user's raw utterance when the tool selector omitted ``query``."""
    m = re.search(
        r"\b(?:pu[sś][cć]|w[lł][aą]cz|zagraj|odtw[oó]rz|play)\s+(?P<rest>.+?)$",
        utterance.strip(),
        flags=re.IGNORECASE,
    )
    if not m:
        return ""
    cleaned = clean_search_query(m.group("rest"))
    tokens = [t for t in _tokenize(cleaned) if t not in _QUERY_FILLER_WORDS]
    return cleaned if tokens else ""


def query_matches_hit(query: str, *, name: str, artist: str = "") -> bool:
    """Verify that a search hit's title or artist matches the user's query words."""
    q_tokens = [t for t in _tokenize(query) if t not in _QUERY_FILLER_WORDS]
    if not q_tokens:
        q_tokens = _tokenize(query)
    if not q_tokens:
        return False

    cand_tokens = _tokenize(f"{name} {artist}")
    if not cand_tokens:
        return False

    single = len(q_tokens) == 1
    matched = sum(1 for qt in q_tokens if any(_tokens_match(qt, ct, single_token_query=single) for ct in cand_tokens))
    if len(q_tokens) <= 2:
        return matched == len(q_tokens)
    required = max(2, math.ceil(len(q_tokens) * 0.6))
    return matched >= required


def _is_exact_artist_match(q_tokens: list[str], artist_tokens: list[str]) -> bool:
    if not q_tokens or len(q_tokens) != len(artist_tokens):
        return False
    single = len(q_tokens) == 1
    return all(
        any(_tokens_match(qt, at, single_token_query=single) for at in artist_tokens)
        for qt in q_tokens
    ) and all(
        any(_tokens_match(qt, at, single_token_query=single) for qt in q_tokens)
        for at in artist_tokens
    )


def _pick_best_search_hit(query: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the best matching artist/track/playlist from a ``limit=1`` Spotify search response."""
    q_tokens = [t for t in _tokenize(query) if t not in _QUERY_FILLER_WORDS] or _tokenize(query)
    single = len(q_tokens) == 1
    wants_playlist = bool(re.search(r"\bplaylist", _fold_ascii(query)))

    candidates: list[tuple[tuple[float, int, int], dict[str, Any]]] = []

    tracks = [x for x in (payload.get("tracks") or {}).get("items") or [] if isinstance(x, dict)]
    for item in tracks[:1]:
        name = str(item.get("name") or "").strip()
        artists = ", ".join(str(a.get("name") or "").strip() for a in (item.get("artists") or []) if isinstance(a, dict) and a.get("name"))
        uri = str(item.get("uri") or "").strip()
        if not name or not uri or not query_matches_hit(query, name=name, artist=artists):
            continue
        name_tokens = _tokenize(name)
        title_hits = sum(1 for qt in q_tokens if any(_tokens_match(qt, nt, single_token_query=single) for nt in name_tokens))
        all_tokens = _tokenize(f"{name} {artists}")
        ratio = sum(1 for qt in q_tokens if any(_tokens_match(qt, ct, single_token_query=single) for ct in all_tokens)) / max(1, len(q_tokens))
        kind_prio = 3 if title_hits > 0 and not wants_playlist else 1
        candidates.append(((ratio, kind_prio, title_hits), {"kind": "track", "name": name, "artist": artists, "uri": uri}))

    artists_items = [x for x in (payload.get("artists") or {}).get("items") or [] if isinstance(x, dict)]
    for item in artists_items[:1]:
        name = str(item.get("name") or "").strip()
        uri = str(item.get("uri") or "").strip()
        if not name or not uri or not query_matches_hit(query, name=name):
            continue
        name_tokens = _tokenize(name)
        ratio = sum(1 for qt in q_tokens if any(_tokens_match(qt, nt, single_token_query=single) for nt in name_tokens)) / max(1, len(q_tokens))
        exact_artist = _is_exact_artist_match(q_tokens, name_tokens)
        if wants_playlist:
            kind_prio = 1
        elif exact_artist:
            # Prefer an exact artist match ("Muse") over a track that happens to be titled "Muse" by someone else.
            kind_prio = 4
        else:
            kind_prio = 2
        candidates.append(((ratio, kind_prio, len(name_tokens)), {"kind": "artist", "name": name, "artist": "", "uri": uri}))

    playlists = [x for x in (payload.get("playlists") or {}).get("items") or [] if isinstance(x, dict)]
    for item in playlists[:1]:
        name = str(item.get("name") or "").strip()
        owner = str(((item.get("owner") or {}).get("display_name")) or "").strip()
        uri = str(item.get("uri") or "").strip()
        if not name or not uri or not query_matches_hit(query, name=name, artist=owner):
            continue
        all_tokens = _tokenize(f"{name} {owner}")
        ratio = sum(1 for qt in q_tokens if any(_tokens_match(qt, ct, single_token_query=single) for ct in all_tokens)) / max(1, len(q_tokens))
        kind_prio = 3 if wants_playlist else 0
        candidates.append(((ratio, kind_prio, 0), {"kind": "playlist", "name": name, "artist": owner, "uri": uri}))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


def write_token_atomic(path: Path, token_data: dict[str, Any]) -> None:
    """Atomically write Spotify OAuth token JSON with mode 0600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(token_data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_token_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("spotify token file must contain a JSON object")
    # L2 round 5: a refresh_token is what makes a login usable — the client refreshes it whenever
    # the access token lapses. A file with only an access_token dies at the first expiry (or the
    # first forced refresh), so it never counts as ready: not configured, silent handoff.
    if not str(raw.get("refresh_token") or "").strip():
        raise ValueError("spotify token file has no refresh_token")
    return raw


def _expired_login_result(lang: str) -> ToolResult:
    speak = (
        "Sesja logowania Spotify wygasła, przekazuję to Hermesowi."
        if lang == "pl"
        else "Spotify login expired, handing off to Hermes."
    )
    return ToolResult(
        data={"error": "login_expired"},
        speak=speak,
        error="login_expired",
        handoff=True,
    )


async def _locked_ensure_access_token(
    token_path: Path,
    configured_client_id: str,
    *,
    timeout_s: float,
    transport: httpx.AsyncBaseTransport | None,
    force_refresh: bool,
    lang: str,
) -> tuple[str | None, ToolResult | None]:
    async with _REFRESH_LOCK:
        token_data: dict[str, Any] | None = None
        current_mtime_ns = 0

        if token_path in _PENDING_PERSIST:
            pending = _PENDING_PERSIST[token_path]
            try:
                await asyncio.to_thread(write_token_atomic, token_path, pending)
                st_after = await asyncio.to_thread(_stat_or_none, token_path)
                mtime_ns = st_after.st_mtime_ns if st_after is not None else 0
                _TOKEN_CACHE[token_path] = (mtime_ns, pending)
                _PENDING_PERSIST.pop(token_path, None)
                current_mtime_ns = mtime_ns
            except OSError as exc:
                log.warning(
                    "Retry of Spotify token persistence failed for %s (%s); keeping in memory",
                    token_path,
                    type(exc).__name__,
                )
            token_data = pending
        else:
            st = await asyncio.to_thread(_stat_or_none, token_path)
            if st is None:
                return None, ToolResult(
                    data={"error": "not_configured"},
                    speak="",
                    error="not_configured",
                    handoff=True,
                )
            current_mtime_ns = st.st_mtime_ns

            bad = _INVALID_GRANT_CACHE.get(token_path)
            if bad is not None:
                until, bad_mtime = bad
                if time.monotonic() < until and current_mtime_ns == bad_mtime:
                    return None, _expired_login_result(lang)
                _INVALID_GRANT_CACHE.pop(token_path, None)

            cached = _TOKEN_CACHE.get(token_path)
            if cached is not None and cached[0] == current_mtime_ns:
                token_data = cached[1]
            else:
                try:
                    token_data = await asyncio.to_thread(_read_token_file, token_path)
                except ValueError as exc:
                    # Corrupt JSON / not an object (NEW-8): same as no login at all — silent handoff.
                    log.warning("Spotify token file %s is unreadable (%s); treating as not configured", token_path, exc)
                    return None, ToolResult(
                        data={"error": "not_configured"},
                        speak="",
                        error="not_configured",
                        handoff=True,
                    )
                if token_data is None:
                    return None, ToolResult(
                        data={"error": "not_configured"},
                        speak="",
                        error="not_configured",
                        handoff=True,
                    )
                _TOKEN_CACHE[token_path] = (current_mtime_ns, token_data)

        access_token = str(token_data.get("access_token") or "").strip()
        refresh_token = str(token_data.get("refresh_token") or "").strip()
        expires_at = float(token_data.get("expires_at") or 0.0)
        client_id = (configured_client_id or str(token_data.get("client_id") or "")).strip()

        needs_refresh = force_refresh or not access_token or (time.time() >= expires_at - TOKEN_EXPIRY_SKEW_S)
        if not needs_refresh:
            return access_token, None

        if not refresh_token or not client_id:
            return None, ToolResult(
                data={"error": "invalid_token_config"},
                speak="Brakuje danych logowania Spotify. Uruchom skrypt logowania.",
                error="invalid_token_config",
            )

        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as refresh_client:
            resp = await refresh_client.post(
                SPOTIFY_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if resp.status_code == 400:
            err_kind = ""
            try:
                body = resp.json()
                if isinstance(body, dict):
                    err_kind = str(body.get("error") or "").strip().lower()
            except Exception:
                err_kind = ""
            if err_kind == "invalid_grant":
                log.warning("Spotify refresh returned invalid_grant; caching expiry for %.0fs", INVALID_GRANT_TTL_S)
                _INVALID_GRANT_CACHE[token_path] = (time.monotonic() + INVALID_GRANT_TTL_S, current_mtime_ns)
                _TOKEN_CACHE.pop(token_path, None)
                _PENDING_PERSIST.pop(token_path, None)
                return None, _expired_login_result(lang)

        if resp.status_code != 200:
            log.warning("Spotify token refresh failed with HTTP %d", resp.status_code)
            return None, ToolResult(
                data={"error": "token_refresh_failed", "status": resp.status_code},
                speak="Nie udało się odświeżyć sesji Spotify.",
                error="token_refresh_failed",
            )

        refreshed = resp.json()
        new_access = str(refreshed.get("access_token") or "").strip()
        if not new_access:
            return None, ToolResult(
                data={"error": "token_refresh_empty"},
                speak="Nie udało się odświeżyć sesji Spotify.",
                error="token_refresh_empty",
            )
        new_refresh = str(refreshed.get("refresh_token") or refresh_token).strip()
        new_expires_at = int(time.time()) + int(refreshed.get("expires_in") or 3600)
        new_scope = str(refreshed.get("scope") or token_data.get("scope") or SPOTIFY_SCOPES)

        persisted: dict[str, Any] = {
            "access_token": new_access,
            "refresh_token": new_refresh,
            "expires_at": new_expires_at,
            "scope": new_scope,
        }
        if client_id:
            persisted["client_id"] = client_id

        # Keep rotated token in memory BEFORE attempting disk write so neither an OSError nor outer
        # cancellation can ever lose a rotated refresh_token.
        _PENDING_PERSIST[token_path] = persisted
        try:
            await asyncio.to_thread(write_token_atomic, token_path, persisted)
            st_saved = await asyncio.to_thread(_stat_or_none, token_path)
            saved_mtime = st_saved.st_mtime_ns if st_saved is not None else 0
            _TOKEN_CACHE[token_path] = (saved_mtime, persisted)
            _PENDING_PERSIST.pop(token_path, None)
        except OSError as exc:
            log.warning(
                "Could not persist refreshed Spotify token to %s (%s); keeping in memory for retry",
                token_path,
                type(exc).__name__,
            )
        return new_access, None


async def _ensure_access_token(
    token_path: Path,
    configured_client_id: str,
    *,
    timeout_s: float = 3.0,
    transport: httpx.AsyncBaseTransport | None = None,
    force_refresh: bool = False,
    lang: str = "pl",
) -> tuple[str | None, ToolResult | None]:
    """Return a valid access token, shielding refresh + atomic persist from mid-turn cancellation."""
    task = asyncio.create_task(
        _locked_ensure_access_token(
            token_path,
            configured_client_id,
            timeout_s=timeout_s,
            transport=transport,
            force_refresh=force_refresh,
            lang=lang,
        )
    )
    return await asyncio.shield(task)


@dataclass(frozen=True)
class _HermesCtx:
    """Hermes token source for one turn: the person's Hermes profile (its HERMES_HOME)."""

    profile: str
    home: Path
    cfg: Any  # SpotifyConfig


def _hermes_unavailable_result() -> ToolResult:
    # Quiet handoff: broker failure / no login for this profile -> Hermes takes the turn.
    return ToolResult(
        data={"error": "hermes_spotify_unavailable"},
        speak="",
        error="hermes_spotify_unavailable",
        handoff=True,
    )


async def _hermes_request_token(
    ctx: _HermesCtx, *, force_refresh: bool
) -> tuple[HermesSpotifyToken | None, ToolResult | None]:
    token = await hermes_access_token(ctx.cfg, ctx.profile, force_refresh=force_refresh)
    if token is None:
        return None, _hermes_unavailable_result()
    return token, None


async def _spotify_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    token_path: Path,
    client_id: str,
    timeout_s: float,
    transport: httpx.AsyncBaseTransport | None,
    lang: str,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    hermes: _HermesCtx | None = None,
) -> tuple[httpx.Response | None, ToolResult | None]:
    if hermes is not None:
        token_obj, err = await _hermes_request_token(hermes, force_refresh=False)
        if err is not None or token_obj is None:
            return None, err
        bearer = token_obj.access_token
        api_base = token_obj.base_url or SPOTIFY_API_BASE
    else:
        token, err = await _ensure_access_token(
            token_path, client_id, timeout_s=timeout_s, transport=transport, lang=lang
        )
        if err is not None or token is None:
            return None, err
        bearer = token
        api_base = SPOTIFY_API_BASE

    url = f"{api_base}{path}"
    resp = await client.request(
        method,
        url,
        params=params,
        json=json_body,
        headers={"Authorization": f"Bearer {bearer}"},
    )
    if resp.status_code == 401:
        if hermes is not None:
            token_obj, err = await _hermes_request_token(hermes, force_refresh=True)
            if err is not None or token_obj is None:
                return None, err
            bearer = token_obj.access_token
        else:
            token, err = await _ensure_access_token(
                token_path,
                client_id,
                timeout_s=timeout_s,
                transport=transport,
                force_refresh=True,
                lang=lang,
            )
            if err is not None or token is None:
                return None, err
            bearer = token
        resp = await client.request(
            method,
            url,
            params=params,
            json=json_body,
            headers={"Authorization": f"Bearer {bearer}"},
        )
    return resp, None


def _select_device(devices: list[dict[str, Any]], preferred_name: str = "") -> dict[str, Any] | None:
    eligible = [
        d for d in devices if isinstance(d, dict) and not d.get("is_restricted") and str(d.get("id") or "").strip()
    ]
    if not eligible:
        return None
    active = next((d for d in eligible if d.get("is_active")), None)
    if active is not None:
        return active
    pref = _fold_ascii(preferred_name.strip())
    if pref:
        exact = next((d for d in eligible if _fold_ascii(str(d.get("name") or "").strip()) == pref), None)
        if exact is not None:
            return exact
        partial = next((d for d in eligible if pref in _fold_ascii(str(d.get("name") or ""))), None)
        if partial is not None:
            return partial
    return eligible[0]


async def _resolve_target_device(
    client: httpx.AsyncClient,
    *,
    token_path: Path,
    client_id: str,
    preferred_name: str,
    timeout_s: float,
    transport: httpx.AsyncBaseTransport | None,
    lang: str,
    use_cache: bool = True,
    hermes: _HermesCtx | None = None,
) -> tuple[dict[str, Any] | None, ToolResult | None]:
    cache_key = (token_path, _fold_ascii(preferred_name.strip()))
    if use_cache:
        cached = _DEVICE_CACHE.get(cache_key)
        if cached is not None:
            dev, ts = cached
            if (time.monotonic() - ts) <= DEVICE_CACHE_TTL_S:
                return dev, None
            _DEVICE_CACHE.pop(cache_key, None)

    resp, err = await _spotify_request(
        client,
        "GET",
        "/me/player/devices",
        token_path=token_path,
        client_id=client_id,
        timeout_s=timeout_s,
        transport=transport,
        lang=lang,
        hermes=hermes,
    )
    if err is not None or resp is None:
        return None, err
    if resp.status_code != 200:
        speak = "Nie udało się sprawdzić urządzeń Spotify." if lang == "pl" else "Could not check Spotify devices."
        return None, ToolResult(
            data={"error": "devices_failed", "status": resp.status_code},
            speak=speak,
            error="devices_failed",
        )
    devices = [d for d in (resp.json().get("devices") or []) if isinstance(d, dict)]
    chosen = _select_device(devices, preferred_name)
    if chosen is None:
        _DEVICE_CACHE.pop(cache_key, None)
        return None, ToolResult(
            data={"error": "no_active_device"},
            speak="",
            error="no_active_device",
            handoff=True,
        )
    _DEVICE_CACHE[cache_key] = (chosen, time.monotonic())
    return chosen, None


def _invalidate_device_cache(token_path: Path, preferred_name: str) -> None:
    _DEVICE_CACHE.pop((token_path, _fold_ascii(preferred_name.strip())), None)


def _finalize(result: ToolResult, lang: str) -> ToolResult:
    if result.error is not None:
        set_spotify_playing(False)
    if result.speak:
        result.speak = normalize_for_tts(clean_speakable(result.speak), lang)
    return result


async def spotify(
    action: str = "now_playing",
    query: str | None = None,
    level: int | float | None = None,
    lang: str = "pl",
    *,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ToolResult:
    """Control household Spotify playback via Spotify Web API."""
    cfg = settings if settings is not None else Settings()
    token_path = resolve(cfg.spotify.token_path)
    client_id = cfg.spotify.client_id.strip()
    preferred_device = cfg.spotify.device_name.strip()
    timeout_s = cfg.spotify.timeout_s
    hermes_ctx: _HermesCtx | None = None
    if cfg.spotify.token_source == "hermes":
        # Borrow an access token from the turn's person's Hermes profile (HERMES_HOME);
        # token_path then just keys the per-account device cache.
        profile = profile_for_current_turn()
        hermes_ctx = _HermesCtx(
            profile=profile, home=hermes_home_for(cfg.spotify, profile), cfg=cfg.spotify
        )
        token_path = hermes_ctx.home
    act = (action or "now_playing").strip().lower()
    if act == "resume":
        act = "play"

    ctx = current_session_context.get()
    utterance = str(ctx.get("utterance") or "") if ctx else ""

    try:
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
            if act == "now_playing":
                resp, err = await _spotify_request(
                    client,
                    "GET",
                    "/me/player/currently-playing",
                    token_path=token_path,
                    client_id=client_id,
                    timeout_s=timeout_s,
                    transport=transport,
                    lang=lang,
                    params={"market": cfg.spotify.market},
                    hermes=hermes_ctx,
                )
                if err is not None:
                    return _finalize(err, lang)
                assert resp is not None
                if resp.status_code == 204 or not resp.content:
                    set_spotify_playing(False)
                    speak = "Teraz nic nie gra na Spotify." if lang == "pl" else "Nothing is playing on Spotify right now."
                    return _finalize(ToolResult(data={"is_playing": False}, speak=speak), lang)
                if resp.status_code != 200:
                    speak = "Nie udało się sprawdzić, co teraz gra." if lang == "pl" else "Could not check what is playing."
                    return _finalize(
                        ToolResult(data={"error": "spotify_http", "status": resp.status_code}, speak=speak, error="spotify_http"),
                        lang,
                    )
                body = resp.json()
                item = body.get("item") if isinstance(body, dict) else None
                is_playing = bool(body.get("is_playing")) if isinstance(body, dict) else False
                set_spotify_playing(is_playing)
                if not isinstance(item, dict) or not item.get("name"):
                    speak = "Teraz nic nie gra na Spotify." if lang == "pl" else "Nothing is playing on Spotify right now."
                    return _finalize(ToolResult(data={"is_playing": False}, speak=speak), lang)
                title = str(item.get("name")).strip()
                artists = ", ".join(
                    str(a.get("name")).strip()
                    for a in (item.get("artists") or [])
                    if isinstance(a, dict) and a.get("name")
                )
                if is_playing:
                    if lang == "en":
                        speak = f"Now playing {title} by {artists}." if artists else f"Now playing {title}."
                    else:
                        speak = f"Teraz gra {title}, {artists}." if artists else f"Teraz gra {title}."
                else:
                    if lang == "en":
                        speak = f"Playback is paused on {title} by {artists}." if artists else f"Playback is paused on {title}."
                    else:
                        speak = f"Odtwarzanie jest wstrzymane: {title}, {artists}." if artists else f"Odtwarzanie jest wstrzymane: {title}."
                return _finalize(
                    ToolResult(
                        data={"is_playing": is_playing, "track": title, "artist": artists},
                        speak=speak,
                    ),
                    lang,
                )

            if act == "play":
                raw_q = (query or "").strip()
                if is_generic_or_mood_query(raw_q, utterance):
                    return _finalize(
                        ToolResult(
                            data={"error": "generic_query", "query": raw_q or utterance},
                            speak="",
                            error="generic_query",
                            handoff=True,
                        ),
                        lang,
                    )

                cleaned_query = clean_search_query(raw_q)
                if not cleaned_query and utterance:
                    cleaned_query = extract_play_query_from_utterance(utterance)

                if cleaned_query:
                    if is_generic_or_mood_query(cleaned_query):
                        return _finalize(
                            ToolResult(
                                data={"error": "generic_query", "query": cleaned_query},
                                speak="",
                                error="generic_query",
                                handoff=True,
                            ),
                            lang,
                        )
                    s_resp, err = await _spotify_request(
                        client,
                        "GET",
                        "/search",
                        token_path=token_path,
                        client_id=client_id,
                        timeout_s=timeout_s,
                        transport=transport,
                        lang=lang,
                        params={
                            "q": cleaned_query,
                            "type": "track,artist,playlist",
                            "limit": 1,
                            "market": cfg.spotify.market,
                        },
                        hermes=hermes_ctx,
                    )
                    if err is not None:
                        return _finalize(err, lang)
                    assert s_resp is not None
                    if s_resp.status_code != 200:
                        speak = "Nie udało się wyszukać muzyki na Spotify." if lang == "pl" else "Spotify search failed."
                        return _finalize(
                            ToolResult(data={"error": "search_failed", "status": s_resp.status_code}, speak=speak, error="search_failed"),
                            lang,
                        )
                    hit = _pick_best_search_hit(cleaned_query, s_resp.json())
                    if hit is None:
                        return _finalize(
                            ToolResult(
                                data={"error": "search_mismatch", "query": cleaned_query},
                                speak="",
                                error="search_mismatch",
                                handoff=True,
                            ),
                            lang,
                        )

                    target_dev, err = await _resolve_target_device(
                        client,
                        token_path=token_path,
                        client_id=client_id,
                        preferred_name=preferred_device,
                        timeout_s=timeout_s,
                        transport=transport,
                        lang=lang,
                        hermes=hermes_ctx,
                    )
                    if err is not None or target_dev is None:
                        return _finalize(err, lang)  # type: ignore[arg-type]

                    device_id = str(target_dev.get("id") or "")
                    play_body: dict[str, Any] = (
                        {"uris": [hit["uri"]]} if hit["kind"] == "track" else {"context_uri": hit["uri"]}
                    )
                    p_resp, err = await _spotify_request(
                        client,
                        "PUT",
                        "/me/player/play",
                        token_path=token_path,
                        client_id=client_id,
                        timeout_s=timeout_s,
                        transport=transport,
                        lang=lang,
                        params={"device_id": device_id} if device_id else None,
                        json_body=play_body,
                        hermes=hermes_ctx,
                    )
                    if p_resp is not None and p_resp.status_code == 404:
                        _invalidate_device_cache(token_path, preferred_device)
                        target_dev, err = await _resolve_target_device(
                            client,
                            token_path=token_path,
                            client_id=client_id,
                            preferred_name=preferred_device,
                            timeout_s=timeout_s,
                            transport=transport,
                            lang=lang,
                            use_cache=False,
                            hermes=hermes_ctx,
                        )
                        if err is not None or target_dev is None:
                            return _finalize(err, lang)  # type: ignore[arg-type]
                        device_id = str(target_dev.get("id") or "")
                        p_resp, err = await _spotify_request(
                            client,
                            "PUT",
                            "/me/player/play",
                            token_path=token_path,
                            client_id=client_id,
                            timeout_s=timeout_s,
                            transport=transport,
                            lang=lang,
                            params={"device_id": device_id} if device_id else None,
                            json_body=play_body,
                            hermes=hermes_ctx,
                        )
                    if err is not None or p_resp is None:
                        return _finalize(err, lang)  # type: ignore[arg-type]
                    if p_resp.status_code not in (200, 202, 204):
                        _invalidate_device_cache(token_path, preferred_device)
                        speak = "Nie udało się włączyć odtwarzania na Spotify." if lang == "pl" else "Could not start Spotify playback."
                        return _finalize(
                            ToolResult(data={"error": "play_failed", "status": p_resp.status_code}, speak=speak, error="play_failed"),
                            lang,
                        )
                    set_spotify_playing(True)
                    if lang == "en":
                        speak = (
                            f"Playing {hit['name']} by {hit['artist']}."
                            if hit["kind"] == "track" and hit["artist"]
                            else f"Playing {hit['name']}."
                        )
                    elif hit["kind"] == "track" and hit["artist"]:
                        speak = f"Włączyłam {hit['name']}, {hit['artist']}."
                    elif hit["kind"] == "playlist":
                        speak = f"Włączyłam playlistę {hit['name']}."
                    else:
                        speak = f"Włączyłam {hit['name']}."
                    return _finalize(
                        ToolResult(
                            data={
                                "action": "play",
                                "query": cleaned_query,
                                "kind": hit["kind"],
                                "name": hit["name"],
                                "artist": hit["artist"],
                                "uri": hit["uri"],
                                "device": target_dev.get("name"),
                            },
                            speak=speak,
                        ),
                        lang,
                    )

                target_dev, err = await _resolve_target_device(
                    client,
                    token_path=token_path,
                    client_id=client_id,
                    preferred_name=preferred_device,
                    timeout_s=timeout_s,
                    transport=transport,
                    lang=lang,
                    hermes=hermes_ctx,
                )
                if err is not None or target_dev is None:
                    return _finalize(err, lang)  # type: ignore[arg-type]
                device_id = str(target_dev.get("id") or "")
                p_resp, err = await _spotify_request(
                    client,
                    "PUT",
                    "/me/player/play",
                    token_path=token_path,
                    client_id=client_id,
                    timeout_s=timeout_s,
                    transport=transport,
                    lang=lang,
                    params={"device_id": device_id} if device_id else None,
                    hermes=hermes_ctx,
                )
                if err is not None or p_resp is None:
                    return _finalize(err, lang)  # type: ignore[arg-type]
                if p_resp.status_code not in (200, 202, 204):
                    _invalidate_device_cache(token_path, preferred_device)
                    speak = "Nie udało się wznowić odtwarzania na Spotify." if lang == "pl" else "Could not resume Spotify playback."
                    return _finalize(
                        ToolResult(data={"error": "resume_failed", "status": p_resp.status_code}, speak=speak, error="resume_failed"),
                        lang,
                    )
                set_spotify_playing(True)
                speak = "Wznowiłam odtwarzanie." if lang == "pl" else "Resumed playback."
                return _finalize(ToolResult(data={"action": "play", "device": target_dev.get("name")}, speak=speak), lang)

            if act in ("pause", "next", "previous", "volume_up", "volume_down", "volume_set"):
                needs_live_volume = act in ("volume_up", "volume_down")
                target_dev, err = await _resolve_target_device(
                    client,
                    token_path=token_path,
                    client_id=client_id,
                    preferred_name=preferred_device,
                    timeout_s=timeout_s,
                    transport=transport,
                    lang=lang,
                    use_cache=not needs_live_volume,
                    hermes=hermes_ctx,
                )
                if err is not None or target_dev is None:
                    return _finalize(err, lang)  # type: ignore[arg-type]
                device_id = str(target_dev.get("id") or "")
                dev_params = {"device_id": device_id} if device_id else {}

                async def _control_call(method: str, endpoint: str, extra_params: dict[str, Any] | None = None):
                    nonlocal target_dev, device_id, dev_params
                    call_params = {**(extra_params or {}), **dev_params} or None
                    r, e = await _spotify_request(
                        client,
                        method,
                        endpoint,
                        token_path=token_path,
                        client_id=client_id,
                        timeout_s=timeout_s,
                        transport=transport,
                        lang=lang,
                        params=call_params,
                        hermes=hermes_ctx,
                    )
                    if r is not None and r.status_code == 404 and not needs_live_volume:
                        _invalidate_device_cache(token_path, preferred_device)
                        refreshed_dev, dev_err = await _resolve_target_device(
                            client,
                            token_path=token_path,
                            client_id=client_id,
                            preferred_name=preferred_device,
                            timeout_s=timeout_s,
                            transport=transport,
                            lang=lang,
                            use_cache=False,
                            hermes=hermes_ctx,
                        )
                        if dev_err is not None or refreshed_dev is None:
                            return None, dev_err
                        target_dev = refreshed_dev
                        device_id = str(target_dev.get("id") or "")
                        dev_params = {"device_id": device_id} if device_id else {}
                        retry_params = {**(extra_params or {}), **dev_params} or None
                        r, e = await _spotify_request(
                            client,
                            method,
                            endpoint,
                            token_path=token_path,
                            client_id=client_id,
                            timeout_s=timeout_s,
                            transport=transport,
                            lang=lang,
                            params=retry_params,
                            hermes=hermes_ctx,
                        )
                    return r, e

                if act == "pause":
                    resp, err = await _control_call("PUT", "/me/player/pause")
                    if err is not None or resp is None:
                        return _finalize(err, lang)  # type: ignore[arg-type]
                    if resp.status_code not in (200, 202, 204):
                        _invalidate_device_cache(token_path, preferred_device)
                        speak = "Nie udało się zatrzymać muzyki." if lang == "pl" else "Could not pause playback."
                        return _finalize(
                            ToolResult(data={"error": "pause_failed", "status": resp.status_code}, speak=speak, error="pause_failed"),
                            lang,
                        )
                    set_spotify_playing(False)
                    speak = "Zatrzymałam odtwarzanie." if lang == "pl" else "Paused playback."
                    return _finalize(ToolResult(data={"action": "pause", "device": target_dev.get("name")}, speak=speak), lang)

                if act in ("next", "previous"):
                    endpoint = "/me/player/next" if act == "next" else "/me/player/previous"
                    resp, err = await _control_call("POST", endpoint)
                    if err is not None or resp is None:
                        return _finalize(err, lang)  # type: ignore[arg-type]
                    if resp.status_code not in (200, 202, 204):
                        _invalidate_device_cache(token_path, preferred_device)
                        speak = "Nie udało się przełączyć utworu." if lang == "pl" else "Could not skip track."
                        return _finalize(
                            ToolResult(data={"error": f"{act}_failed", "status": resp.status_code}, speak=speak, error=f"{act}_failed"),
                            lang,
                        )
                    set_spotify_playing(True)
                    if act == "next":
                        speak = "Następny utwór." if lang == "pl" else "Next track."
                    else:
                        speak = "Poprzedni utwór." if lang == "pl" else "Previous track."
                    return _finalize(ToolResult(data={"action": act, "device": target_dev.get("name")}, speak=speak), lang)

                # Volume actions: volume_up, volume_down, volume_set
                current_vol = target_dev.get("volume_percent")
                base_vol = int(current_vol) if isinstance(current_vol, (int, float)) else 50
                step = max(1, min(100, int(cfg.spotify.volume_step)))
                if act == "volume_set":
                    if level is None:
                        speak = "Podaj poziom głośności od zera do stu." if lang == "pl" else "Say a volume level from 0 to 100."
                        return _finalize(ToolResult(data={"error": "missing_level"}, speak=speak, error="missing_level"), lang)
                    target = max(0, min(100, int(round(float(level)))))
                elif act == "volume_up":
                    delta = int(round(float(level))) if level is not None and 1 <= float(level) <= 100 else step
                    target = min(100, base_vol + delta)
                else:  # volume_down
                    delta = int(round(float(level))) if level is not None and 1 <= float(level) <= 100 else step
                    target = max(0, base_vol - delta)

                resp, err = await _control_call("PUT", "/me/player/volume", {"volume_percent": target})
                if err is not None or resp is None:
                    return _finalize(err, lang)  # type: ignore[arg-type]
                if resp.status_code not in (200, 202, 204):
                    _invalidate_device_cache(token_path, preferred_device)
                    speak = "Nie udało się zmienić głośności na Spotify." if lang == "pl" else "Could not change Spotify volume."
                    return _finalize(
                        ToolResult(data={"error": "volume_failed", "status": resp.status_code}, speak=speak, error="volume_failed"),
                        lang,
                    )
                if target <= 0:
                    speak = "Wyciszyłam muzykę." if lang == "pl" else "Muted music."
                elif lang == "en":
                    speak = f"Music volume {target} percent."
                else:
                    pct_word = polish_declension(target, "procent", "procenty", "procent")
                    speak = f"Głośność muzyki {target} {pct_word}."
                return _finalize(
                    ToolResult(data={"action": act, "level": target, "device": target_dev.get("name")}, speak=speak),
                    lang,
                )

            speak = f"Nieznana akcja Spotify: {act}." if lang == "pl" else f"Unknown Spotify action: {act}."
            return _finalize(ToolResult(data={"error": "unknown_action"}, speak=speak, error="unknown_action"), lang)

    except httpx.TimeoutException:
        log.warning("Spotify request timed out")
        speak = "Spotify nie odpowiedziało na czas." if lang == "pl" else "Spotify timed out."
        return _finalize(ToolResult(data={"error": "timeout"}, speak=speak, error="timeout"), lang)
    except httpx.HTTPError as exc:
        log.warning("Spotify HTTP error: %s", exc)
        speak = "Nie udało się połączyć ze Spotify." if lang == "pl" else "Could not connect to Spotify."
        return _finalize(ToolResult(data={"error": "network_error"}, speak=speak, error="network_error"), lang)
