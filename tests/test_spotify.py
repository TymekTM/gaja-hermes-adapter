"""Unit tests for the local Spotify Web API tool and PKCE login script (httpx.MockTransport only)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import time
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import gaja.tools.spotify_tool as spotify_mod
from gaja.config import Settings
from gaja.pipeline import ACKS
from gaja.server import create_app
from gaja.tools.base import current_session_context
from gaja.tools.registry import ToolRegistry, ToolSpec
from gaja.tools.spotify_tool import (
    PLAYING_STATE_TTL_S,
    SPOTIFY_SCOPES,
    _pick_best_search_hit,
    is_generic_or_mood_query,
    is_spotify_playing,
    is_spotify_ready,
    query_matches_hit,
    reset_spotify_caches,
    set_spotify_playing,
    spotify,
    write_token_atomic,
)
from scripts.spotify_login import make_pkce_pair, parse_redirect_input, run_login
from tests.conftest import FakeHermes, FakeLLM
from tests.test_ws_protocol import is_listening, is_type, read_until


@pytest.fixture(autouse=True)
def _clean_spotify_state() -> Any:
    reset_spotify_caches()
    yield
    reset_spotify_caches()


def _settings(tmp_path: Path, client_id: str = "test-client-id", device_name: str = "") -> Settings:
    return Settings(
        spotify={
            "enabled": True,
            "client_id": client_id,
            "token_path": tmp_path / "spotify_token.json",
            "device_name": device_name,
            "timeout_s": 3.0,
            "volume_step": 15,
            "market": "PL",
        }
    )


def _write_token(
    path: Path,
    *,
    access_token: str = "acc-valid",
    refresh_token: str = "ref-initial",
    expires_at: float | None = None,
    client_id: str | None = "test-client-id",
) -> None:
    payload: dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": int(time.time() + 3600) if expires_at is None else expires_at,
        "scope": SPOTIFY_SCOPES,
    }
    if client_id is not None:
        payload["client_id"] = client_id
    write_token_atomic(path, payload)


def test_spotify_login_pkce_flow_writes_0600_token(tmp_path: Path) -> None:
    verifier, challenge = make_pkce_pair()
    assert 43 <= len(verifier) <= 128
    assert challenge

    captured_form: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://accounts.spotify.com/api/token"
        parsed = dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))
        captured_form.update(parsed)
        return httpx.Response(
            200,
            json={
                "access_token": "acc-from-pkce",
                "refresh_token": "ref-from-pkce",
                "expires_in": 3600,
                "scope": SPOTIFY_SCOPES,
            },
        )

    out = io.StringIO()
    token_file = tmp_path / "data" / "spotify_token.json"

    class DynamicStdin(io.StringIO):
        def readline(self, size: int = -1) -> str:  # noqa: ARG002
            printed = out.getvalue()
            url_line = next(line for line in printed.splitlines() if line.startswith("https://accounts.spotify.com/authorize?"))
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(url_line).query)
            assert qs["scope"] == [SPOTIFY_SCOPES]
            assert qs["code_challenge_method"] == ["S256"]
            assert qs["client_id"] == ["my-gaja-client"]
            assert qs["redirect_uri"] == ["http://127.0.0.1:8888/callback"]
            state = qs["state"][0]
            return f"http://127.0.0.1:8888/callback?code=auth-code-xyz&state={state}\n"

    run_login(
        client_id="my-gaja-client",
        redirect_uri="http://127.0.0.1:8888/callback",
        token_path=token_file,
        stdin=DynamicStdin(),
        stdout=out,
        transport=httpx.MockTransport(handler),
    )

    assert token_file.exists()
    assert (token_file.stat().st_mode & 0o777) == 0o600
    saved = json.loads(token_file.read_text(encoding="utf-8"))
    assert saved["access_token"] == "acc-from-pkce"
    assert saved["refresh_token"] == "ref-from-pkce"
    assert saved["scope"] == SPOTIFY_SCOPES
    assert saved["expires_at"] > time.time() + 3000
    printed_url = next(line for line in out.getvalue().splitlines() if line.startswith("https://accounts.spotify.com/authorize?"))
    sent_challenge = urllib.parse.parse_qs(urllib.parse.urlparse(printed_url).query)["code_challenge"][0]
    expected_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(captured_form["code_verifier"].encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    assert sent_challenge == expected_challenge
    assert captured_form["grant_type"] == "authorization_code"
    assert captured_form["code"] == "auth-code-xyz"


def test_parse_redirect_input_requires_and_verifies_state() -> None:
    with pytest.raises(ValueError, match="missing 'state'"):
        parse_redirect_input("http://127.0.0.1:8888/callback?code=abc", "expected")
    with pytest.raises(ValueError, match="missing 'state'"):
        parse_redirect_input("http://127.0.0.1:8888/callback?code=abc&state=", "expected")
    with pytest.raises(ValueError, match="state mismatch"):
        parse_redirect_input("http://127.0.0.1:8888/callback?code=abc&state=wrong", "expected")
    assert parse_redirect_input("http://127.0.0.1:8888/callback?code=abc&state=expected", "expected") == "abc"


@pytest.mark.asyncio
async def test_token_refresh_and_rotation_persists_atomically_0600_off_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L6: token reads and writes run off the event loop via asyncio.to_thread."""
    cfg = _settings(tmp_path, client_id="")
    token_path = cfg.spotify.token_path
    _write_token(
        token_path,
        access_token="expired-acc",
        refresh_token="old-refresh-token",
        expires_at=time.time() - 10,
        client_id="stored-client-id",
    )
    token_path.chmod(0o644)

    to_thread_funcs: list[str] = []
    real_to_thread = asyncio.to_thread

    async def tracked_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        to_thread_funcs.append(getattr(func, "__name__", str(func)))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(spotify_mod.asyncio, "to_thread", tracked_to_thread)

    calls: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url), request.headers.get("Authorization", "")))
        if str(request.url) == "https://accounts.spotify.com/api/token":
            form = dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))
            assert form == {
                "grant_type": "refresh_token",
                "refresh_token": "old-refresh-token",
                "client_id": "stored-client-id",
            }
            return httpx.Response(
                200,
                json={
                    "access_token": "fresh-access-token",
                    "refresh_token": "rotated-refresh-token",
                    "expires_in": 3600,
                    "scope": SPOTIFY_SCOPES,
                },
            )
        if request.url.path == "/v1/me/player/devices":
            return httpx.Response(200, json={"devices": [{"id": "dev-1", "name": "Salon", "is_active": True, "volume_percent": 40}]})
        if request.url.path == "/v1/me/player/pause":
            return httpx.Response(204)
        raise AssertionError(f"unexpected request {request.url}")

    res = await spotify(action="pause", settings=cfg, transport=httpx.MockTransport(handler))
    assert res.error is None
    assert "Zatrzymałam" in res.speak
    assert calls[0][1] == "https://accounts.spotify.com/api/token"
    assert calls[1][2] == "Bearer fresh-access-token"
    assert "_read_token_file" in to_thread_funcs
    assert "write_token_atomic" in to_thread_funcs

    assert (token_path.stat().st_mode & 0o777) == 0o600
    persisted = json.loads(token_path.read_text(encoding="utf-8"))
    assert persisted["access_token"] == "fresh-access-token"
    assert persisted["refresh_token"] == "rotated-refresh-token"
    assert persisted["expires_at"] > time.time() + 3500
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.asyncio
async def test_shielded_refresh_survives_cancellation_and_failed_write_keeps_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """H6 + M5: cancellation cannot lose a rotated refresh token, and a failed write keeps it in memory & retries."""
    cfg = _settings(tmp_path)
    token_path = cfg.spotify.token_path
    _write_token(
        token_path,
        access_token="expired-acc",
        refresh_token="ref-v1",
        expires_at=time.time() - 10,
    )

    refresh_started = asyncio.Event()
    allow_refresh_finish = asyncio.Event()

    class SlowRefreshTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if str(request.url) == "https://accounts.spotify.com/api/token":
                refresh_started.set()
                await allow_refresh_finish.wait()
                return httpx.Response(
                    200,
                    json={
                        "access_token": "acc-v2",
                        "refresh_token": "rotated-secret-ref-v2",
                        "expires_in": 3600,
                        "scope": SPOTIFY_SCOPES,
                    },
                )
            return httpx.Response(204)

    # 1. H6: Cancel outer spotify() mid-refresh -> shielded refresh still persists rotated-secret-ref-v2!
    outer_task = asyncio.create_task(spotify(action="pause", settings=cfg, transport=SlowRefreshTransport()))
    await refresh_started.wait()
    outer_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer_task
    allow_refresh_finish.set()
    await asyncio.sleep(0.05)

    saved_after_cancel = json.loads(token_path.read_text(encoding="utf-8"))
    assert saved_after_cancel["refresh_token"] == "rotated-secret-ref-v2"

    # 2. M5: Expire token again, make write_token_atomic fail once -> kept in memory, retried next call, no token logged
    reset_spotify_caches()
    _write_token(
        token_path,
        access_token="expired-acc-2",
        refresh_token="ref-v2",
        expires_at=time.time() - 10,
    )
    fail_writes = True
    real_write = spotify_mod.write_token_atomic

    def flaky_write(path: Path, data: dict[str, Any]) -> None:
        if fail_writes:
            raise OSError("disk full")
        real_write(path, data)

    monkeypatch.setattr(spotify_mod, "write_token_atomic", flaky_write)

    def refresh_v3_handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://accounts.spotify.com/api/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "acc-secret-v3",
                    "refresh_token": "ref-secret-v3",
                    "expires_in": 3600,
                    "scope": SPOTIFY_SCOPES,
                },
            )
        if request.url.path == "/v1/me/player/devices":
            return httpx.Response(200, json={"devices": [{"id": "d1", "name": "Salon", "is_active": True}]})
        return httpx.Response(204)

    transport_v3 = httpx.MockTransport(refresh_v3_handler)
    with caplog.at_level(logging.WARNING, logger="gaja.tools.spotify_tool"):
        res1 = await spotify(action="pause", settings=cfg, transport=transport_v3)
    assert res1.error is None
    assert "Could not persist refreshed Spotify token" in caplog.text
    assert "acc-secret-v3" not in caplog.text
    assert "ref-secret-v3" not in caplog.text

    # Disk still has old file while write was failing, now allow disk write -> next call persists ref-secret-v3 without re-refreshing!
    fail_writes = False
    res2 = await spotify(action="next", settings=cfg, transport=transport_v3)
    assert res2.error is None
    on_disk = json.loads(token_path.read_text(encoding="utf-8"))
    assert on_disk["refresh_token"] == "ref-secret-v3"


@pytest.mark.asyncio
async def test_invalid_grant_refresh_hands_off_and_caches_failure(tmp_path: Path) -> None:
    """M4: 400 invalid_grant sets handoff=True with 'login expired' message and caches failure for 10 min."""
    cfg = _settings(tmp_path)
    _write_token(
        cfg.spotify.token_path,
        access_token="expired",
        refresh_token="revoked-refresh",
        expires_at=time.time() - 10,
    )
    assert is_spotify_ready(cfg) is True

    token_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls
        if str(request.url) == "https://accounts.spotify.com/api/token":
            token_calls += 1
            return httpx.Response(400, json={"error": "invalid_grant", "error_description": "Refresh token revoked"})
        raise AssertionError(f"unexpected {request.url}")

    transport = httpx.MockTransport(handler)
    res1 = await spotify(action="play", query="Muse", settings=cfg, transport=transport)
    assert res1.error == "login_expired"
    assert res1.handoff is True
    assert "wygasła" in res1.speak
    assert token_calls == 1
    assert is_spotify_ready(cfg) is False

    # Second call within 10 min does NOT hammer SPOTIFY_TOKEN_URL again
    res2 = await spotify(action="pause", settings=cfg, transport=transport)
    assert res2.error == "login_expired"
    assert res2.handoff is True
    assert token_calls == 1


@pytest.mark.asyncio
async def test_missing_token_file_hands_off_silently_without_extra_sentence(tmp_path: Path, make_services) -> None:
    """M6: missing token file sets speak='' and handoff=True so no extra 'nieskonfigurowane' sentence is spoken."""
    cfg = _settings(tmp_path)
    res = await spotify(action="pause", settings=cfg)
    assert res.error == "not_configured"
    assert res.handoff is True
    assert res.speak == ""

    reg = ToolRegistry()

    async def unconfigured_spotify(**kwargs: Any):
        return await spotify(settings=cfg, **kwargs)

    reg.register(
        ToolSpec(
            name="spotify",
            description_pl="Spotify",
            description_en="Spotify",
            parameters={"type": "object", "properties": {"action": {"type": "string"}}},
            handler=unconfigured_spotify,
        )
    )
    llm = FakeLLM(p_tool=0.95, p_hermes=0.02, tool_call=' spotify(action="pause")')
    hermes = FakeHermes(result="Zatrzymałam na Spotify przez Hermesa.")
    services = make_services(llm=llm, hermes=hermes, router=True)
    services.tools = reg
    services.router = RouterForTest(llm, reg.schemas())
    app = create_app(services=services)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "lang": "pl"}))
        read_until(ws, is_listening)
        ws.send_text(json.dumps({"type": "text", "text": "zatrzymaj muzykę na Spotify"}))
        messages = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "handoff" and m["state"] == "done")
        spoken = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "audio_end" and m["id"] == 2)
        metrics_msgs = read_until(ws, is_type("metrics"))
    assert hermes.tasks == ["zatrzymaj muzykę na Spotify"]
    assert metrics_msgs[-1]["route"] == "hermes"
    utterances = [m["text"] for m in messages + spoken if isinstance(m, dict) and m.get("type") == "utterance"]
    assert not any("skonfigurowane" in u for u in utterances)
    assert utterances[0] in ACKS["pl"]


@pytest.mark.asyncio
async def test_corrupt_token_file_is_not_ready_and_hands_off_silently(tmp_path: Path) -> None:
    """NEW-8: corrupt token JSON used to leave is_spotify_ready True (stat only) and the tool
    raised a bare JSONDecodeError, so the user heard a generic error. A ValueError from the token
    file now counts as not_configured: readiness False and a silent handoff to Hermes."""
    cfg = _settings(tmp_path)
    cfg.spotify.token_path.write_text('{"access_token": "acc", oops', encoding="utf-8")

    assert is_spotify_ready(cfg) is False

    res = await spotify(action="pause", settings=cfg)
    assert res.error == "not_configured"
    assert res.handoff is True
    assert res.speak == ""

    # A fixed file recovers without a restart (the corrupt cache is keyed by mtime).
    _write_token(cfg.spotify.token_path)
    assert is_spotify_ready(cfg) is True


@pytest.mark.asyncio
async def test_token_file_without_tokens_is_not_ready_and_hands_off(tmp_path: Path) -> None:
    """L2: A token file that is valid JSON but has no refresh_token/access_token
    must count as not ready, with a handoff to Hermes."""
    cfg = _settings(tmp_path)
    cfg.spotify.token_path.write_text("{}", encoding="utf-8")

    assert is_spotify_ready(cfg) is False

    res = await spotify(action="pause", settings=cfg)
    assert res.error == "not_configured"
    assert res.handoff is True
    assert res.speak == ""


@pytest.mark.asyncio
async def test_token_file_with_only_an_access_token_is_not_ready_and_hands_off(tmp_path: Path) -> None:
    """L2 r5: the refresh_token is what keeps a login alive (access tokens lapse). A file with
    only an access_token must count as not configured — readiness False plus a silent handoff to
    Hermes — instead of dying on the first refresh."""
    cfg = _settings(tmp_path)
    cfg.spotify.token_path.write_text(
        json.dumps({"access_token": "acc-only", "expires_at": 9999999999}), encoding="utf-8"
    )

    assert is_spotify_ready(cfg) is False

    res = await spotify(action="pause", settings=cfg)
    assert res.error == "not_configured"
    assert res.handoff is True
    assert res.speak == ""

    # A complete token file recovers without a restart.
    _write_token(cfg.spotify.token_path)
    assert is_spotify_ready(cfg) is True


def RouterForTest(llm: FakeLLM, schemas: list[dict[str, Any]]):
    from gaja.llm.router import Router

    return Router(llm, "sys", threshold=0.9, history_turns=2, timeout_s=2.0, tools=schemas, tool_threshold=0.5)


@pytest.mark.asyncio
async def test_cold_device_wakes_configured_or_first_unrestricted_and_empty_devices_hand_off(tmp_path: Path) -> None:
    """H1: cold 'puść Muse' falls back to configured device_name or first non-restricted device; empty devices -> handoff=True."""
    # 1. Configured device_name wins among idle non-restricted devices and passes device_id to PUT /me/player/play
    cfg_named = _settings(tmp_path, device_name="Onkyo")
    _write_token(cfg_named.spotify.token_path)
    played_params: dict[str, str] = {}

    def idle_devices_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search":
            return httpx.Response(
                200,
                json={
                    "tracks": {"items": []},
                    "artists": {"items": [{"name": "Muse", "uri": "spotify:artist:muse1"}]},
                    "playlists": {"items": []},
                },
            )
        if request.url.path == "/v1/me/player/devices":
            return httpx.Response(
                200,
                json={
                    "devices": [
                        {"id": "dev-restricted", "name": "Restricted", "is_active": False, "is_restricted": True},
                        {"id": "dev-phone", "name": "Telefon", "is_active": False, "is_restricted": False, "volume_percent": 30},
                        {"id": "dev-onkyo", "name": "Onkyo Salon", "is_active": False, "is_restricted": False, "volume_percent": 40},
                    ]
                },
            )
        if request.url.path == "/v1/me/player/play":
            played_params.update(dict(request.url.params))
            return httpx.Response(204)
        raise AssertionError(f"unexpected {request.url}")

    res_named = await spotify(
        action="play", query="Muse", settings=cfg_named, transport=httpx.MockTransport(idle_devices_handler)
    )
    assert res_named.error is None
    assert played_params == {"device_id": "dev-onkyo"}
    assert "Włączyłam Muse" in res_named.speak

    # 2. Without configured device_name, falls back to the first non-restricted device ("dev-phone")
    reset_spotify_caches()
    cfg_default = _settings(tmp_path)
    played_params.clear()
    res_first = await spotify(
        action="play", query="Muse", settings=cfg_default, transport=httpx.MockTransport(idle_devices_handler)
    )
    assert res_first.error is None
    assert played_params == {"device_id": "dev-phone"}

    # 3. With no non-restricted device at all, sets handoff=True instead of speaking a local dead-end
    reset_spotify_caches()

    def empty_devices_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/me/player/devices":
            return httpx.Response(
                200,
                json={"devices": [{"id": "r1", "name": "Locked", "is_active": False, "is_restricted": True}]},
            )
        if request.url.path == "/v1/search":
            return httpx.Response(
                200,
                json={"tracks": {"items": []}, "artists": {"items": [{"name": "Muse", "uri": "spotify:artist:m"}]}, "playlists": {"items": []}},
            )
        raise AssertionError(f"unexpected {request.url}")

    res_none = await spotify(
        action="play", query="Muse", settings=cfg_default, transport=httpx.MockTransport(empty_devices_handler)
    )
    assert res_none.error == "no_active_device"
    assert res_none.handoff is True
    assert res_none.speak == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "utterance",
    [
        "puść coś spokojnego",
        "puść coś",
        "puść mi jakąś muzykę",
        "puść muzykę do pracy",
        "puść jakiś jazz",
    ],
)
async def test_mood_and_generic_queries_hand_off_before_searching(tmp_path: Path, utterance: str) -> None:
    """H2: mood/genre/'coś' queries hand off to Hermes before calling /v1/search."""
    cfg = _settings(tmp_path)
    _write_token(cfg.spotify.token_path)
    assert is_generic_or_mood_query("", utterance) is True

    def fail_if_called(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise AssertionError("Spotify HTTP API should not be called for generic/mood queries")

    tok = current_session_context.set({"utterance": utterance})
    try:
        res = await spotify(action="play", query=None, settings=cfg, transport=httpx.MockTransport(fail_if_called))
    finally:
        current_session_context.reset(tok)
    assert res.error == "generic_query"
    assert res.handoff is True
    assert res.speak == ""


@pytest.mark.asyncio
async def test_search_mismatch_hands_off_silently(tmp_path: Path) -> None:
    """H2 + M6: when /search finds no matching hit, set handoff=True instead of dead-ending."""
    cfg = _settings(tmp_path)
    _write_token(cfg.spotify.token_path)
    play_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal play_called
        if request.url.path == "/v1/search":
            assert request.url.params["limit"] == "1"
            assert request.url.params["type"] == "track,artist,playlist"
            return httpx.Response(
                200,
                json={
                    "tracks": {
                        "items": [
                            {
                                "name": "Przez Twe Oczy Zielone",
                                "uri": "spotify:track:wrong1",
                                "artists": [{"name": "Akcent"}],
                            }
                        ]
                    },
                    "artists": {"items": [{"name": "Zenek Martyniuk", "uri": "spotify:artist:wrong2"}]},
                    "playlists": {"items": []},
                },
            )
        if request.url.path == "/v1/me/player/play":
            play_called = True
            return httpx.Response(204)
        return httpx.Response(200, json={"devices": [{"id": "d1", "name": "Salon", "is_active": True}]})

    res = await spotify(action="play", query="Radiohead Karma Police", settings=cfg, transport=httpx.MockTransport(handler))
    assert res.error == "search_mismatch"
    assert res.handoff is True
    assert res.speak == ""
    assert not play_called


def test_query_matches_hit_and_ranking_reject_prefix_collisions_and_prefer_exact_artist() -> None:
    """M3: reject 'Muse'/'Museum', 'rock'/'Rocket Man', 'Queen'/'Queens of the Stone Age'; prefer artist Muse over track 'Muse' by 'Random'."""
    assert query_matches_hit("Dawida Podsiadło", name="Dawid Podsiadło")
    assert query_matches_hit("puść Korteza na Spotify", name="Pierwszy", artist="Kortez")
    assert query_matches_hit("Metallicę", name="Metallica")
    assert query_matches_hit("Queen Bohemian", name="Bohemian Rhapsody", artist="Queen")

    assert not query_matches_hit("Queen Yesterday", name="Bohemian Rhapsody", artist="Queen")
    assert not query_matches_hit("Muse", name="Museum", artist="Some Band")
    assert not query_matches_hit("rock", name="Rocket Man", artist="Elton John")
    assert not query_matches_hit("Queen", name="No One Knows", artist="Queens of the Stone Age")

    # Ranking: exact artist "Muse" beats a track titled "Muse" by "Random"
    payload = {
        "tracks": {"items": [{"name": "Muse", "uri": "spotify:track:random_muse", "artists": [{"name": "Random"}]}]},
        "artists": {"items": [{"name": "Muse", "uri": "spotify:artist:real_muse"}]},
        "playlists": {"items": []},
    }
    best = _pick_best_search_hit("Muse", payload)
    assert best is not None
    assert best["kind"] == "artist"
    assert best["uri"] == "spotify:artist:real_muse"


def test_playing_state_scoped_per_session_short_ttl_and_cleared_on_error(tmp_path: Path) -> None:
    """M2: 10-min TTL, scoped per session/device, cleared on pause and on any Spotify error."""
    assert PLAYING_STATE_TTL_S == pytest.approx(600.0)

    tok_a = current_session_context.set({"session_id": "sess-a"})
    try:
        set_spotify_playing(True)
        assert is_spotify_playing() is True
    finally:
        current_session_context.reset(tok_a)

    # Another session does NOT see sess-a's playing state
    tok_b = current_session_context.set({"session_id": "sess-b"})
    try:
        assert is_spotify_playing() is False
    finally:
        current_session_context.reset(tok_b)

    # Error in sess-a clears sess-a's playing state
    cfg = _settings(tmp_path)  # missing token file -> error="not_configured"
    tok_a = current_session_context.set({"session_id": "sess-a"})
    try:
        res = asyncio.run(spotify(action="next", settings=cfg))
        assert res.error == "not_configured"
        assert is_spotify_playing() is False
    finally:
        current_session_context.reset(tok_a)


@pytest.mark.asyncio
async def test_play_query_searches_limit_1_and_plays_matching_hit(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    _write_token(cfg.spotify.token_path)
    played_payload: dict[str, Any] = {}
    played_device: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal played_device
        if request.url.path == "/v1/search":
            assert request.url.params["limit"] == "1"
            return httpx.Response(
                200,
                json={
                    "tracks": {
                        "items": [
                            {
                                "name": "Małomiasteczkowy",
                                "uri": "spotify:track:123",
                                "artists": [{"name": "Dawid Podsiadło"}],
                            }
                        ]
                    },
                    "artists": {"items": [{"name": "Dawid Podsiadło", "uri": "spotify:artist:456"}]},
                    "playlists": {"items": [None]},
                },
            )
        if request.url.path == "/v1/me/player/devices":
            return httpx.Response(200, json={"devices": [{"id": "dev-avr", "name": "Onkyo", "is_active": True, "volume_percent": 45}]})
        if request.url.path == "/v1/me/player/play":
            played_device = request.url.params.get("device_id")
            played_payload.update(json.loads(request.content.decode("utf-8")))
            return httpx.Response(204)
        raise AssertionError(f"unexpected {request.url}")

    set_spotify_playing(False)
    res = await spotify(
        action="play",
        query="Dawida Podsiadło Małomiasteczkowy",
        settings=cfg,
        transport=httpx.MockTransport(handler),
    )
    assert res.error is None
    assert played_device == "dev-avr"
    assert played_payload == {"uris": ["spotify:track:123"]}
    assert "Włączyłam Małomiasteczkowy, Dawid Podsiadło" in res.speak
    assert is_spotify_playing() is True


@pytest.mark.asyncio
async def test_playback_controls_volume_now_playing_and_fresh_device_cache(tmp_path: Path) -> None:
    """Controls pass device_id and skip /me/player/devices when cached device_id is fresh (M6)."""
    cfg = _settings(tmp_path)
    _write_token(cfg.spotify.token_path)
    observed: list[tuple[str, str, dict[str, str]]] = []
    devices_fetches = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal devices_fetches
        observed.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path == "/v1/me/player/devices":
            devices_fetches += 1
            return httpx.Response(
                200,
                json={"devices": [{"id": "dev-1", "name": "Salon", "is_active": True, "volume_percent": 50}]},
            )
        if request.url.path == "/v1/me/player/currently-playing":
            return httpx.Response(
                200,
                json={
                    "is_playing": True,
                    "item": {"name": "Szary Świat", "artists": [{"name": "Sanah"}, {"name": "Kwiat Jabłoni"}]},
                },
            )
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    r_next = await spotify(action="next", settings=cfg, transport=transport)
    assert r_next.error is None and "Następny" in r_next.speak
    assert devices_fetches == 1

    # Fresh cached device_id skips /me/player/devices on subsequent non-relative controls!
    r_prev = await spotify(action="previous", settings=cfg, transport=transport)
    assert r_prev.error is None and "Poprzedni" in r_prev.speak
    assert devices_fetches == 1
    assert ("POST", "/v1/me/player/previous", {"device_id": "dev-1"}) in observed

    r_up = await spotify(action="volume_up", settings=cfg, transport=transport)
    assert r_up.error is None and r_up.data["level"] == 65
    assert "sześćdziesiąt pięć procent" in r_up.speak

    r_down = await spotify(action="volume_down", settings=cfg, transport=transport)
    assert r_down.error is None and r_down.data["level"] == 35

    fetches_before_set = devices_fetches
    r_set = await spotify(action="volume_set", level=80, settings=cfg, transport=transport)
    assert r_set.error is None and r_set.data["level"] == 80
    assert devices_fetches == fetches_before_set  # volume_set uses cached device_id without extra GET
    assert ("PUT", "/v1/me/player/volume", {"volume_percent": "80", "device_id": "dev-1"}) in observed

    r_now = await spotify(action="now_playing", settings=cfg, transport=transport)
    assert r_now.error is None
    assert "Teraz gra Szary Świat, Sanah, Kwiat Jabłoni" in r_now.speak
