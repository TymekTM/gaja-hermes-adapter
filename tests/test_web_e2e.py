"""Browser end-to-end tests of the web UI (Playwright + system Chromium) against clients/mock_server.py.

Run:  TMPDIR=/tmp .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_web_e2e.py
Set GAJA_E2E_BASE=http://127.0.0.1:8790/ to run the protocol-agnostic subset against the real adapter
(tests that need the mock's scripted scenarios are skipped then).
"""

import asyncio
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import websockets
from playwright.sync_api import sync_playwright

from clients.mock_server import MockServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAKE_WAV = os.path.join(ROOT, "bench", "data", "stt", "pl", "wer", "pl_000_fleurs1829.wav")
EXTERNAL = os.environ.get("GAJA_E2E_BASE", "")
needs_mock = pytest.mark.skipif(bool(EXTERNAL), reason="needs the mock server's scripted scenarios")

CHROME_ARGS = [
    "--no-sandbox",
    "--autoplay-policy=no-user-gesture-required",
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
]


def chromium_path():
    for c in ("/usr/lib/chromium/chromium", shutil.which("chromium"), shutil.which("google-chrome"), shutil.which("chromium-browser")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


@pytest.fixture(scope="module")
def base():
    if EXTERNAL:
        yield EXTERNAL if EXTERNAL.endswith("/") else EXTERNAL + "/"
        return
    port = 8796
    loop = asyncio.new_event_loop()
    mock = MockServer(host="127.0.0.1", port=port, web_dir=os.path.join(ROOT, "web"))
    stop = threading.Event()

    def run():
        asyncio.set_event_loop(loop)

        async def main():
            server = await websockets.serve(mock.handle_ws, host="127.0.0.1", port=port, process_request=mock.process_request, max_size=None)
            while not stop.is_set():
                await asyncio.sleep(0.1)
            server.close()
            await server.wait_closed()

        try:
            loop.run_until_complete(main())
        except BaseException:  # keep the reason visible instead of silently refusing connections
            import traceback

            traceback.print_exc()
            raise
        finally:
            loop.close()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    time.sleep(0.6)
    yield f"http://127.0.0.1:{port}/"
    stop.set()
    th.join(timeout=3)


@pytest.fixture(scope="module")
def browser():
    exe = chromium_path()
    assert exe, "Chromium executable not found"
    os.environ["TMPDIR"] = "/tmp"  # keep unix socket paths short
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe, args=CHROME_ARGS + [f"--use-file-for-fake-audio-capture={FAKE_WAV}"])
        yield b
        b.close()


class Page:
    """A fresh browser context + page that records console errors."""

    def __init__(self, browser, base, query="", viewport=(1280, 800), scheme="dark", locale="pl-PL", storage=None, mobile=False):
        self.errors = []
        self.sent = []  # JSON frames the page sent over the websocket
        self.ctx = browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]},
            permissions=["microphone"],
            color_scheme=scheme,
            locale=locale,
            is_mobile=mobile,
            has_touch=mobile,
        )
        if storage is not None:
            self.ctx.add_init_script(f"if (!sessionStorage.getItem('__seeded')) {{ localStorage.setItem('gaja.settings', {json.dumps(json.dumps(storage))}); sessionStorage.setItem('__seeded', '1'); }}")
        self.pg = self.ctx.new_page()
        self.pg.on("websocket", lambda ws: ws.on("framesent", self._on_frame))
        self.pg.on("pageerror", lambda e: self.errors.append(f"[pageerror] {e}"))
        self.pg.on("console", lambda m: self.errors.append(f"[{m.type}] {m.text}") if m.type == "error" else None)
        self.pg.goto(base + query)
        self.pg.wait_for_function("window.__gaja && window.__gaja.S.conn === 'open'", timeout=8000)

    def _on_frame(self, payload):
        if isinstance(payload, str):
            try:
                self.sent.append(json.loads(payload))
            except ValueError:
                pass

    def sent_types(self, kind):
        return [m for m in self.sent if m.get("type") == kind]

    def stats(self):
        return self.pg.evaluate("({ ...window.__gajaStats })")

    def wait(self, expr, timeout=25000):
        self.pg.wait_for_function(expr, timeout=timeout)

    def type(self, text):
        self.pg.fill("#text-input", text)
        self.pg.keyboard.press("Enter")

    def close(self):
        self.ctx.close()


@pytest.fixture()
def make_page(browser, base):
    pages = []

    def make(query="", **kw):
        pg = Page(browser, base, query, **kw)
        pages.append(pg)
        return pg

    yield make
    for pg in pages:
        pg.close()


def no_errors(page):
    # a deliberately dropped socket logs a failed-handshake error: not a UI bug
    errs = [e for e in page.errors if "WebSocket connection" not in e]
    assert not errs, f"browser console had errors: {errs}"


# ------------------------------------------------------------------------------------------ audio


def test_typed_text_plays_without_mic(make_page):
    """The bug that mattered: a typed turn must be audible even though the microphone was never started."""
    page = make_page()
    assert page.pg.evaluate("window.__gaja.S.mic") == "off"
    page.type("Dzień dobry, poproszę o odpowiedź.")
    page.wait("window.__gajaStats.framesReceived >= 1")  # the real server sends one binary frame per sentence
    page.wait("window.__gajaStats.samplesPlayed > 2000")
    page.wait("window.__gajaStats.utterancesPlayed >= 1")
    st = page.stats()
    assert st["ctx"] == "running", st
    assert st["framesSent"] == 0 and page.pg.evaluate("window.__gaja.S.mic") == "off", "no mic frames expected"
    assert "Dzień dobry" in page.pg.inner_text(".msg.you .bubble")
    assert page.pg.locator(".msg.gaja .sent").count() >= 1
    no_errors(page)


def test_typed_text_plays_after_suggestion_tap_on_phone(make_page):
    """Same on a phone-sized touch page: the first gesture is a tap on a suggestion chip."""
    page = make_page(viewport=(390, 844), mobile=True)
    page.pg.tap("#suggest button >> nth=0")
    page.wait("window.__gajaStats.utterancesPlayed >= 1")
    assert page.stats()["ctx"] == "running"
    no_errors(page)


@pytest.mark.parametrize("engine,rate", [("piper", 22050), ("paradee", 24000), ("supertonic", 44100)])
@needs_mock
def test_sample_rate_follows_audio_start(make_page, engine, rate):
    """Playback duration must equal PCM duration whatever rate the engine announces."""
    page = make_page(f"?tts={engine}&pace=realtime")
    page.type("Powiedz coś")
    page.wait("window.__gajaStats.utterancesPlayed >= 2", timeout=30000)
    st = page.stats()
    assert st["lastRate"] == rate, st
    pcm_seconds = st["bytesReceived"] / 2 / rate
    played_seconds = st["samplesPlayed"] / st["ctxRate"]
    assert abs(played_seconds - pcm_seconds) / pcm_seconds < 0.08, (pcm_seconds, played_seconds, st)
    no_errors(page)


@needs_mock
def test_barge_in_drops_audio_and_marks_cut(make_page):
    page = make_page()
    page.type("przerwij mnie")
    page.wait("window.__gajaStats.droppedStale > 0 || window.__gajaStats.flushes > 0")
    page.wait("window.__gaja.S.server === 'listening'")
    page.wait("document.querySelector('.msg.gaja.cut')")
    page.pg.wait_for_timeout(300)
    st = page.stats()
    assert st["flushes"] >= 1, st
    assert page.pg.evaluate("window.__gaja.S.awaiting.size") == 0
    # nothing is queued any more: the playback buffer must be empty and stay empty
    page.pg.wait_for_timeout(500)
    assert page.stats()["buffered"] < 200, page.stats()
    assert page.pg.locator(".msg.gaja .tag").count() == 1
    no_errors(page)


@needs_mock
def test_stale_frames_are_dropped(make_page):
    page = make_page("?demo=stale")
    page.wait("window.__gajaStats.utterancesPlayed >= 1")
    st = page.stats()
    assert st["droppedStale"] > 0, st
    no_errors(page)


@needs_mock
def test_sentences_highlight_with_emotion_while_spoken(make_page):
    page = make_page("?demo=speak&emotion=happy&pace=realtime")
    page.wait("document.querySelector('.sent[data-state=playing]')")
    assert page.pg.get_attribute(".sent[data-state=playing]", "data-emo") == "happy"
    # the status line cross-fades (~130 ms) and the box is heavily loaded in CI: wait, don't sample
    page.wait("document.body.dataset.state === 'speaking'")
    page.wait("document.querySelector('#status-text').textContent.includes('radośnie')", timeout=15000)
    page.wait("document.querySelectorAll('.sent[data-state=done]').length >= 1")
    page.wait("window.__gajaStats.utterancesPlayed >= 3", timeout=30000)
    page.wait("document.body.dataset.state === 'idle'", timeout=20000)
    assert page.pg.locator(".sent[data-state=queued], .sent[data-state=playing]").count() == 0
    no_errors(page)


# ------------------------------------------------------------------------------------------ microphone


@needs_mock
def test_hands_free_with_fake_microphone(make_page):
    """Tap -> mic on with AEC/NS/AGC requested -> frames flow -> server VAD -> reply is played."""
    # half duplex: the looping fake mic is far louder than the reply, so with barge-in enabled it would
    # (correctly) keep cutting the answer off - that path has its own test below
    page = make_page("?demo=live&pace=realtime", storage={"bargein": False})
    page.pg.click("#btn-mic")
    page.wait("window.__gaja.S.mic === 'live'")
    page.wait("window.__gajaStats.framesSent > 20")
    mic = page.pg.evaluate("window.__gaja.audio.micSettings")
    assert mic and mic.get("echoCancellation") is True, mic
    page.wait("document.querySelector('.msg.you .bubble') && !document.querySelector('.msg.you.ghost')", timeout=20000)
    page.wait("window.__gajaStats.utterancesPlayed >= 1", timeout=20000)
    assert page.pg.locator(".msg.gaja .sent").count() >= 1
    for _ in range(4):  # while Gaja is audible the button is "Stop"; the next tap turns the mic off
        if page.pg.evaluate("window.__gaja.S.mic") == "off":
            break
        page.pg.click("#btn-mic")
        page.pg.wait_for_timeout(600)
    page.wait("window.__gaja.S.mic === 'off'")
    no_errors(page)


@needs_mock
def test_echo_gate_keeps_gaja_from_interrupting_herself(make_page):
    """Half duplex ("Przerywanie głosem" off): while Gaja speaks the mic is sent as silence, so even a
    loud mic (here: a speech file standing in for her own echo) cannot cut her off."""
    page = make_page("?demo=live&pace=realtime", storage={"bargein": False})
    page.pg.click("#btn-mic")
    page.wait("window.__gaja.S.mic === 'live'")
    # let the fake mic's first phrase pass through (it triggers one normal turn), then ask for a long answer
    page.wait("window.__gajaStats.utterancesPlayed >= 1", timeout=25000)
    # The fake mic *loops* its phrase, so now and then it is "the user speaking" while Gaja is still thinking
    # (the gate is not up yet) and legitimately cancels a turn. What must never happen is a cut *while she is
    # audible*, so try a few turns: at least one long answer has to play through under the loud mic.
    for _ in range(4):
        page.wait("window.__gaja.S.server === 'listening' || window.__gaja.S.server === 'idle'", timeout=30000)
        before = page.stats()
        cuts = page.pg.locator(".sent[data-state=cut]").count()
        page.type("opowiedz coś długiego")
        page.wait("window.__gaja.S.playing === true", timeout=20000)
        page.wait(
            "window.__gajaStats.utterancesPlayed >= %d || document.querySelectorAll('.sent[data-state=cut]').length > %d"
            % (before["utterancesPlayed"] + 2, cuts),
            timeout=40000,
        )
        if page.pg.locator(".sent[data-state=cut]").count() == cuts:
            break
    else:
        raise AssertionError("every long answer was cut off although the echo gate was up")
    assert page.stats()["framesGated"] > before["framesGated"] + 5
    no_errors(page)


@needs_mock
def test_user_voice_barges_in_when_enabled(make_page):
    """Same loud mic, barge-in on: real speech over Gaja *does* interrupt (the gate opens)."""
    page = make_page("?demo=live&pace=realtime")
    page.pg.click("#btn-mic")
    page.wait("window.__gaja.S.mic === 'live'")
    # the looping fake mic keeps "speaking": the gate opens, the server hears it and cuts Gaja off
    page.wait("document.querySelector('.msg.you .bubble')", timeout=25000)
    page.type("opowiedz coś długiego")
    page.wait("document.querySelector('.sent[data-state=cut]')", timeout=40000)
    no_errors(page)


def test_mic_denied_and_insecure_guidance(make_page):
    page = make_page("?insecure=1")
    page.wait("!document.getElementById('notice').hidden")
    text = page.pg.inner_text("#notice")
    assert "tailscale serve" in text and "HTTPS" in text
    page.pg.click("#notice .btn.primary")  # "Pisz zamiast mówić"
    assert page.pg.evaluate("document.activeElement.id") == "text-input"

    page2 = make_page("?micdenied=1")
    page2.pg.click("#btn-mic")
    page2.wait("!document.getElementById('notice').hidden")
    assert "mikrofon" in page2.pg.inner_text("#notice").lower()
    no_errors(page2)


# ------------------------------------------------------------------------------------------ UI


@needs_mock
def test_hermes_card_lifecycle(make_page):
    page = make_page("?demo=hermes&hermes_s=4&pace=realtime", viewport=(390, 844), mobile=True)
    page.wait("document.querySelector('.hcard[data-state=working]')", timeout=15000)
    assert "Hermes pracuje" in page.pg.inner_text(".hcard .htitle")
    assert re.fullmatch(r"[0-9a-f]{8}", page.pg.get_attribute(".hcard", "data-id")), "cards are keyed by handoff.id"
    page.pg.click(".hcard-head")
    assert page.pg.get_attribute(".hcard", "data-open") == "true"
    page.wait("document.querySelectorAll('.hsteps li').length >= 2")
    page.wait("document.querySelector('.hcard[data-state=done]')", timeout=20000)
    assert "Hermes skończył" in page.pg.inner_text(".hcard .htitle")
    assert page.pg.is_hidden(".hcancel"), "no cancel button on a finished card"
    assert page.pg.locator(".msg.note.hermes").count() >= 2  # started + done are logged in the chat
    no_errors(page)


def test_hermes_cards_are_keyed_by_id_and_merge_streamed_fragments(make_page):
    """The real adapter streams Hermes' answer token by token; two tasks with the same text are two cards."""
    page = make_page(viewport=(390, 844), mobile=True)
    page.pg.evaluate(
        """() => {
        const { tasks } = window.__gaja;
        const ev = (id, state, text) => tasks.handle({ type: 'handoff', state, id, task: 'zadanie', text });
        ev('aaaa1111', 'started', 'safety-net');
        ev('bbbb2222', 'started', 'router');
        ev('aaaa1111', 'progress', 'Hermes uruchamia narzędzie: terminal');
        ev('aaaa1111', 'progress', 'Trwa przetwarzanie narzędzia...');
        ev('aaaa1111', 'progress', '\\n\\nD');
        ev('aaaa1111', 'progress', 'ziś piątek,');
        ev('aaaa1111', 'progress', ' dziewiątego października.');
        ev('bbbb2222', 'progress', 'Drugie zadanie.');
    }"""
    )
    assert page.pg.locator(".hcard").count() == 2
    steps = page.pg.evaluate("[...window.__gaja.tasks.items.get('aaaa1111').steps].map(s => [s.kind, s.text])")
    assert steps == [
        ["started", ""],  # a routing label ("safety-net"), not something to read
        ["progress", "Hermes uruchamia narzędzie: terminal"],
        ["progress", "Trwa przetwarzanie narzędzia..."],
        ["progress", "Dziś piątek, dziewiątego października."],
    ], steps
    assert page.pg.evaluate("window.__gaja.tasks.items.get('bbbb2222').steps.length") == 2
    # `done` of one task leaves the other one running
    page.pg.evaluate("window.__gaja.tasks.handle({ type: 'handoff', state: 'done', id: 'aaaa1111', task: 'zadanie', text: 'Dziś piątek.' })")
    assert page.pg.evaluate("window.__gaja.tasks.items.get('bbbb2222').state") == "working"
    assert page.pg.evaluate("window.__gaja.tasks.active") is True
    no_errors(page)


@needs_mock
def test_barge_in_does_not_cancel_hermes_task(make_page):
    """v1.7: barge-in stops Gaja's speech only; the Hermes card keeps running and then completes."""
    page = make_page("?hermes_s=12")
    page.type("Zapytaj Hermesa jaki jest dziś dzień tygodnia")
    page.wait("document.querySelector('.hcard[data-state=working]')", timeout=15000)
    page.wait("window.__gajaStats.utterancesPlayed >= 1 && window.__gaja.S.server === 'listening'", timeout=20000)
    page.type("przerwij mnie")  # a long answer that the mock's barge-in cuts off mid-way
    page.wait("document.querySelector('.msg.gaja.cut')", timeout=20000)
    page.wait("window.__gaja.S.server === 'listening'")
    assert page.pg.get_attribute(".hcard", "data-state") == "working", "barge-in must not touch the task"
    assert page.pg.locator(".hcard[data-state=cancelled]").count() == 0
    assert page.sent_types("cancel_task") == []
    page.wait("document.querySelector('.hcard[data-state=done]')", timeout=25000)
    assert "Hermes skończył" in page.pg.inner_text(".hcard .htitle")
    no_errors(page)


@needs_mock
def test_cancel_button_cancels_one_task(make_page):
    """The Przerwij button sends cancel_task{id}; the card flips on handoff{cancelled}; no `done` follows."""
    page = make_page("?hermes_s=30")
    page.type("Zapytaj Hermesa o pierwszą rzecz")
    page.wait("document.querySelectorAll('.hcard[data-state=working]').length === 1", timeout=15000)
    page.wait("window.__gajaStats.utterancesPlayed >= 1 && window.__gaja.S.server === 'listening'", timeout=20000)
    page.type("Zapytaj Hermesa o drugą rzecz")
    page.wait("document.querySelectorAll('.hcard[data-state=working]').length === 2", timeout=15000)
    first = page.pg.get_attribute(".hcard >> nth=0", "data-id")
    btn = page.pg.locator(".hcard >> nth=0").locator(".hcancel")
    assert btn.is_visible()
    assert "Przerwij zadanie Hermesa" in btn.get_attribute("aria-label")
    btn.click()
    page.wait("document.querySelector('.hcard[data-state=cancelled]')", timeout=10000)
    assert page.sent_types("cancel_task") == [{"type": "cancel_task", "id": first}]
    assert "Zadanie przerwane" in page.pg.inner_text(".hcard[data-state=cancelled] .htitle")
    assert page.pg.locator(".hcard[data-state=cancelled] .hcancel").is_hidden()
    assert page.pg.locator(".hcard[data-state=working]").count() == 1, "the other task keeps running"
    assert page.pg.get_attribute(".hcard[data-state=working]", "data-id") != first
    assert "Przerwano zadanie" in page.pg.inner_text(".msg.note.hermes >> nth=-1")
    page.pg.wait_for_timeout(600)
    assert page.pg.locator(".hcard[data-state=done]").count() == 0
    no_errors(page)


@needs_mock
def test_voice_cancel_and_late_cancel_error(make_page):
    """`interrupt{scope:all}` cancels every card; a cancel for a task that is gone is an info toast, not an error."""
    page = make_page("?hermes_s=30")
    page.type("Zapytaj Hermesa o coś")
    page.wait("document.querySelector('.hcard[data-state=working]')", timeout=15000)
    page.pg.evaluate("window.__gaja.ws.sendJson({ type: 'interrupt', scope: 'all' })")
    page.wait("document.querySelector('.hcard[data-state=cancelled]')", timeout=10000)
    page.pg.evaluate("window.__gaja.ws.sendCancelTask('deadbeef')")  # nothing running: the server answers `error`
    page.wait("!document.getElementById('toast').hidden")
    assert "już się zakończyło" in page.pg.inner_text("#toast")
    assert page.pg.get_attribute("#toast", "role") == "status"
    assert page.pg.locator(".msg.note.error").count() == 0
    no_errors(page)


@needs_mock
def test_tool_event_renders_a_chip(make_page):
    page = make_page()
    page.type("jaka jest pogoda w Warszawie")
    page.wait("document.querySelector('.tool-chip')", timeout=15000)
    assert page.pg.inner_text(".tool-chip").strip().endswith("pogoda · Warszawa")
    assert page.pg.get_attribute(".tool-chip", "data-tool") == "get_weather"
    page.wait("window.__gajaStats.utterancesPlayed >= 1")
    assert "dwanaście stopni" in page.pg.inner_text(".msg.gaja .bubble")
    order = page.pg.evaluate("[...document.querySelectorAll('#log > li')].map(li => li.className.split(' ').slice(0, 2).join('.'))")
    assert order == ["msg.you", "msg.note", "msg.gaja"], order  # the chip sits between the question and the answer
    assert page.pg.locator("#hermes .hcard").count() == 0, "a tool turn is not a Hermes task"
    # a failing tool gets the warning style
    page.wait("window.__gaja.S.server === 'listening'")
    page.type("awaria narzędzia")
    page.wait("document.querySelector('.tool-chip.fail')", timeout=15000)
    assert "kurs walut" in page.pg.inner_text(".tool-chip.fail")
    no_errors(page)


@needs_mock
def test_stt_notices_show_pill_and_toast(make_page):
    page = make_page()
    assert page.pg.is_hidden("#sttnote")
    page.pg.evaluate("window.__gaja.ws.sendText('stt-degraded')")
    page.wait("!document.getElementById('sttnote').hidden")
    assert "tryb awaryjny" in page.pg.inner_text("#sttnote")
    assert page.pg.get_attribute("#sttnote", "role") == "status"
    page.wait("!document.getElementById('toast').hidden")
    assert "awaryjnie" in page.pg.inner_text("#toast")
    page.pg.evaluate("window.__gaja.ws.sendText('stt-recovered')")
    page.wait("document.getElementById('sttnote').hidden")
    page.wait("document.getElementById('toast').textContent.includes('znów działa')")
    no_errors(page)


@needs_mock
def test_stt_degraded_from_the_start_and_from_health(browser, base):
    # a session that begins degraded gets the notice right after `ready`
    page = Page(browser, base, "?stt=degraded")
    page.wait("!document.getElementById('sttnote').hidden")
    page.close()
    # /health says degraded and the socket stays silent: the pill still shows, the diagnostics line has the engine
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    ctx.route(
        "**/health",
        lambda r: r.fulfill(
            json={"status": "ok", "engines": ["piper"], "stt_engine": "whistle", "stt_device": "cpu", "stt_degraded": True, "router": {"enabled": True, "tools": True}}
        ),
    )
    pg = ctx.new_page()
    pg.goto(base)
    pg.wait_for_function("window.__gaja && window.__gaja.S.conn === 'open'")
    pg.wait_for_selector("#sttnote:not([hidden])")
    pg.click("#btn-settings")
    pg.wait_for_selector("#settings.is-open")
    foot = pg.inner_text("#settings-foot")
    assert "stt: whistle/cpu (fallback)" in foot and "tools on" in foot, foot
    ctx.close()


@needs_mock
def test_result_metrics_do_not_pollute_latency_strip(make_page):
    page = make_page("?hermes_s=2", storage={"metrics": True})
    page.type("Zapytaj Hermesa o coś")
    page.wait("document.querySelector('.hcard[data-state=done]')", timeout=15000)
    page.wait("window.__gajaStats.utterancesPlayed >= 2", timeout=20000)
    page.pg.wait_for_timeout(400)
    assert "9999" not in page.pg.inner_text("#metrics")
    no_errors(page)


@needs_mock
def test_legacy_assistant_delta_still_tolerated(make_page):
    """Older servers stream assistant_delta; the current one does not. Both must build one clean bubble."""
    page = make_page("?deltas=1")
    page.type("Powiedz coś")
    page.wait("window.__gajaStats.utterancesPlayed >= 2", timeout=25000)
    page.wait("document.body.dataset.state === 'idle'", timeout=20000)
    assert page.pg.locator(".msg.gaja .sent").count() >= 2
    assert page.pg.inner_text(".msg.gaja .tail").strip() == ""
    no_errors(page)


def settings_ids(page):
    return page.pg.eval_on_selector_all("#voices input", "els => els.map(e => e.value)")


@needs_mock
def test_settings_voice_list_from_server_and_switch_engine(make_page):
    page = make_page()
    page.pg.click("#btn-settings")
    page.wait("document.getElementById('settings').classList.contains('is-open')")
    assert settings_ids(page) == ["", "piper", "piper-darkman", "paradee", "supertonic"]
    # pick the 44.1 kHz engine: the UI says hello again (no reconnect) and follows `ready`
    page.pg.click("#voices label:has-text('Supertonic')")
    page.wait("window.__gaja.S.ttsName === 'supertonic' && window.__gaja.S.sampleRate === 44100")
    assert json.loads(page.pg.evaluate("localStorage.getItem('gaja.settings')"))["tts"] == "supertonic"
    page.pg.keyboard.press("Escape")
    page.wait("!document.getElementById('settings').classList.contains('is-open')")
    page.type("Cześć")
    page.wait("window.__gajaStats.lastRate === 44100 && window.__gajaStats.utterancesPlayed >= 1", timeout=30000)
    no_errors(page)


@needs_mock
def test_settings_voice_list_without_server_list(browser, base):
    """An older server: no `ready.engines` and no /health list -> the two basic engines are offered."""
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    ctx.route("**/health", lambda r: r.abort())
    pg = ctx.new_page()
    pg.goto(base + "?engines=none")
    pg.wait_for_function("window.__gaja && window.__gaja.S.conn === 'open'")
    pg.click("#btn-settings")
    pg.wait_for_selector("#settings.is-open #voice-note")
    ids = pg.eval_on_selector_all("#voices input", "els => els.map(e => e.value)")
    assert ids == ["", "piper", "paradee"], ids
    assert "podstawowe" in pg.inner_text("#voice-note")
    ctx.close()


def test_stale_saved_voice_falls_back_to_auto(make_page):
    page = make_page(storage={"tts": "does-not-exist"})
    page.wait("window.__gaja.S.sessionId !== ''", timeout=8000)
    assert page.pg.evaluate("window.__gaja.settings.tts") == ""
    no_errors(page)


def test_language_switch_translates_ui(make_page):
    page = make_page()
    assert page.pg.inner_text("#status-text") == "Gotowa"
    page.pg.click("#btn-settings")
    page.pg.click("#settings .seg label:has-text('English')")
    page.wait("document.getElementById('status-text').textContent === 'Ready'")
    assert page.pg.get_attribute("html", "lang") == "en"
    assert page.pg.get_attribute("#text-input", "placeholder") == "Write to Gaja…"
    no_errors(page)


def test_theme_follows_system_and_can_be_overridden(browser, base):
    def bg(scheme, storage=None):
        pg = Page(browser, base, scheme=scheme, storage=storage)
        color = pg.pg.evaluate("getComputedStyle(document.body).backgroundColor")
        pg.close()
        return color

    dark, light = bg("dark"), bg("light")
    assert dark != light
    assert bg("light", {"theme": "dark"}) == dark
    assert bg("dark", {"theme": "light"}) == light


def test_metrics_toggle(make_page):
    page = make_page()
    assert page.pg.is_hidden("#metrics")
    page.pg.click("#btn-settings")
    page.pg.click("#settings .switch:has(#set-metrics)")
    page.wait("!document.getElementById('metrics').hidden")
    no_errors(page)


@needs_mock
def test_reconnects_after_drop(make_page):
    page = make_page("?demo=drop&down=3")
    page.wait("document.body.dataset.state === 'offline'", timeout=10000)
    assert page.pg.is_visible("#conn")
    page.wait("window.__gaja.S.conn === 'open'", timeout=20000)
    no_errors(page)


def test_phone_layout_does_not_overflow(make_page):
    page = make_page(viewport=(390, 844), mobile=True)
    m = page.pg.evaluate("({ iw: innerWidth, sw: document.documentElement.scrollWidth })")
    assert m["sw"] <= m["iw"], m


def test_manifest_and_icons_are_served(base):
    man = json.load(urllib.request.urlopen(base + "manifest.webmanifest", timeout=5))
    assert man["name"] == "Gaja" and man["display"] == "standalone"
    for icon in man["icons"]:
        with urllib.request.urlopen(base + icon["src"], timeout=5) as r:
            assert r.status == 200 and len(r.read()) > 200
    with urllib.request.urlopen(base + "icons/apple-touch-icon.png", timeout=5) as r:
        assert r.status == 200


# ------------------------------------------------------------------ New tests: History, PWA, Settings, Diag

@needs_mock
def test_history_drawer_and_persistence(make_page):
    page = make_page()
    page.type("Dzień dobry z testu")
    page.wait("document.querySelector('.msg.you')")
    page.pg.click("#btn-history")
    page.wait("document.getElementById('history-drawer').classList.contains('is-open')")
    page.wait("document.querySelector('.history-item')")
    items = page.pg.eval_on_selector_all(".history-item", "els => els.map(e => e.textContent)")
    assert any("Dzień dobry z testu" in it for it in items), items

    # Search filter
    page.pg.fill("#history-search", "Dzień dobry")
    page.wait("document.querySelectorAll('.history-item').length >= 1")
    page.pg.fill("#history-search", "nieistniejące-hasło-xyz")
    page.wait("document.querySelector('.history-empty')")

    # Clear history (accept window.confirm)
    page.pg.on("dialog", lambda dialog: dialog.accept())
    page.pg.fill("#history-search", "")
    page.pg.click("#btn-clear-history")
    page.wait("document.querySelector('.history-empty')")
    page.pg.click("#history-close")
    page.wait("!document.getElementById('history-drawer').classList.contains('is-open')")
    no_errors(page)


@needs_mock
def test_settings_vocabulary_management(make_page):
    page = make_page()
    page.pg.click("#btn-settings")
    page.wait("document.getElementById('settings').classList.contains('is-open')")
    page.wait("!document.getElementById('sec-vocab').hidden")
    page.wait("document.querySelector('.vocab-item')")

    # Check terms
    terms = page.pg.eval_on_selector_all(".vocab-item strong", "els => els.map(e => e.textContent)")
    assert "Kubernetes" in terms, terms

    # Check suggestion and approve
    page.wait("document.querySelector('.sug-card')")
    sug_term = page.pg.inner_text(".sug-card strong")
    assert sug_term == "Grafana"
    page.pg.click(".sug-card .btn.primary")
    page.wait("document.querySelectorAll('.sug-card').length === 0")

    # Check that Grafana was added to terms list
    updated_terms = page.pg.eval_on_selector_all(".vocab-item strong", "els => els.map(e => e.textContent)")
    assert "Grafana" in updated_terms, updated_terms
    no_errors(page)


@needs_mock
def test_settings_memory_management(make_page):
    page = make_page()
    page.pg.click("#btn-settings")
    page.wait("document.getElementById('settings').classList.contains('is-open')")
    page.wait("!document.getElementById('sec-memory').hidden")
    page.wait("document.querySelector('.memory-item')")

    # Add a new fact
    page.pg.fill("#input-memory-text", "Lubię gorącą herbatę")
    page.pg.click("#form-add-memory button[type=submit]")
    page.wait("document.querySelectorAll('.memory-item').length >= 2")

    # Delete fact
    page.pg.click(".memory-item:last-child .btn-del")
    page.wait("document.querySelectorAll('.memory-item').length === 1")
    no_errors(page)


@needs_mock
def test_settings_timers_and_ws_notification(make_page):
    page = make_page()
    page.pg.click("#btn-settings")
    page.wait("document.getElementById('settings').classList.contains('is-open')")
    page.wait("!document.getElementById('sec-timers').hidden")
    page.wait("document.querySelector('.timer-item')")
    assert "pizza" in page.pg.inner_text(".timer-item")

    # Trigger timer WS event via test message
    page.pg.keyboard.press("Escape")
    page.wait("!document.getElementById('settings').classList.contains('is-open')")
    page.pg.evaluate("window.__gaja.ws.sendText(JSON.stringify({type: 'trigger_timer', label: 'herbata'}))")
    page.wait("!document.getElementById('toast').hidden")
    assert "herbata" in page.pg.inner_text("#toast")
    no_errors(page)


@needs_mock
def test_settings_devices_and_diagnostics(make_page):
    page = make_page()
    page.pg.click("#btn-settings")
    page.wait("document.getElementById('settings').classList.contains('is-open')")

    # Devices
    page.wait("!document.getElementById('sec-devices').hidden")
    page.wait("document.querySelector('.device-card')")
    assert "Salon" in page.pg.inner_text(".device-card")

    # Diagnostics
    page.wait("!document.getElementById('sec-diagnostics').hidden")
    page.wait("document.getElementById('diag-gpu').textContent !== 'Ładowanie…'")
    gpu_text = page.pg.inner_text("#diag-gpu")
    assert "GTX 1050 Ti" in gpu_text or "GPU" in gpu_text
    stages_text = page.pg.inner_text("#diag-stages")
    assert "STT" in stages_text or "Router" in stages_text or "Oczekiwanie" in stages_text
    page.wait("document.querySelector('.wake-row')")
    assert "atom-salon" in page.pg.inner_text(".wake-row")
    no_errors(page)


@needs_mock
def test_graceful_404_degradation(browser, base):
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    pg = ctx.new_page()
    pg.goto(base + "?mock404=1")
    pg.wait_for_function("window.__gaja && window.__gaja.S.conn === 'open'")
    pg.click("#btn-settings")
    pg.wait_for_selector("#settings.is-open")
    # All 404 endpoints must degrade gracefully by hiding sections
    pg.wait_for_function("document.getElementById('sec-vocab').hidden")
    pg.wait_for_function("document.getElementById('sec-memory').hidden")
    pg.wait_for_function("document.getElementById('sec-timers').hidden")
    ctx.close()


def test_service_worker_served(base):
    with urllib.request.urlopen(base + "sw.js", timeout=5) as r:
        assert r.status == 200
        content = r.read().decode("utf-8")
        assert "CACHE_NAME" in content
        assert "/api/" in content


@needs_mock
def test_capture_required_screenshots(browser, base):
    screens_dir = "/tmp/gaja-night/screens"
    os.makedirs(screens_dir, exist_ok=True)

    # 1. Desktop viewport (1280x800)
    ctx_desktop = browser.new_context(viewport={"width": 1280, "height": 800})
    pg_d = ctx_desktop.new_page()
    pg_d.goto(base)
    pg_d.wait_for_function("window.__gaja && window.__gaja.S.conn === 'open'")
    time.sleep(0.5)

    # Desktop main
    pg_d.screenshot(path=os.path.join(screens_dir, "desktop-main.png"))

    # Desktop history
    pg_d.click("#btn-history")
    pg_d.wait_for_selector("#history-drawer.is-open")
    time.sleep(0.4)
    pg_d.screenshot(path=os.path.join(screens_dir, "desktop-history.png"))
    pg_d.click("#history-close")
    pg_d.wait_for_selector("#history-drawer:not(.is-open)")

    # Desktop settings
    pg_d.click("#btn-settings")
    pg_d.wait_for_selector("#settings.is-open")
    time.sleep(0.5)
    pg_d.screenshot(path=os.path.join(screens_dir, "desktop-settings.png"))

    # Desktop diagnostics
    pg_d.eval_on_selector("#sec-diagnostics", "el => el.scrollIntoView()")
    time.sleep(0.4)
    pg_d.screenshot(path=os.path.join(screens_dir, "desktop-diagnostics.png"))
    ctx_desktop.close()

    # 2. Phone viewport (390x844)
    ctx_phone = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True)
    pg_p = ctx_phone.new_page()
    pg_p.goto(base)
    pg_p.wait_for_function("window.__gaja && window.__gaja.S.conn === 'open'")
    time.sleep(0.5)

    # Phone main
    pg_p.screenshot(path=os.path.join(screens_dir, "phone-main.png"))

    # Phone history
    pg_p.click("#btn-history")
    pg_p.wait_for_selector("#history-drawer.is-open")
    time.sleep(0.4)
    pg_p.screenshot(path=os.path.join(screens_dir, "phone-history.png"))
    pg_p.click("#history-close")
    pg_p.wait_for_selector("#history-drawer:not(.is-open)")

    # Phone settings
    pg_p.click("#btn-settings")
    pg_p.wait_for_selector("#settings.is-open")
    time.sleep(0.5)
    pg_p.screenshot(path=os.path.join(screens_dir, "phone-settings.png"))

    # Phone diagnostics
    pg_p.eval_on_selector("#sec-diagnostics", "el => el.scrollIntoView()")
    time.sleep(0.4)
    pg_p.screenshot(path=os.path.join(screens_dir, "phone-diagnostics.png"))
    ctx_phone.close()

    # Verify all 8 files exist
    expected = [
        "desktop-main.png", "desktop-history.png", "desktop-settings.png", "desktop-diagnostics.png",
        "phone-main.png", "phone-history.png", "phone-settings.png", "phone-diagnostics.png",
    ]
    for fname in expected:
        fpath = os.path.join(screens_dir, fname)
        assert os.path.isfile(fpath) and os.path.getsize(fpath) > 1000, fpath
