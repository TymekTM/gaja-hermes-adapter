#!/usr/bin/env python3
"""Drives the web UI against the REAL adapter (typed + fake-mic turns, voice switch, Hermes handoff, barge-in)
and saves phone screenshots as docs/ui/real-*.png. Read-only on the server side: it only talks over WS /ws.

    TMPDIR=/tmp .venv/bin/python clients/real_adapter_check.py [--base http://127.0.0.1:8790/] [--only a,b,c,d,e,f,g,h,i,j,k]

Prints one JSON report line per check. Needs Playwright and a system Chromium.
"""

import argparse
import asyncio
import json
import os
import threading
import time
import wave

from playwright.sync_api import sync_playwright

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "docs", "ui")
SRC_WAV = os.path.join(ROOT, "bench", "out", "e2e", "handoff_pl_16k.wav")  # a short Polish request
FAKE_H_WAV = "/tmp/gaja_fake_mic_pl_loop30.wav"  # short non-handoff phrase + 25 s of silence: loops mid-reply
SRC_H_WAV = os.path.join(ROOT, "bench", "data", "stt", "pl", "latency", "pl_lat_001_3.20s.wav")
FAKE_WAV = "/tmp/gaja_fake_mic_pl.wav"  # the same, padded with silence: Chrome loops the file, the VAD needs pauses
CHROME_ARGS = [
    "--no-sandbox",
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
]


class Session:
    def __init__(self, browser, base, wav=None, device="phone", scheme="dark"):
        phone = device == "phone"
        self.ctx = browser.new_context(
            viewport={"width": 390, "height": 844} if phone else {"width": 1440, "height": 900},
            device_scale_factor=2 if phone else 1, is_mobile=phone, has_touch=phone,
            permissions=["microphone"], color_scheme=scheme, locale="pl-PL",
        )
        self.pg = self.ctx.new_page()
        self.events = []
        self.sent = []
        self.errors = []
        self.t0 = time.time()
        self.pg.on("pageerror", lambda e: self.errors.append(f"[pageerror] {e}"))
        self.pg.on("console", lambda m: self.errors.append(f"[{m.type}] {m.text}") if m.type == "error" else None)
        self.pg.on("websocket", self._ws)
        self.pg.goto(base)
        self.pg.wait_for_function("window.__gaja && window.__gaja.S.conn === 'open'", timeout=20000)

    def _ws(self, ws):
        def got(payload):
            if isinstance(payload, str):
                try:
                    d = json.loads(payload)
                except ValueError:
                    return
                self.events.append((round(time.time() - self.t0, 2), d))

        def sent(payload):
            if isinstance(payload, str):
                try:
                    self.sent.append((round(time.time() - self.t0, 2), json.loads(payload)))
                except ValueError:
                    pass

        ws.on("framereceived", got)
        ws.on("framesent", sent)

    def wait(self, expr, timeout=60000):
        self.pg.wait_for_function(expr, timeout=timeout)

    def stats(self):
        return self.pg.evaluate("({ ...window.__gajaStats })")

    def shot(self, name):
        self.pg.screenshot(path=os.path.join(OUT, f"real-{name}.png"))
        return name

    def type(self, text):
        self.pg.evaluate("document.body.dataset.input !== 'open' && document.querySelector('#btn-keyboard').click()")
        self.pg.fill("#text-input", text)
        self.pg.keyboard.press("Enter")

    def kinds(self, kind, since=0):
        return [(t, e) for t, e in self.events if e.get("type") == kind and t >= since]

    def close(self):
        self.ctx.close()


def dump(s, n=25):
    """what happened on the wire: shown when a check fails"""
    ev = [(t, e.get("type"), e.get("state") or e.get("text") or "") for t, e in s.events if e.get("type") not in ("pong", "audio_end")]
    print(json.dumps({"debug_events": ev[-n:], "stats": s.stats(), "state": s.pg.evaluate("document.body.dataset.state")}, ensure_ascii=False, default=str), flush=True)


def report(name, **kw):
    print(json.dumps({"check": name, **kw}, ensure_ascii=False, default=str), flush=True)


def check_a_typed(browser, base):
    s = Session(browser, base)
    assert s.pg.evaluate("window.__gaja.S.mic") == "off"
    s.type("Cześć Gaja, opowiedz mi krótko, co potrafisz.")
    s.wait("window.__gajaStats.framesReceived >= 1")
    s.shot("thinking-or-start")
    s.wait("window.__gajaStats.samplesPlayed > 2000")
    s.wait("document.body.dataset.state === 'speaking'")
    s.wait("document.querySelector('.sent[data-state=playing]')")
    s.shot("speaking")
    s.wait("window.__gajaStats.utterancesPlayed >= 1")
    s.wait("document.body.dataset.state === 'idle'", timeout=120000)
    st = s.stats()
    s.shot("typed-done")
    report("a_typed_turn_audio_without_mic", ok=st["samplesPlayed"] > 2000 and st["framesSent"] == 0, ctx=st["ctx"],
           samples=st["samplesPlayed"], utterances=st["utterancesPlayed"], underruns=st["underruns"], frames_sent=st["framesSent"],
           errors=s.errors)
    s.close()


def check_b_mic(browser_mic, base):
    s = Session(browser_mic, base)
    s.pg.evaluate("window.__gaja.settings.bargein = false")  # looping fake mic would otherwise keep cutting the reply
    s.pg.click("#btn-mic")
    s.wait("window.__gaja.S.mic === 'live'")
    s.wait("window.__gajaStats.framesSent > 20")
    s.shot("listening")
    try:
        s.wait("document.body.dataset.state === 'user'", timeout=30000)
        s.shot("user-speaking")
        s.wait("document.querySelector('.msg.you .bubble') && !document.querySelector('.msg.you.ghost')", timeout=120000)
        s.shot("thinking")
        s.wait("window.__gajaStats.utterancesPlayed >= 1", timeout=180000)
    except Exception:
        dump(s)
        s.shot("mic-failure")
        raise
    t = s.kinds("transcript")
    m = s.kinds("metrics")
    s.shot("mic-reply")
    st = s.stats()
    report("b_mic_turn", ok=bool(t) and st["samplesPlayed"] > 2000, transcript=[e["text"] for _, e in t][:2],
           metrics=[e for _, e in m][:1], samples=st["samplesPlayed"], underruns=st["underruns"], errors=s.errors)
    s.pg.click("#btn-mic")
    s.close()


def check_c_voices(browser, base):
    s = Session(browser, base)
    voices = s.pg.evaluate("[...document.querySelectorAll('#voices input')].map(i => i.value)")
    out = {"voices": voices}
    s.pg.click("#btn-settings")
    s.wait("document.querySelector('#settings:not([inert])')")
    s.shot("settings")
    for v in ("piper-darkman", "supertonic"):
        t0 = time.time()
        before = s.stats()["utterancesPlayed"]
        s.pg.click(f"#voices label:has(input[value={v}])")
        s.wait(f"window.__gaja.S.ttsName === '{v}'", timeout=60000)
        out[f"{v}_ready_s"] = round(time.time() - t0, 1)
        s.shot(f"settings-{v}")
        s.pg.click("#settings-close")
        s.type("Powiedz coś o pogodzie w ogólności, jednym zdaniem.")
        s.wait(f"window.__gajaStats.utterancesPlayed > {before}", timeout=120000)
        out[f"{v}_first_audio_s"] = round(time.time() - t0, 1)
        out[f"{v}_rate"] = s.stats()["lastRate"]
        s.wait("document.body.dataset.state === 'idle'", timeout=120000)
        s.pg.click("#btn-settings")
        s.wait("document.querySelector('#settings:not([inert])')")
    s.pg.click("#settings-close")
    report("c_voice_switch", ok=True, **out, errors=s.errors)
    s.close()


def check_d_hermes(browser, base):
    s = Session(browser, base)
    s.type("Sprawdź w Hermesie, jaki jest dziś dzień tygodnia.")
    s.wait("window.__gaja.tasks.active || document.querySelector('#hermes:not([hidden])')", timeout=120000)
    s.shot("hermes-started")
    t_start = time.time()
    s.wait("document.querySelector('#hermes [data-state=done]') || document.querySelector('#hermes [data-state=error]')", timeout=180000)
    s.shot("hermes-done")
    hs = [(t, e["state"], (e.get("text") or "")[:90]) for t, e in s.kinds("handoff")]
    s.wait("document.body.dataset.state === 'idle'", timeout=120000)
    s.shot("hermes-after")
    report("d_hermes_handoff", ok=any(h[1] == "done" for h in hs), seconds=round(time.time() - t_start, 1), handoff_events=hs[:14],
           n_progress=sum(1 for h in hs if h[1] == "progress"), errors=s.errors)
    s.close()


def check_e_barge(browser, base):
    s = Session(browser, base)
    s.pg.evaluate("window.__gaja.settings.bargein = true")
    s.type("Opowiedz mi długą historię o smoku, kilka zdań.")
    s.wait("document.body.dataset.state === 'speaking'", timeout=120000)
    s.wait("window.__gajaStats.samplesPlayed > 4000")
    # same thing the mic gate does when the user starts talking: the client flushes immediately
    t0 = s.pg.evaluate("performance.now()")
    s.pg.click("#btn-mic")  # while speaking the big button is "stop"
    s.wait("window.__gajaStats.flushes >= 1", timeout=5000)
    s.wait("window.__gajaStats.buffered < 200", timeout=5000)
    dt = s.pg.evaluate(f"performance.now() - {t0}")
    s.shot("barge-in")
    s.pg.wait_for_timeout(1200)
    a = s.stats()["samplesPlayed"]
    s.pg.wait_for_timeout(1500)
    b = s.stats()["samplesPlayed"]
    report("e_barge_in", ok=(b - a) < 200, stop_ms=round(dt), samples_growth_after_stop=b - a, state=s.pg.evaluate("document.body.dataset.state"),
           cut=s.pg.locator(".msg.gaja.cut").count(), errors=s.errors)
    s.close()


def check_f_english(browser, base):
    s = Session(browser, base)
    s.pg.click("#btn-settings")
    s.wait("document.querySelector('#settings:not([inert])')")
    s.pg.click(".seg[data-setting=lang] label:has(input[value=en])")
    s.wait("document.documentElement.lang === 'en'")
    s.wait("window.__gaja.S.conn === 'open' && !window.__gaja.S.awaitReady", timeout=15000)
    title = s.pg.inner_text("#settings-title")
    s.pg.click("#settings-close")
    s.type("Hello Gaja, tell me in one sentence what you can do.")
    s.wait("window.__gajaStats.utterancesPlayed >= 1", timeout=120000)
    s.wait("document.body.dataset.state === 'idle'", timeout=120000)
    st = s.stats()
    s.shot("english")
    report("f_english", ok=title == "Settings" and st["samplesPlayed"] > 2000, title=title, tts=s.pg.evaluate("window.__gaja.S.ttsName"),
           rate=st["lastRate"], reply=s.pg.inner_text(".msg.gaja .bubble")[:120], errors=s.errors)
    s.close()


class Proxy:
    """A TCP forwarder that we can stop and restart: the adapter itself is never touched."""

    def __init__(self, port, target=("127.0.0.1", 8790)):
        self.port, self.target, self.server, self.conns = port, target, None, set()
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(10)

    async def _pipe(self, r, w):
        try:
            while data := await r.read(65536):
                w.write(data)
                await w.drain()
        except Exception:
            pass
        finally:
            w.close()

    async def _on_conn(self, r, w):
        try:
            ur, uw = await asyncio.open_connection(*self.target)
        except OSError:
            w.close()
            return
        self.conns.update((w, uw))
        await asyncio.gather(self._pipe(r, uw), self._pipe(ur, w))

    async def _start(self):
        self.server = await asyncio.start_server(self._on_conn, "127.0.0.1", self.port)

    async def _stop(self):
        self.server.close()
        for w in list(self.conns):
            w.close()
        self.conns.clear()
        await self.server.wait_closed()

    def start(self):
        self._run(self._start())

    def stop(self):
        self._run(self._stop())


def check_g_reconnect(browser, base):
    px = Proxy(8798)
    px.start()
    s = Session(browser, "http://127.0.0.1:8798/")
    try:
        s.type("Powiedz jedno krótkie zdanie.")
        s.wait("window.__gajaStats.utterancesPlayed >= 1", timeout=120000)
        s.wait("document.body.dataset.state === 'idle'", timeout=120000)
        px.stop()  # == adapter restart: every socket dies, new connects are refused
        s.wait("window.__gaja.S.conn !== 'open'", timeout=10000)
        s.pg.wait_for_timeout(2500)
        offline = s.pg.evaluate("document.body.dataset.state")
        s.shot("offline")
        px.start()
        t0 = time.time()
        s.wait("window.__gaja.S.conn === 'open' && !window.__gaja.S.awaitReady", timeout=30000)
        back_s = round(time.time() - t0, 1)
        before = s.stats()["utterancesPlayed"]
        s.type("Powiedz jeszcze jedno krótkie zdanie.")
        s.wait(f"window.__gajaStats.utterancesPlayed > {before}", timeout=120000)
        report("g_reconnect_after_restart", ok=True, offline_state=offline, reconnected_after_s=back_s, history_kept_in_ui=s.pg.locator(".msg.you").count() >= 2, errors=len(s.errors))
    finally:
        s.close()
        px.stop()


def check_h_voice_bargein(browser_mic, base):
    """a long reply is playing; the fake mic (looping Polish phrase) starts talking -> playback must stop, server goes listening"""
    s = Session(browser_mic, base)
    s.pg.evaluate("window.__gaja.settings.bargein = true")
    s.pg.click("#btn-mic")
    s.wait("window.__gaja.S.mic === 'live'", timeout=15000)
    # the looping phrase makes one normal turn first; then ask for a long story and let the next loop barge in
    s.wait("window.__gajaStats.utterancesPlayed >= 1", timeout=60000)
    s.wait("document.body.dataset.state === 'listening' || document.body.dataset.state === 'idle'", timeout=60000)
    s.type("Opowiedz mi bardzo długą historię o smoku, co najmniej dziesięć zdań.")
    s.wait("window.__gaja.S.playing === true", timeout=120000)
    n0 = s.stats()["flushes"]
    try:
        s.wait(f"window.__gajaStats.flushes > {n0} && window.__gaja.S.server === 'listening'", timeout=60000)
    except Exception:
        dump(s)
        s.shot("voice-bargein-failure")
        raise
    s.shot("voice-barge-in")
    a = s.stats()["samplesPlayed"]
    s.pg.wait_for_timeout(800)
    b = s.stats()["samplesPlayed"]
    states = [(t, e.get("state"), e.get("reason")) for t, e in s.events if e.get("type") == "state"][-6:]
    report("h_voice_barge_in", ok=True, growth_after=b - a, states=states, cut=s.pg.locator(".msg.gaja.cut").count(), errors=s.errors)
    s.pg.evaluate("window.__gaja.S.mic === 'live' && document.querySelector('#btn-mic').click()")
    s.close()


def hermes_events(s):
    return [(t, e["state"], e.get("id"), (e.get("text") or "")[:80]) for t, e in s.kinds("handoff")]


def check_i_tool(browser, base):
    """v1.7 tool route: a `tool` event, a chip in the transcript, the spoken answer as a normal bubble; plus /health diagnostics"""
    s = Session(browser, base)
    s.type("jaka jest pogoda w Warszawie")
    try:
        s.wait("document.querySelector('.tool-chip')", timeout=60000)
        s.shot("tool-chip-start")
        s.wait("window.__gajaStats.utterancesPlayed >= 1", timeout=120000)
        s.wait("document.body.dataset.state === 'idle'", timeout=120000)
    except Exception:
        dump(s)
        s.shot("tool-failure")
        raise
    s.shot("tool-chip")
    tools = [e for _, e in s.kinds("tool")]
    chip = s.pg.inner_text(".tool-chip").strip()
    order = s.pg.evaluate("[...document.querySelectorAll('#log > li')].map(li => li.className.split(' ').slice(0, 2).join('.'))")
    s.pg.click("#btn-settings")
    s.wait("document.querySelector('#settings:not([inert])')")
    s.pg.wait_for_timeout(700)
    foot = s.pg.inner_text("#settings-foot")
    s.shot("settings-diag")
    report("i_tool_turn", ok=bool(tools) and chip != "", chip=chip, tool_event=tools[:1], log_order=order,
           reply=s.pg.inner_text(".msg.gaja .bubble")[:140], stt_pill_hidden=s.pg.is_hidden("#sttnote"), diag=foot.splitlines()[:3],
           notices=[e for _, e in s.kinds("notice")], errors=s.errors)
    s.close()


def check_j_hermes_survives_typing(browser, base):
    """a read-only Hermes task, then a typed turn while it runs: the card must stay `working`, then complete"""
    s = Session(browser, base)
    s.type("Zapytaj Hermesa jaki jest dziś dzień tygodnia.")
    try:
        s.wait("document.querySelector('.hcard[data-state=working]')", timeout=120000)
    except Exception:
        dump(s)
        raise
    card_id = s.pg.get_attribute(".hcard", "data-id")
    s.wait("window.__gajaStats.utterancesPlayed >= 1 || document.body.dataset.state === 'speaking'", timeout=60000)
    s.shot("hermes-running")
    t_type = time.time() - s.t0
    s.type("Opowiedz mi jednym zdaniem dowcip o kocie.")
    s.wait("document.querySelectorAll('.msg.you').length >= 2", timeout=20000)
    s.pg.wait_for_timeout(1500)
    state_after_typing = s.pg.get_attribute(".hcard", "data-state")
    done_before = any(h[1] == "done" for h in hermes_events(s))
    s.shot("hermes-running-typed")
    try:
        s.wait("document.querySelector('.hcard[data-state=done]') || document.querySelector('.hcard[data-state=error]')", timeout=180000)
        s.wait("document.body.dataset.state === 'idle'", timeout=180000)
    except Exception:
        dump(s)
        raise
    s.pg.click(".hcard-head")
    s.pg.wait_for_timeout(1000)
    s.shot("hermes-done-v17")
    hs = hermes_events(s)
    states = [h[1] for h in hs]
    ids = {h[2] for h in hs}
    t_done = next((h[0] for h in hs if h[1] == "done"), None)
    spoken_after = [e.get("text", "")[:80] for t, e in s.kinds("utterance") if t_done is not None and t >= t_done]
    report("j_hermes_survives_typed_interrupt", ok=state_after_typing in ("working", "done") and "cancelled" not in states and "done" in states and len(ids) == 1,
           card_id=card_id, state_1_5s_after_typing=state_after_typing, done_already_before_check=done_before,
           typed_at_s=round(t_type, 1), result_announced=spoken_after, handoff_states=states, ids=sorted(ids), result=[h[3] for h in hs if h[1] == "done"],
           cancel_sent=[m for _, m in s.sent if m.get("type") in ("cancel_task", "interrupt")], errors=s.errors)
    s.close()


def check_k_cancel_button(browser, base):
    """a second harmless Hermes task cancelled with the card's button: cancel_task{id} -> handoff{cancelled}, no done"""
    s = Session(browser, base)
    s.type("Zapytaj Hermesa jaki jest dziś dzień miesiąca.")
    try:
        s.wait("document.querySelector('.hcard[data-state=working] .hcancel:not([hidden])')", timeout=120000)
    except Exception:
        dump(s)
        raise
    card_id = s.pg.get_attribute(".hcard", "data-id")
    s.pg.wait_for_timeout(800)
    s.shot("hermes-cancel-button")
    s.pg.click(".hcard[data-state=working] .hcancel")
    try:
        s.wait("document.querySelector('.hcard[data-state=cancelled]')", timeout=30000)
    except Exception:
        dump(s)
        s.shot("hermes-cancel-failure")
        raise
    s.shot("hermes-cancelled")
    s.pg.wait_for_timeout(9000)  # no `done` may follow
    hs = hermes_events(s)
    s.wait("document.body.dataset.state === 'idle'", timeout=60000)
    report("k_cancel_button", ok=any(h[1] == "cancelled" and h[2] == card_id for h in hs) and not any(h[1] == "done" for h in hs),
           card_id=card_id, handoff=[(h[1], h[2]) for h in hs], sent=[m for _, m in s.sent if m.get("type") == "cancel_task"],
           card_state=s.pg.get_attribute(".hcard", "data-state"), errors_shown=s.pg.locator(".msg.note.error").count(), errors=s.errors)
    s.close()


def check_s_gallery(browser, base):
    """phone + desktop x dark + light: a typed conversation and a Hermes hand-off (read-only question)"""
    for device in ("phone", "desktop"):
        for scheme in ("dark", "light"):
            s = Session(browser, base, device=device, scheme=scheme)
            tag = f"{device}-{scheme}"
            s.type("Cześć Gaja, opowiedz mi krótko, co potrafisz.")
            s.wait("document.body.dataset.state === 'speaking'", timeout=120000)
            s.wait("document.querySelector('.sent[data-state=playing]')")
            s.pg.wait_for_timeout(600)
            s.shot(f"{tag}-conversation-speaking")
            s.wait("document.body.dataset.state === 'idle'", timeout=120000)
            s.shot(f"{tag}-conversation")
            s.type("Sprawdź w Hermesie, jaki jest dziś dzień tygodnia.")
            s.wait("document.querySelector('#hermes:not([hidden])')", timeout=120000)
            s.wait("document.querySelector('#hermes [data-state=working]') || document.querySelector('#hermes [data-state=done]')", timeout=60000)
            s.shot(f"{tag}-hermes-working")
            s.wait("document.querySelector('#hermes [data-state=done]') || document.querySelector('#hermes [data-state=error]')", timeout=180000)
            s.wait("document.body.dataset.state === 'idle'", timeout=120000)
            s.pg.click("#hermes .hcard-head")  # expand the timeline
            s.pg.wait_for_timeout(1200)
            s.shot(f"{tag}-hermes-done")
            report("s_gallery", ok=True, variant=tag, errors=s.errors)
            s.close()


CHECKS = {"a": check_a_typed, "b": check_b_mic, "c": check_c_voices, "d": check_d_hermes, "e": check_e_barge, "f": check_f_english, "g": check_g_reconnect, "h": check_h_voice_bargein, "i": check_i_tool, "j": check_j_hermes_survives_typing, "k": check_k_cancel_button, "s": check_s_gallery}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8790/")
    ap.add_argument("--chromium", default="/usr/lib/chromium/chromium")
    ap.add_argument("--only", default="a,b,c,d,e,f,g,h")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.environ.setdefault("TMPDIR", "/tmp")
    with wave.open(SRC_WAV) as w:
        params, data = w.getparams(), w.readframes(w.getnframes())
    with wave.open(FAKE_WAV, "wb") as o:
        o.setparams(params)
        o.writeframes(b"\0\0" * 8000 + data + b"\0\0" * 16000 * 8)
    with wave.open(SRC_H_WAV) as w:
        hp, hd = w.getparams(), w.readframes(w.getnframes())
    with wave.open(FAKE_H_WAV, "wb") as o:
        o.setparams(hp)
        o.writeframes(b"\0\0" * 8000 + hd + b"\0\0" * 16000 * 25)
    with sync_playwright() as p:
        plain = p.chromium.launch(executable_path=args.chromium, args=CHROME_ARGS + [f"--use-file-for-fake-audio-capture={FAKE_WAV}"])
        loop30 = p.chromium.launch(executable_path=args.chromium, args=CHROME_ARGS + [f"--use-file-for-fake-audio-capture={FAKE_H_WAV}"])
        for key in args.only.split(","):
            try:
                CHECKS[key](loop30 if key == "h" else plain, args.base)
            except Exception as e:  # keep going: one failing check should not hide the others
                report(key, ok=False, error=repr(e)[:600])
        plain.close()
        loop30.close()


if __name__ == "__main__":
    main()
