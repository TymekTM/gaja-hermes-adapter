#!/usr/bin/env python3
"""Takes a screenshot of every UI state at phone and desktop sizes into docs/ui/.

Starts nothing itself: run `python clients/mock_server.py --port 8799` first (or pass --base, e.g. the real
adapter at http://127.0.0.1:8790/ - the states that need a scripted server are skipped there).
Needs Playwright and a system Chromium (default /usr/lib/chromium/chromium).

    TMPDIR=/tmp .venv/bin/python clients/ui_screenshots.py [--only speaking-happy,history] [--sizes phone]
"""

import argparse
import os

from playwright.sync_api import sync_playwright

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "docs", "ui")
SIZES = {"phone": (390, 844, 2), "desktop": (1440, 900, 1)}
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


def state(pg, name, timeout=15000):
    pg.wait_for_function(f"document.body.dataset.state === '{name}'", timeout=timeout)


def played(pg, n, timeout=30000):
    pg.wait_for_function(f"window.__gajaStats.utterancesPlayed >= {n}", timeout=timeout)


def type_turn(pg, text):
    pg.evaluate("document.querySelector('#btn-keyboard').offsetParent && document.body.dataset.input !== 'open' && document.querySelector('#btn-keyboard').click()")
    pg.fill("#text-input", text)
    pg.keyboard.press("Enter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8799/")
    ap.add_argument("--chromium", default="/usr/lib/chromium/chromium")
    ap.add_argument("--only", default="")
    ap.add_argument("--sizes", default="phone,desktop")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    only = {s for s in args.only.split(",") if s}

    def want(name):
        return not only or name in only

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=args.chromium, headless=True, args=CHROME_ARGS, env={"TMPDIR": "/tmp", "HOME": "/tmp", "PATH": "/usr/bin:/bin"}
        )
        for size in args.sizes.split(","):
            w, h, dsf = SIZES[size]
            phone = size == "phone"

            def shot(name, query="", prepare=None, scheme="dark", locale="pl-PL"):
                if not want(name):
                    return
                ctx = browser.new_context(
                    viewport={"width": w, "height": h},
                    device_scale_factor=dsf,
                    permissions=["microphone"],
                    locale=locale,
                    color_scheme=scheme,
                    is_mobile=phone,
                    has_touch=phone,
                )
                pg = ctx.new_page()
                errors = []
                pg.on("pageerror", lambda e: errors.append(str(e)))
                pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                pg.goto(args.base + query)
                try:
                    if prepare:
                        prepare(pg)
                    pg.wait_for_timeout(450)
                    pg.screenshot(path=os.path.join(args.out, f"{size}-{name}.png"))
                    print(f"{size}-{name}: ok{'  CONSOLE: ' + '; '.join(errors) if errors else ''}")
                except Exception as e:  # noqa: BLE001
                    print(f"{size}-{name}: FAILED {e}")
                ctx.close()

            shot("idle", prepare=lambda pg: pg.wait_for_timeout(1800))
            shot("idle-light", prepare=lambda pg: pg.wait_for_timeout(3000), scheme="light")

            def listening(pg):
                pg.click("#btn-mic")
                state(pg, "listening")
                pg.wait_for_timeout(900)

            shot("listening", prepare=listening)
            shot("user-speaking", "?demo=user", prepare=lambda pg: (state(pg, "user"), pg.wait_for_timeout(2600)))
            shot("thinking", "?demo=thinking", prepare=lambda pg: (state(pg, "thinking"), pg.wait_for_timeout(3000)))

            def speaking(pg):
                state(pg, "speaking")
                pg.wait_for_function("document.querySelector('.sent[data-state=playing]')")
                pg.wait_for_timeout(1100)

            shot("speaking-happy", "?demo=speak&emotion=happy&pace=realtime", prepare=speaking)
            shot("speaking-sad", "?demo=speak&emotion=sad&pace=realtime", prepare=speaking)
            shot("speaking-excited", "?demo=speak&emotion=excited&pace=realtime", prepare=speaking)
            shot("speaking-light", "?demo=speak&emotion=calm&pace=realtime", prepare=speaking, scheme="light")

            def hermes(pg):
                pg.wait_for_selector(".hcard[data-state='working']", timeout=15000)
                pg.wait_for_timeout(2600)

            shot("hermes-working", "?demo=hermes&hermes_s=40&pace=realtime", prepare=hermes)

            def hermes_open(pg):
                hermes(pg)
                pg.click(".hcard-head")
                pg.wait_for_timeout(600)

            shot("hermes-expanded", "?demo=hermes&hermes_s=40&pace=realtime", prepare=hermes_open)

            def history(pg):
                pg.wait_for_timeout(500)
                type_turn(pg, "Jak się masz?")
                played(pg, 2)
                pg.wait_for_timeout(1800)
                type_turn(pg, "Opowiedz mi coś ciekawego")
                played(pg, 4)
                pg.wait_for_timeout(600)

            shot("history", "?metrics=1&pace=realtime", prepare=history)
            shot("history-light", "?pace=realtime", prepare=history, scheme="light")

            def settings(pg):
                pg.wait_for_timeout(500)
                pg.click("#btn-settings")
                pg.wait_for_timeout(800)

            shot("settings", "?metrics=1", prepare=settings)
            shot("settings-light", "", prepare=settings, scheme="light")
            shot("settings-no-engines", "?engines=none", prepare=settings)
            shot("error", "?demo=error", prepare=lambda pg: (state(pg, "error"), pg.wait_for_timeout(900)))
            shot("insecure", "?insecure=1", prepare=lambda pg: (pg.wait_for_selector("#notice:not([hidden])"), pg.wait_for_timeout(800)))
            shot("mic-denied", "?micdenied=1", prepare=lambda pg: (pg.click("#btn-mic"), pg.wait_for_selector("#notice:not([hidden])"), pg.wait_for_timeout(800)))
            shot("offline", "?demo=drop&down=60", prepare=lambda pg: (state(pg, "offline", 20000), pg.wait_for_timeout(1500)))

            try:  # the `drop` scenario above leaves the mock refusing connections for a while
                import urllib.request
                urllib.request.urlopen(args.base + "reset", timeout=2).read()
            except Exception:  # noqa: BLE001
                pass

            def typed(pg):
                pg.wait_for_timeout(600)
                type_turn(pg, "Jak się masz?")
                played(pg, 1, 15000)
                pg.wait_for_timeout(500)

            shot("typed-metrics", "?metrics=1&pace=realtime", prepare=typed)

            def english(pg):
                pg.wait_for_timeout(500)
                type_turn(pg, "How are you today?")
                played(pg, 1, 15000)
                pg.wait_for_timeout(500)

            shot("english", "?lang=en&pace=realtime", prepare=english, locale="en-GB")

        browser.close()
    try:  # a `drop` scenario leaves the mock refusing connections for a while
        import urllib.request
        urllib.request.urlopen(args.base + "reset", timeout=2).read()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
