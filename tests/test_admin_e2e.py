"""Browser end-to-end tests of the admin panel (web/admin.html), Playwright + system Chromium.

Starts scripts/admin_dev_server.py (model-free, temp config copies, never touches the real service) on
127.0.0.1:8797 with a throw-away data directory.

Run:  TMPDIR=/tmp .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_admin_e2e.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import pytest
from playwright.sync_api import sync_playwright

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PORT = 8797
BASE = f"http://127.0.0.1:{PORT}"
PASSWORD = "e2e-password-123"


def chromium_path():
    for c in ("/usr/lib/chromium/chromium", shutil.which("chromium"), shutil.which("google-chrome"), shutil.which("chromium-browser")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


@pytest.fixture(scope="module")
def server():
    data = tempfile.mkdtemp(prefix="gaja-admin-e2e-")
    env = {k: v for k, v in os.environ.items() if k != "INVOCATION_ID"}
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "scripts", "admin_dev_server.py"), "--port", str(PORT), "--password", PASSWORD, "--data-dir", data],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail("admin dev server exited early")
            try:
                urllib.request.urlopen(BASE + "/health", timeout=1).read()
                break
            except OSError:
                time.sleep(0.3)
        else:
            pytest.fail("admin dev server did not start")
        yield data
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(data, ignore_errors=True)


@pytest.fixture(scope="module")
def browser():
    exe = chromium_path()
    assert exe, "Chromium executable not found"
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        yield b
        b.close()


@pytest.fixture
def page(browser, server):
    ctx = browser.new_context(viewport={"width": 1280, "height": 860}, color_scheme="dark")
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    yield pg
    ctx.close()


def login(page, password=PASSWORD):
    page.goto(BASE + "/admin")
    page.fill("#a-password", password)
    page.click(".a-login-btn")


def field(page, path):
    return page.locator(f'.a-field[data-path="{path}"]')


def local_toml(data):
    with open(os.path.join(data, "config.local.toml"), encoding="utf-8") as f:
        return f.read()


def test_login_error_then_success(page):
    login(page, "wrong-password")
    page.wait_for_selector(".a-login-error, [role=alert]")
    assert page.locator("#a-password").is_visible()
    page.fill("#a-password", PASSWORD)
    page.click(".a-login-btn")
    page.wait_for_selector(".a-card")
    assert not page.errors


def test_every_section_renders(page):
    login(page)
    page.wait_for_selector(".a-card")
    for sec in ("pamiec", "minutniki", "slownik", "urzadzenia", "diagnostyka", "ustawienia"):
        page.click(f'.a-nav-link[data-id="{sec}"]')
        page.wait_for_timeout(500)
        assert page.locator("h1").first.inner_text()
    assert not page.errors


def test_edit_save_and_reset_keeps_comments(page, server):
    login(page)
    page.wait_for_selector(".a-card")
    page.fill(".a-search-input", "llm.max_tokens")
    f = field(page, "llm.max_tokens")
    f.locator("input").fill("111")
    page.click(".a-savebar button >> text=Zapisz")
    page.wait_for_selector(".a-banner")
    text = local_toml(server)
    assert "max_tokens = 111" in text
    assert "# a little warmer than the default" in text  # seeded comment survives the write
    # reset to default removes the override
    f.locator("button >> text=przywróć domyślne").click()
    page.click(".a-savebar button >> text=Zapisz")
    page.wait_for_function("document.querySelector('.a-savebar') === null || !document.querySelector('.a-savebar').offsetParent")
    assert "max_tokens" not in local_toml(server)
    assert "# a little warmer than the default" in local_toml(server)


def test_restart_conflict_is_explained(page):
    login(page)
    page.wait_for_selector(".a-card")
    page.fill(".a-search-input", "llm.max_tokens")
    field(page, "llm.max_tokens").locator("input").fill("222")
    page.click(".a-savebar button >> text=Zapisz")
    page.wait_for_selector(".a-banner")
    page.locator(".a-banner .btn").click()
    page.wait_for_selector("dialog.a-dialog[open]")
    page.click("dialog.a-dialog .btn >> text=Zrestartuj")
    page.wait_for_selector("text=systemd", timeout=5000)  # 409: not running under systemd
    assert page.locator(".a-overlay").count() == 0


def test_phone_width_has_no_horizontal_scroll(browser, server):
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    pg = ctx.new_page()
    try:
        login(pg)
        pg.wait_for_selector(".a-card")
        assert not pg.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
        pg.click(".a-menu-btn")
        pg.click('.a-nav-link[data-id="diagnostyka"]')
        pg.wait_for_selector(".a-status")
        assert not pg.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
    finally:
        ctx.close()
