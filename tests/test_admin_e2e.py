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
    for sec in ("domownicy", "pamiec", "minutniki", "slownik", "urzadzenia", "diagnostyka", "ustawienia"):
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


def test_household_section_lists_edits_and_saves(page, server):
    login(page)
    page.wait_for_selector(".a-card")
    page.click('.a-nav-link[data-id="domownicy"]')
    page.wait_for_selector(".a-person")
    assert not page.errors
    # Two seeded people; the sensitive profile is visibly marked on every person card.
    assert page.locator(".a-person").count() == 2
    sensitive = page.locator(".a-profile-check.is-sensitive", has_text="Finanse")
    assert sensitive.count() == 2 and sensitive.first.locator(".a-badge", has_text="czuły").count() == 1
    # Ola speaks with Darkman; the voice list comes from /api/admin/tts/voices.
    voice = page.locator('select[aria-label="Głos: Ola"]')
    assert voice.input_value() == "piper-darkman"
    assert voice.locator("option[value='supertonic']").count() == 1
    # General voice and profile block renders.
    assert page.locator('select[aria-label="Głos odpowiedzi"]').input_value() == "supertonic"
    # Rename Tymek, save, and see the change survive a reload.
    name = page.locator('.a-person-head input').first
    name.fill("Tymek K")
    page.click(".a-savebar button >> text=Zapisz")
    page.wait_for_selector(".a-toast")
    page.reload()
    page.click('.a-nav-link[data-id="domownicy"]')
    page.wait_for_selector(".a-person")
    assert page.locator('.a-person-head input').first.input_value() == "Tymek K"
    import json
    with open(os.path.join(server, "household.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["people"][0]["name"] == "Tymek K"
    assert saved["note"]  # fields the panel does not edit pass through untouched


def test_default_profile_change_redraws_and_leaves_extras_alone(page, server):
    import json

    login(page)
    page.wait_for_selector(".a-card")
    page.click('.a-nav-link[data-id="domownicy"]')
    page.wait_for_selector(".a-person")
    sel = page.locator(".a-person").first.locator('select[aria-label^="Profil domyślny"]')
    finanse = page.locator(".a-person").first.locator(".a-profile-check", has_text="Finanse")
    reqs = []
    page.on("request", lambda r: reqs.append(r.post_data) if r.method == "PUT" else None)

    sel.select_option("finanse")
    # The chips redraw with the change: the new default is checked and locked...
    assert finanse.locator("input").is_checked()
    assert finanse.locator("input").is_disabled()

    sel.select_option("default")
    # ...and switching back unticks it: it was only ever the default, never an extra.
    assert not finanse.locator("input").is_checked()
    assert not finanse.locator("input").is_disabled()
    # The round trip changed nothing, so there is nothing to save.
    assert page.locator(".a-savebar").is_hidden()

    # A default that stays switched saves without the default leaking into the extras.
    sel.select_option("dom")
    page.click(".a-savebar button >> text=Zapisz")
    page.wait_for_selector(".a-toast")
    tymek = json.loads(reqs[-1])["people"][0]
    assert tymek["default_profile"] == "dom"
    assert tymek["profiles"] == ["serzant"]  # neither 'finanse' nor 'dom' ever joined the extras
    assert not page.errors


def test_general_profile_offers_no_sensitive_profile(page):
    login(page)
    page.wait_for_selector(".a-card")
    page.click('.a-nav-link[data-id="domownicy"]')
    page.wait_for_selector(".a-person")
    general = page.locator('select[aria-label="Profil ogólny"]')
    assert general.locator("option[value='default']").count() == 1
    # Unknown voices use the general profile: a sensitive one is never on offer.
    assert general.locator("option[value='finanse']").count() == 0
    # A person's own default may still be sensitive (typed text from their phone reaches it).
    assert page.locator('select[aria-label="Profil domyślny: Ola"] option[value="finanse"]').count() == 1
    assert not page.errors


def _hold_household_puts(page):
    """Answer household PUTs in the browser (400, nothing saved) and return the sent documents."""
    import json

    sent = []

    def handle(route):
        if route.request.method != "PUT":
            return route.continue_()
        sent.append(json.loads(route.request.post_data))
        route.fulfill(status=400, content_type="application/json", body='{"detail": "test: nie zapisano"}')

    page.route("**/api/admin/household", handle)
    return sent


def test_new_person_id_follows_the_name_until_typed_by_hand(page):
    login(page)
    page.wait_for_selector(".a-card")
    page.click('.a-nav-link[data-id="domownicy"]')
    page.wait_for_selector(".a-person")
    sent = _hold_household_puts(page)
    page.click("button:has-text('Dodaj osobę')")
    card = page.locator(".a-person.is-new")
    name, ident = card.locator(".a-person-head input"), card.locator('input[aria-label^="Identyfikator"]')
    name.fill("Ala Nowak")
    assert ident.input_value() == "ala-nowak"  # the id field shows the id the name produces
    ident.fill("ala2")
    name.fill("Ala Nowakowska")  # a hand-typed id is no longer overwritten by the name
    assert ident.input_value() == "ala2"
    page.click(".a-savebar button >> text=Zapisz")
    page.wait_for_selector(".a-banner")
    assert sent[-1]["people"][-1]["id"] == "ala2"
    assert sent[-1]["people"][-1]["name"] == "Ala Nowakowska"
    assert not page.errors


def test_new_person_can_be_bound_and_bindings_follow_an_id_change(page):
    login(page)
    page.wait_for_selector(".a-card")
    page.click('.a-nav-link[data-id="domownicy"]')
    page.wait_for_selector(".a-person")
    sent = _hold_household_puts(page)
    phone = page.locator('select[aria-label="Sparowane telefony: phone-dev-2"]')
    page.click("button:has-text('Dodaj osobę')")
    # A person without a valid id yet is not offered (its option would bind to '').
    assert phone.locator("option").count() == 3  # nieprzypisane, Tymek, Ola
    card = page.locator(".a-person.is-new")
    card.locator(".a-person-head input").fill("Ala")
    assert phone.locator("option[value='ala']").inner_text() == "Ala"  # the selects redraw with the id
    phone.select_option("ala")
    card = page.locator(".a-person.is-new")
    assert card.locator(".a-row-title", has_text="Telefon bez osoby").count() == 1
    card.locator('input[aria-label^="Identyfikator"]').fill("ala2")
    page.click(".a-savebar button >> text=Zapisz")
    page.wait_for_selector(".a-banner")
    assert sent[-1]["clients"]["phone-dev-2"]["person"] == "ala2"  # the binding followed the new id
    assert not page.errors


def test_refresh_asks_before_dropping_unsaved_edits(page):
    login(page)
    page.wait_for_selector(".a-card")
    page.click('.a-nav-link[data-id="domownicy"]')
    page.wait_for_selector(".a-person")
    name = page.locator(".a-person-head input").first
    saved = name.input_value()
    name.fill("Ktoś inny")
    page.click("button:has-text('Odśwież')")
    page.wait_for_selector("dialog.a-dialog[open]")
    page.click("dialog.a-dialog .btn >> text=Anuluj")
    assert page.locator(".a-person-head input").first.input_value() == "Ktoś inny"  # the edit survives
    page.click("button:has-text('Odśwież')")
    page.click("dialog.a-dialog .btn >> text=Odrzuć i odśwież")
    page.wait_for_selector(".a-savebar", state="hidden")
    assert page.locator(".a-person-head input").first.input_value() == saved
    assert not page.errors


def test_person_cards_are_separate_and_counts_are_spaced(page):
    login(page)
    page.wait_for_selector(".a-card")
    page.click('.a-nav-link[data-id="domownicy"]')
    page.wait_for_selector(".a-person")
    first, second = (page.locator(".a-person").nth(i).bounding_box() for i in (0, 1))
    assert second["y"] >= first["y"] + first["height"] + 8  # a gap, not touching doubled borders
    chip = page.locator(".a-person .a-chip").first.bounding_box()
    assert chip["height"] < 40 and chip["width"] > chip["height"]  # a pill, not a 73x72 circle
    assert page.locator("h2.a-h2", has_text="Telefony bez osoby").inner_text().endswith(" 1")
    assert not page.errors


def test_voice_picker_marks_partial_languages_and_unloaded_voices(page):
    login(page)
    page.wait_for_selector(".a-card")
    page.click('.a-nav-link[data-id="domownicy"]')
    page.wait_for_selector(".a-person")
    ola = page.locator('select[aria-label="Głos: Ola"]')
    # Ola has no language of her own, so her voice must cover every language Gaja answers in.
    assert "tylko EN" in ola.locator("option[value='paradee']").inner_text()
    assert "tylko PL" in ola.locator("option[value='piper-darkman']").inner_text()
    assert "tylko" not in ola.locator("option[value='supertonic']").inner_text()
    assert page.locator('select[aria-label="Głos odpowiedzi"] option[value="paradee"]').inner_text().endswith("tylko EN")
    # The dev server never builds an engine: a picked voice says its first reply will be slow.
    ola_card = page.locator(".a-person").nth(1)
    assert "pierwsza odpowiedź" in ola_card.locator(".a-voice-hint").inner_text()
    assert "Tylko PL:" in ola_card.locator(".a-voice-hint").inner_text()  # the closed select truncates the mark
    tymek_card = page.locator(".a-person").first
    assert tymek_card.locator(".a-voice-hint").is_hidden()  # "domyślny (według języka)": no hint
    tymek_card.locator('select[aria-label^="Głos:"]').select_option("supertonic")
    assert tymek_card.locator(".a-voice-hint").is_visible()
    assert not page.errors


def test_household_section_blocks_an_invalid_new_person(page):
    login(page)
    page.wait_for_selector(".a-card")
    page.click('.a-nav-link[data-id="domownicy"]')
    page.wait_for_selector(".a-person")
    page.click("button:has-text('Dodaj osobę')")
    page.wait_for_selector(".a-person.is-new")
    page.click(".a-savebar button >> text=Zapisz")
    page.wait_for_selector(".a-banner")  # inline server-style error, nothing saved
    assert "imię" in page.locator(".a-banner").inner_text()
    page.click(".a-savebar button >> text=Odrzuć")
    page.wait_for_selector(".a-savebar", state="hidden")
    assert page.locator(".a-person.is-new").count() == 0
    assert not page.errors


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
