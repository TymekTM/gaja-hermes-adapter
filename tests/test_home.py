"""Unit tests for the local Home Assistant tool (httpx.MockTransport with house snapshot fixtures)."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from gaja.config import Settings
from gaja.tools.base import current_session_context
from gaja.tools.home_tool import (
    clamp_temperature,
    home,
    home_configured,
    is_write_allowed,
    resolve_device,
)

FAKE_TOKEN = "ha-super-secret-token-never-log"

# Generic fixture household (no real entity ids; NEW-6) served by the home_cfg fixture below.
SNAPSHOT_STATES: dict[str, dict[str, Any]] = {
    "climate.ania": {
        "entity_id": "climate.ania",
        "state": "off",
        "attributes": {
            "hvac_modes": ["fan_only", "dry", "cool", "heat", "heat_cool", "off"],
            "min_temp": 7,
            "max_temp": 35,
            "target_temp_step": 1,
            "current_temperature": 25.0,
            "temperature": None,
            "friendly_name": "Klima — Pokój Ani",
        },
    },
    "climate.sypialnia": {
        "entity_id": "climate.sypialnia",
        "state": "off",
        "attributes": {
            "hvac_modes": ["fan_only", "dry", "cool", "heat", "heat_cool", "off"],
            "min_temp": 7,
            "max_temp": 35,
            "target_temp_step": 1,
            "current_temperature": 25.0,
            "temperature": None,
            "friendly_name": "Klima — Sypialnia Rodziców",
        },
    },
    "climate.bartek": {
        "entity_id": "climate.bartek",
        "state": "fan_only",
        "attributes": {
            "hvac_modes": ["fan_only", "dry", "cool", "heat", "heat_cool", "off"],
            "min_temp": 7,
            "max_temp": 35,
            "target_temp_step": 1,
            "current_temperature": 22.0,
            "temperature": None,
            "friendly_name": "Klima — Pokój Bartka",
        },
    },
    "sensor.bartek_humidity": {
        "entity_id": "sensor.bartek_humidity",
        "state": "55.0",
        "attributes": {
            "unit_of_measurement": "%",
            "device_class": "humidity",
            "friendly_name": "Klima — Pokój Bartka Wilgotność",
        },
    },
    "media_player.tv_salon": {
        "entity_id": "media_player.tv_salon",
        "state": "unavailable",
        "attributes": {"friendly_name": 'TV Salon 75"'},
    },
    "media_player.tv_bartek": {
        "entity_id": "media_player.tv_bartek",
        "state": "off",
        "attributes": {
            "friendly_name": 'TV Bartka 55"',
            "entity_picture": "/api/media_player_proxy/media_player.tv_bartek?token=secret-pic-tok",
            "access_token": "secret-entity-access-token",
        },
    },
    "media_player.amplituner": {
        "entity_id": "media_player.amplituner",
        "state": "on",
        "attributes": {"friendly_name": "TX-NR6100", "volume_level": 0.35, "is_volume_muted": False},
    },
    "climate.boiler_heating": {
        "entity_id": "climate.boiler_heating",
        "state": "auto",
        "attributes": {
            "hvac_modes": ["off", "auto"],
            "min_temp": 3,
            "max_temp": 60,
            "target_temp_step": 1,
            "current_temperature": 29.8,
            "temperature": 20,
            "friendly_name": "Kocioł (piwnica) Heating",
        },
    },
    "button.boiler_charge": {
        "entity_id": "button.boiler_charge",
        "state": "unknown",
        "attributes": {"friendly_name": "Kocioł (piwnica) Activate one-time charge"},
    },
    "number.boiler_dhw_temperature": {
        "entity_id": "number.boiler_dhw_temperature",
        "state": "55",
        "attributes": {"min": 10, "max": 60, "step": 1, "unit_of_measurement": "°C"},
    },
    "select.boiler_dhw_mode": {
        "entity_id": "select.boiler_dhw_mode",
        "state": "balanced",
        "attributes": {"options": ["balanced", "off"]},
    },
}


class FakeHomeAssistant:
    def __init__(self, *, ignore_writes: bool = False) -> None:
        self.states = copy.deepcopy(SNAPSHOT_STATES)
        self.service_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.get_counts: dict[str, int] = {}
        self.ignore_writes = ignore_writes

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == f"Bearer {FAKE_TOKEN}"
        path = request.url.path
        if request.method == "GET" and path.startswith("/api/states/"):
            eid = path.removeprefix("/api/states/")
            self.get_counts[eid] = self.get_counts.get(eid, 0) + 1
            if eid not in self.states:
                return httpx.Response(404, json={"message": "Entity not found."})
            return httpx.Response(200, json=self.states[eid])

        if request.method == "POST" and path.startswith("/api/services/"):
            rest = path.removeprefix("/api/services/")
            domain, service = rest.split("/", 1)
            payload = json.loads(request.content.decode("utf-8"))
            self.service_calls.append((domain, service, payload))
            eid = payload["entity_id"]
            if not self.ignore_writes and eid in self.states:
                st = self.states[eid]
                if domain == "climate":
                    if service == "turn_on":
                        if st["state"] == "off":
                            st["state"] = "cool"
                        if st["attributes"].get("temperature") is None:
                            st["attributes"]["temperature"] = 22
                    elif service == "turn_off":
                        st["state"] = "off"
                    elif service == "set_hvac_mode":
                        st["state"] = payload["hvac_mode"]
                    elif service == "set_temperature":
                        st["attributes"]["temperature"] = payload["temperature"]
                elif domain == "media_player":
                    if service == "turn_on":
                        if st["state"] != "unavailable":
                            st["state"] = "on"
                    elif service == "turn_off":
                        if st["state"] != "unavailable":
                            st["state"] = "off"
                    elif service == "volume_set":
                        st["attributes"]["volume_level"] = payload["volume_level"]
                    elif service == "volume_up":
                        st["attributes"]["volume_level"] = round(float(st["attributes"].get("volume_level", 0.3)) + 0.05, 2)
                    elif service == "volume_down":
                        st["attributes"]["volume_level"] = round(float(st["attributes"].get("volume_level", 0.3)) - 0.05, 2)
                    elif service == "volume_mute":
                        st["attributes"]["is_volume_muted"] = payload["is_volume_muted"]
            return httpx.Response(200, json=[])

        raise AssertionError(f"unexpected request {request.method} {path}")



# The generic test household: written to a tmp household file (the [home] household_file mechanism,
# NEW-6) so every test exercises the real loader. No real names or entity ids live in the repo.
HOUSEHOLD_TOML = """
write_allowlist = [
  "climate.ania",
  "climate.sypialnia",
  "climate.bartek",
  "media_player.*",
]
boiler_id_substrings = ["boiler"]

[aliases]
"klima u Ani" = "climate.ania"
"pokój Ani" = "climate.ania"
"u Ani" = "climate.ania"
"Ania" = "climate.ania"
"klima u Bartka" = "climate.bartek"
"pokój Bartka" = "climate.bartek"
"u Bartka" = "climate.bartek"
"Bartek" = "climate.bartek"
"klima w sypialni" = "climate.sypialnia"
"sypialnia rodziców" = "climate.sypialnia"
"sypialnia" = "climate.sypialnia"
"u rodziców" = "climate.sypialnia"
"rodzice" = "climate.sypialnia"
"telewizor w salonie" = "media_player.tv_salon"
"tv w salonie" = "media_player.tv_salon"
"tv salon" = "media_player.tv_salon"
"telewizor u Bartka" = "media_player.tv_bartek"
"tv u Bartka" = "media_player.tv_bartek"
"tv Bartka" = "media_player.tv_bartek"
"amplituner" = "media_player.amplituner"
"wzmacniacz" = "media_player.amplituner"
"amplituner strefa 2" = "media_player.amplituner_strefa2"
"głośnik w salonie" = "media_player.glosnik_salon"
"piec" = "climate.boiler_heating"
"kocioł" = "climate.boiler_heating"
"ogrzewanie domu" = "climate.boiler_heating"
"ciepła woda" = "water_heater.boiler_dhw"
"bojler" = "water_heater.boiler_dhw"
"zasobnik" = "water_heater.boiler_dhw"
"palnik" = "binary_sensor.boiler_burner"
"ciśnienie w instalacji" = "sensor.boiler_supply_pressure"
"wilgotność u Ani" = "sensor.ania_humidity"
"wilgotność w sypialni" = "sensor.sypialnia_humidity"
"wilgotność u Bartka" = "sensor.bartek_humidity"
"drukarka" = "sensor.biuro_drukarka"
"konsola" = "binary_sensor.konsola"
"ps5" = "binary_sensor.konsola"
"playstation" = "binary_sensor.konsola"
"qnap" = "binary_sensor.nas"

[spoken_forms]
"climate.ania" = ["klima u Ani", "klimę u Ani", "klimy u Ani"]
"climate.sypialnia" = ["klima w sypialni", "klimę w sypialni", "klimy w sypialni"]
"climate.bartek" = ["klima u Bartka", "klimę u Bartka", "klimy u Bartka"]

"""


@pytest.fixture
def home_cfg(tmp_path: Path) -> Settings:
    """Settings with the generic test household loaded from a real household file (NEW-6)."""
    household = tmp_path / "home.toml"
    household.write_text(HOUSEHOLD_TOML, encoding="utf-8")
    return Settings(home={"household_file": household})


@pytest.fixture
def ha_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HASS_URL", "http://127.0.0.1:8123")
    monkeypatch.setenv("HASS_TOKEN", FAKE_TOKEN)


@pytest.mark.asyncio
async def test_disabled_without_hass_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HASS_URL", raising=False)
    monkeypatch.delenv("HASS_TOKEN", raising=False)
    assert home_configured() is False

    res = await home(action="status", device="klima u Ani", settings=Settings())
    assert res.error == "disabled"
    assert res.handoff is True
    assert res.speak == ""


@pytest.mark.parametrize(
    ("phrase", "utterance", "expected_eid"),
    [
        ("klima u Ani", "", "climate.ania"),
        ("Ania", "", "climate.ania"),
        ("u Ani", "jaka jest temperatura u Ani", "climate.ania"),
        ("w pokoju Ani", "", "climate.ania"),
        ("Bartek", "jaka temperatura u Bartka", "climate.bartek"),
        ("u Bartka", "", "climate.bartek"),
        ("w sypialni", "czy klima w sypialni chodzi", "climate.sypialnia"),
        ("sypialnia", "", "climate.sypialnia"),
        ("u rodziców", "włącz chłodzenie u rodziców", "climate.sypialnia"),
        ("rodzice", "", "climate.sypialnia"),
        ("telewizor w salonie", "", "media_player.tv_salon"),
        ("tv w salonie", "", "media_player.tv_salon"),
        ("telewizor u Bartka", "", "media_player.tv_bartek"),
        ("amplitunera", "zmień głośność amplitunera", "media_player.amplituner"),
        ("piec", "", "climate.boiler_heating"),
        ("stan kotła", "", "climate.boiler_heating"),
        ("Bartka", "jaka jest wilgotność u Bartka", "sensor.bartek_humidity"),
    ],
)
def test_alias_resolution_with_polish_inflection(
    phrase: str, utterance: str, expected_eid: str, home_cfg: Settings
) -> None:
    aliases = home_cfg.home.aliases
    res = resolve_device(phrase, aliases, utterance=utterance)
    assert res.error is None
    assert res.entity_id == expected_eid


@pytest.mark.asyncio
async def test_ambiguous_write_never_falls_back_to_the_llm_device_guess(ha_env: None, home_cfg: Settings) -> None:
    """NEW-3: for a write action an ambiguous utterance must ask back, never execute the LLM's
    guessed device argument. 'włącz klimę' with device='klima w sypialni' (or a raw climate.x)
    used to resolve through the LLM argument and switch an AC without asking."""
    called = False

    def fail_if_called(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        nonlocal called
        called = True
        return httpx.Response(500)

    transport = httpx.MockTransport(fail_if_called)
    cfg = Settings(
        home={
            "aliases": {
                "klima w salonie": "climate.salon",
                "klima w sypialni": "climate.sypialnia",
                "klima u Ani": "climate.ania",
            }
        }
    )
    tok = current_session_context.set({"utterance": "włącz klimę"})
    try:
        for guessed in ("klima w sypialni", "climate.salon"):
            res = await home(action="turn_on", device=guessed, settings=cfg, transport=transport)
            assert res.error == "ambiguous_device", guessed
            assert res.speak.endswith("?")
        assert not called
    finally:
        current_session_context.reset(tok)


def test_resolve_device_write_ambiguity_rules() -> None:
    aliases = {
        "klima w salonie": "climate.salon",
        "klima w sypialni": "climate.sypialnia",
        "klima u Ani": "climate.ania",
    }
    ambiguous = resolve_device("klima w sypialni", aliases, utterance="włącz klimę", action="turn_on")
    assert ambiguous.error == "ambiguous_device" and ambiguous.entity_id is None
    # The utterance naming one device still wins for writes.
    named = resolve_device("klima w sypialni", aliases, utterance="włącz klimę w salonie", action="turn_on")
    assert named.entity_id == "climate.salon"
    # Status reads keep the current behaviour: the specific LLM argument resolves.
    read = resolve_device("klima w sypialni", aliases, utterance="jaka temperatura z klimy", action="status")
    assert read.entity_id == "climate.sypialnia"


# H1: a write may execute only when the utterance itself names the target entity. Every row below
# was executed against a 3-AC household by the round-3 review: extra tokens ("na 22 stopnie",
# "w kuchni") turned the utterance into no_match and the LLM's device guess (or a raw entity_id)
# switched an AC on. All must ask back and leave Home Assistant untouched.
_H1_ASK_BACK_CASES = [
    # (utterance, home() kwargs, expected error, speak fragment)
    ("ustaw klimę na 22 stopnie", dict(action="set_temperature", device="sypialnia", temperature=22), "ambiguous_device", "Które urządzenie"),
    ("ustaw klimę na 22 stopnie", dict(action="set_temperature", device="climate.ania", temperature=22), "ambiguous_device", "Które urządzenie"),
    ("włącz klimę na chłodzenie", dict(action="set_hvac_mode", device="climate.ania", hvac_mode="cool"), "ambiguous_device", "Które urządzenie"),
    ("wyłącz klimę na noc", dict(action="turn_off", device="climate.ania"), "ambiguous_device", "Które urządzenie"),
    # A TV noun must never resolve to (or switch) a climate entity.
    ("włącz telewizor w kuchni", dict(action="turn_on", device="sypialnia"), "ambiguous_device", "Które urządzenie"),
    ("włącz telewizor w kuchni", dict(action="turn_on", device="climate.ania"), "ambiguous_device", "Które urządzenie"),
    # No salon alias exists in this household: the named place is unknown, several ACs exist.
    ("włącz klimę w salonie", dict(action="turn_on", device="climate.ania"), "ambiguous_device", "Które urządzenie"),
    ("wyłącz to", dict(action="turn_off", device="climate.ania"), "ambiguous_device", "Które urządzenie"),
    ("wyłącz", dict(action="turn_off", device="climate.ania"), "ambiguous_device", "Które urządzenie"),
    # An unknown named device asks back differently and hands the turn to Hermes.
    ("włącz zmywarkę w kuchni", dict(action="turn_on", device="klima u Ani"), "no_match", "Nie znam takiego urządzenia"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("utterance", "kwargs", "expected_error", "speak_fragment"),
    _H1_ASK_BACK_CASES,
    ids=[c[0] + " | " + str(c[1].get("action")) for c in _H1_ASK_BACK_CASES],
)
async def test_write_without_a_named_device_asks_back_and_never_executes(
    utterance: str,
    kwargs: dict[str, Any],
    expected_error: str,
    speak_fragment: str,
    ha_env: None,
    home_cfg: Settings,
) -> None:
    """H1 (write safety): when the utterance alone does not resolve to one entity, a write asks
    back instead of executing the LLM's ``device`` guess or a raw entity_id."""
    fake_ha = FakeHomeAssistant()
    transport = httpx.MockTransport(fake_ha.handler)
    tok = current_session_context.set({"utterance": utterance})
    try:
        res = await home(settings=home_cfg, transport=transport, verify_delay_s=0.0, **kwargs)
    finally:
        current_session_context.reset(tok)
    assert res.error == expected_error, (utterance, kwargs, res.speak)
    assert speak_fragment in res.speak
    # Ask-backs end with a question; the unknown-device refusal is a statement plus a handoff.
    assert res.speak.endswith("?") or (expected_error == "no_match" and res.handoff is True)
    assert fake_ha.service_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("utterance", "kwargs", "expected_entity", "expected_service"),
    [
        # The utterance names the entity: it wins over a wrong LLM device guess and executes.
        ("włącz klimę w sypialni", dict(action="turn_on", device="climate.ania"), "climate.sypialnia", "climate.turn_on"),
        ("ustaw klimę u Ani na 22 stopnie", dict(action="set_temperature", device="sypialnia", temperature=22), "climate.ania", "climate.set_temperature"),
        ("ścisz amplituner", dict(action="volume_down", device="amplituner"), "media_player.amplituner", "media_player.volume_down"),
    ],
)
async def test_write_still_executes_when_the_utterance_names_the_entity(
    utterance: str,
    kwargs: dict[str, Any],
    expected_entity: str,
    expected_service: str,
    ha_env: None,
    home_cfg: Settings,
) -> None:
    """H1 positive direction: an alias of the entity present in the utterance (inflection-tolerant
    token matching) is executed even when the LLM device argument disagrees."""
    fake_ha = FakeHomeAssistant()
    fake_ha.states["climate.sypialnia"]["state"] = "cool"  # so turn_on verifies as on
    fake_ha.states["climate.ania"]["state"] = "cool"
    transport = httpx.MockTransport(fake_ha.handler)
    tok = current_session_context.set({"utterance": utterance})
    try:
        res = await home(settings=home_cfg, transport=transport, verify_delay_s=0.0, **kwargs)
    finally:
        current_session_context.reset(tok)
    assert res.error is None, (utterance, res.speak)
    assert res.data["entity_id"] == expected_entity
    assert any(call[0] == expected_service.split(".")[0] and call[1] == expected_service.split(".")[1]
               and call[2].get("entity_id") == expected_entity for call in fake_ha.service_calls)


@pytest.mark.asyncio
async def test_write_without_any_device_noun_executes_only_a_single_candidate_household(
    ha_env: None,
    home_cfg: Settings,
) -> None:
    """H1 exception: an utterance with no device noun at all ("włącz") executes only when exactly
    one entity of the action's domain exists; with several it asks back."""
    fake_ha = FakeHomeAssistant()
    fake_ha.states["climate.ania"]["state"] = "cool"
    transport = httpx.MockTransport(fake_ha.handler)

    single = Settings(home={"aliases": {"klima u Ani": "climate.ania"}, "write_allowlist": ["climate.ania"]})
    tok = current_session_context.set({"utterance": "włącz"})
    try:
        res = await home(action="turn_on", device="climate.ania", settings=single, transport=transport, verify_delay_s=0.0)
    finally:
        current_session_context.reset(tok)
    assert res.error is None
    assert ("climate", "turn_on", {"entity_id": "climate.ania"}) in fake_ha.service_calls

    # Same single-entity household, but the utterance names an unknown place: ask back.
    fake_ha.service_calls.clear()
    tok = current_session_context.set({"utterance": "włącz klimę w salonie"})
    try:
        res_ask = await home(action="turn_on", device="climate.ania", settings=single, transport=transport, verify_delay_s=0.0)
    finally:
        current_session_context.reset(tok)
    assert res_ask.error == "ambiguous_device"
    assert res_ask.speak.endswith("?")
    assert fake_ha.service_calls == []


@pytest.mark.asyncio
async def test_ambiguity_and_no_match_ask_back_without_guessing(ha_env: None, home_cfg: Settings) -> None:
    cfg = home_cfg
    called = False

    def fail_if_called(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        nonlocal called
        called = True
        return httpx.Response(500)

    transport = httpx.MockTransport(fail_if_called)

    amb_ac = await home(action="turn_on", device="klima", settings=cfg, transport=transport)
    assert amb_ac.error == "ambiguous_device"
    assert "Ani" in amb_ac.speak and "Bartka" in amb_ac.speak and "sypialni" in amb_ac.speak
    assert amb_ac.speak.endswith("?")

    amb_tv = await home(action="turn_on", device="telewizor", settings=cfg, transport=transport)
    assert amb_tv.error == "ambiguous_device"
    assert "salonie" in amb_tv.speak and "Bartka" in amb_tv.speak

    unknown = await home(action="status", device="zmywarka w kuchni", settings=cfg, transport=transport)
    assert unknown.error == "no_match"
    assert unknown.handoff is True
    assert not called


@pytest.mark.asyncio
async def test_status_no_match_and_entity_not_found_hand_off_and_boiler_refusal_names_kociol(
    ha_env: None, home_cfg: Settings
) -> None:
    """M8: status questions without a sensor/entity hand off silently to Hermes; boiler writes name 'Kocioł'."""
    cfg = home_cfg
    fake_ha = FakeHomeAssistant()
    transport = httpx.MockTransport(fake_ha.handler)

    tok = current_session_context.set({"utterance": "jaka temperatura w salonie"})
    try:
        salon = await home(action="status", device="w salonie", settings=cfg, transport=transport)
    finally:
        current_session_context.reset(tok)
    assert salon.error == "no_match"
    assert salon.handoff is True
    assert salon.speak == ""

    # Alias resolves to sensor.biuro_drukarka which returns 404 in snapshot -> handoff=True on status
    missing = await home(action="status", device="drukarka", settings=cfg, transport=transport)
    assert missing.error == "entity_not_found"
    assert missing.handoff is True
    assert missing.speak == ""

    # Boiler write stays a local refusal and explicitly names "Kocioł"
    boiler_write = await home(action="set_temperature", device="piec", temperature=22, settings=cfg, transport=transport)
    assert boiler_write.error == "write_not_allowed"
    assert boiler_write.handoff is False
    assert "Kocioł jest tylko do odczytu" in boiler_write.speak


@pytest.mark.asyncio
async def test_allowlist_refuses_boiler_and_sensitive_services(ha_env: None, home_cfg: Settings) -> None:
    cfg = home_cfg
    fake_ha = FakeHomeAssistant()
    transport = httpx.MockTransport(fake_ha.handler)

    # Reading boiler status IS allowed
    read_res = await home(action="status", device="piec", settings=cfg, transport=transport)
    assert read_res.error is None
    assert "Piec" in read_res.speak

    # Writing to boiler is refused by default allowlist
    write_res = await home(action="set_temperature", device="piec", temperature=23, settings=cfg, transport=transport)
    assert write_res.error == "write_not_allowed"
    assert fake_ha.service_calls == []
    assert "Kocioł jest tylko do odczytu" in write_res.speak

    # Wildcard climate.* still does NOT unlock *boiler*, button.press, number.set_value, or select.*
    wildcard_allowlist = ["climate.*", "button.*", "number.*", "select.*", "water_heater.*"]
    boiler_markers = cfg.home.boiler_id_substrings
    assert not is_write_allowed("climate.boiler_heating", "climate", "set_temperature", wildcard_allowlist, boiler_markers)
    assert not is_write_allowed(
        "button.boiler_charge", "button", "press", wildcard_allowlist, boiler_markers
    )
    assert not is_write_allowed(
        "number.boiler_dhw_temperature", "number", "set_value", wildcard_allowlist, boiler_markers
    )
    assert not is_write_allowed(
        "select.boiler_dhw_mode", "select", "select_option", wildcard_allowlist, boiler_markers
    )
    # Explicit exact entity_id DOES unlock when intentionally configured
    assert is_write_allowed(
        "climate.boiler_heating",
        "climate",
        "set_temperature",
        ["climate.boiler_heating"],
        boiler_markers,
    )


def test_wildcard_allowlist_unlocks_only_media_player() -> None:
    """H3/M1d: wildcard patterns may only unlock media_player. climate (and any other domain)
    needs the exact entity_id, so a stray 'climate.*' can never make every AC writable."""
    wildcards = ["climate.*", "switch.*", "media_player.*"]
    assert not is_write_allowed("climate.ania", "climate", "set_temperature", wildcards, [])
    assert not is_write_allowed("switch.garaż", "switch", "turn_on", wildcards, [])
    assert is_write_allowed("media_player.tv_salon", "media_player", "turn_on", wildcards, [])
    # The exact entity_id still unlocks any domain
    assert is_write_allowed("climate.ania", "climate", "set_temperature", ["climate.ania"], [])
    # Code-level sensitive-domain refusals stay
    assert not is_write_allowed("water_heater.boiler_dhw", "water_heater", "set_temperature", wildcards, [])
    assert not is_write_allowed("number.boiler_dhw_temperature", "number", "set_value", wildcards, [])
    assert not is_write_allowed("select.boiler_dhw_mode", "select", "select_option", wildcards, [])
    assert not is_write_allowed("button.boiler_charge", "button", "press", wildcards, [])


def test_boiler_marker_floor_protects_even_without_household_markers() -> None:
    """H3/M1e: generic heating words (boiler, vitodens, viessmann, kocioł/kociol, piec) are a
    code-level floor on top of the household file: with no markers configured at all, a wildcard
    allowlist still cannot unlock them; only an exact entity_id can."""
    no_markers: list[str] = []
    for eid in ("climate.boiler_heating", "climate.vitodens_100", "climate.viessmann_w", "climate.kociol_parter", "climate.piec_kuchenny"):
        assert not is_write_allowed(eid, "climate", "set_temperature", ["climate.*"], no_markers), eid
    assert is_write_allowed(
        "climate.vitodens_100", "climate", "set_temperature", ["climate.vitodens_100"], no_markers
    )
    # Floor round 6 pin: the wildcard domain. media_player.* unlocks ordinary players, but an id
    # carrying only a floor marker ("piec") stays read-only — without BOILER_MARKER_FLOOR the
    # wildcard would unlock media_player.piec_tv for writes (nothing else protects it: no
    # water_heater prefix, no household markers).
    for eid in ("media_player.piec_tv", "media_player.kociol_glosnik", "media_player.vitodens"):
        assert not is_write_allowed(eid, "media_player", "turn_on", ["media_player.*"], no_markers), eid


@pytest.mark.asyncio
async def test_temperature_sanity_range_and_entity_clamping_with_audit_log(
    ha_env: None, home_cfg: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """L4: AC setpoints outside 16-30 °C ask back without writing; entity min/max clamps inside 16-30 °C."""
    cfg = home_cfg
    fake_ha = FakeHomeAssistant()
    transport = httpx.MockTransport(fake_ha.handler)

    assert clamp_temperature(42, {"min_temp": 7, "max_temp": 35, "target_temp_step": 1}) == (35, True)
    assert clamp_temperature(2, {"min_temp": 7, "max_temp": 35, "target_temp_step": 1}) == (7, True)

    # Outside 16-30 °C sanity range -> ask back, zero POST calls
    res_high = await home(action="set_temperature", device="klima u Ani", temperature=42, settings=cfg, transport=transport)
    assert res_high.error == "temperature_out_of_range"
    assert res_high.speak.endswith("?")
    assert fake_ha.service_calls == []

    res_low = await home(action="set_temperature", device="klima u Ani", temperature=14, settings=cfg, transport=transport)
    assert res_low.error == "temperature_out_of_range"
    assert res_low.speak.endswith("?")
    assert fake_ha.service_calls == []

    # Narrow entity max_temp=28 within 16-30 °C -> clamps 29 -> 28 and audits without token
    fake_ha.states["climate.ania"]["attributes"]["min_temp"] = 18
    fake_ha.states["climate.ania"]["attributes"]["max_temp"] = 28

    with caplog.at_level(logging.INFO, logger="gaja.tools.home_tool"):
        res = await home(
            action="set_temperature",
            device="klima u Ani",
            temperature=29,
            settings=cfg,
            transport=transport,
            verify_delay_s=0.0,
        )

    assert res.error is None
    assert res.data["clamped"] is True
    assert res.data["clamped_temperature"] == 28
    assert ("climate", "turn_on", {"entity_id": "climate.ania"}) in fake_ha.service_calls
    assert ("climate", "set_temperature", {"entity_id": "climate.ania", "temperature": 28}) in fake_ha.service_calls
    assert "dwadzieścia osiem stopni" in res.speak

    assert "entity=climate.ania" in caplog.text
    assert "service=climate.set_temperature" in caplog.text
    assert FAKE_TOKEN not in caplog.text

    # Spelled-out Polish number in utterance ("dwadzieścia jeden stopni") recovers 21 °C
    tok = current_session_context.set({"utterance": "włącz klimatyzację w sypialni na dwadzieścia jeden stopni"})
    try:
        res_words = await home(
            action="turn_on",
            device="klima w sypialni",
            settings=cfg,
            transport=transport,
            verify_delay_s=0.0,
        )
    finally:
        current_session_context.reset(tok)
    assert res_words.error is None
    assert ("climate", "set_temperature", {"entity_id": "climate.sypialnia", "temperature": 21}) in fake_ha.service_calls


@pytest.mark.asyncio
async def test_post_write_verify_speaks_real_state_and_catches_unconfirmed_writes(
    ha_env: None, home_cfg: Settings
) -> None:
    cfg = home_cfg
    fake_ha = FakeHomeAssistant()
    transport = httpx.MockTransport(fake_ha.handler)

    # 1. Successful turn_on + re-read confirms 22°C -> feminine Polish confirmation
    res_on = await home(action="turn_on", device="klima u Ani", settings=cfg, transport=transport, verify_delay_s=0.0)
    assert res_on.error is None
    assert "Włączyłam klimę u Ani, dwadzieścia dwa stopnie" in res_on.speak

    # 2. AC auto mode maps to heat_cool
    res_auto = await home(
        action="set_hvac_mode", device="u rodziców", hvac_mode="auto", settings=cfg, transport=transport, verify_delay_s=0.0
    )
    assert res_auto.error is None
    assert ("climate", "set_hvac_mode", {"entity_id": "climate.sypialnia", "hvac_mode": "heat_cool"}) in fake_ha.service_calls
    assert "automatyczny" in res_auto.speak

    # 3. L2: set_hvac_mode off succeeds when verified state is "off"
    res_mode_off = await home(
        action="set_hvac_mode", device="klima u Bartka", hvac_mode="off", settings=cfg, transport=transport, verify_delay_s=0.0
    )
    assert res_mode_off.error is None
    assert res_mode_off.data["state"] == "off"
    assert "Wyłączyłam klimę u Bartka" in res_mode_off.speak

    # 4. Media player volume_set and mute verified on Onkyo
    res_vol = await home(
        action="volume_set", device="amplituner", level=40, settings=cfg, transport=transport, verify_delay_s=0.0
    )
    assert res_vol.error is None
    assert ("media_player", "volume_set", {"entity_id": "media_player.amplituner", "volume_level": 0.4}) in fake_ha.service_calls
    assert "czterdzieści procent" in res_vol.speak

    res_mute = await home(action="mute", device="amplituner", settings=cfg, transport=transport, verify_delay_s=0.0)
    assert res_mute.error is None
    assert "Wyciszyłam amplituner" in res_mute.speak

    # 5. Unavailable entity (TV Salon 75" is unavailable in snapshot): never claim success!
    res_unavail = await home(
        action="turn_on", device="telewizor w salonie", settings=cfg, transport=transport, verify_delay_s=0.0
    )
    assert res_unavail.error == "verify_unavailable"
    assert "niedostępne" in res_unavail.speak
    assert "Włączyłam" not in res_unavail.speak

    # 6. API returns 200 on POST but state stays "off" on re-read -> verify_mismatch, never claim success!
    stuck_ha = FakeHomeAssistant(ignore_writes=True)
    res_stuck = await home(
        action="turn_on",
        device="telewizor u Bartka",
        settings=cfg,
        transport=httpx.MockTransport(stuck_ha.handler),
        verify_delay_s=0.0,
    )
    assert res_stuck.error == "verify_mismatch"
    assert "Włączyłam" not in res_stuck.speak


@pytest.mark.asyncio
async def test_post_write_verify_checks_requested_value_and_polls_lagging_state(
    ha_env: None, home_cfg: Settings
) -> None:
    """H5: unchanged temperature/mode/volume after write returns verify_mismatch; lagging state succeeds after polling."""
    cfg = home_cfg

    # Exact review case: AC already on at 24 °C, mock ignores set_temperature(18) -> must NOT say "Ustawiłam ... 24"
    stuck_ac = FakeHomeAssistant(ignore_writes=True)
    stuck_ac.states["climate.ania"]["state"] = "cool"
    stuck_ac.states["climate.ania"]["attributes"]["temperature"] = 24
    res_temp_mismatch = await home(
        action="set_temperature",
        device="klima u Ani",
        temperature=18,
        settings=cfg,
        transport=httpx.MockTransport(stuck_ac.handler),
        verify_delay_s=0.01,
    )
    assert res_temp_mismatch.error == "verify_mismatch"
    assert "Ustawiłam" not in res_temp_mismatch.speak
    assert "jeszcze tego nie potwierdziło" in res_temp_mismatch.speak
    # 1 initial GET + 4 verify GETs (initial verify + 3 retries) = 5 GETs
    assert stuck_ac.get_counts["climate.ania"] == 5

    # Unchanged volume on Onkyo (requested 70%, stays at 35%) -> verify_mismatch
    res_vol_mismatch = await home(
        action="volume_set",
        device="amplituner",
        level=70,
        settings=cfg,
        transport=httpx.MockTransport(stuck_ac.handler),
        verify_delay_s=0.01,
    )
    assert res_vol_mismatch.error == "verify_mismatch"
    assert "Ustawiłam" not in res_vol_mismatch.speak

    # Lagging AC state: first 2 verify GETs still report 24 °C, 3rd verify GET reports 18 °C -> succeeds!
    verify_reads = 0
    lagging_state = copy.deepcopy(SNAPSHOT_STATES["climate.ania"])
    lagging_state["state"] = "cool"
    lagging_state["attributes"]["temperature"] = 24

    def lagging_handler(request: httpx.Request) -> httpx.Response:
        nonlocal verify_reads
        if request.method == "GET":
            if verify_reads >= 3:
                lagging_state["attributes"]["temperature"] = 18
            verify_reads += 1
            return httpx.Response(200, json=lagging_state)
        return httpx.Response(200, json=[])

    res_lagged = await home(
        action="set_temperature",
        device="klima u Ani",
        temperature=18,
        settings=cfg,
        transport=httpx.MockTransport(lagging_handler),
        verify_delay_s=0.01,
    )
    assert res_lagged.error is None
    assert "Ustawiłam klimę u Ani na osiemnaście stopni" in res_lagged.speak


@pytest.mark.asyncio
async def test_partial_write_reported_truthfully_when_second_service_fails(
    ha_env: None, home_cfg: Settings
) -> None:
    """H6: if turn_on succeeds and set_temperature fails, report partial_write truthfully."""
    cfg = home_cfg
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET":
            return httpx.Response(200, json=SNAPSHOT_STATES["climate.ania"])
        calls.append(path)
        if path.endswith("/climate/turn_on"):
            return httpx.Response(200, json=[])
        if path.endswith("/climate/set_temperature"):
            return httpx.Response(502, json={"message": "Bad gateway"})
        raise AssertionError(path)

    res = await home(
        action="set_temperature",
        device="klima u Ani",
        temperature=21,
        settings=cfg,
        transport=httpx.MockTransport(handler),
        verify_delay_s=0.0,
    )
    assert calls == ["/api/services/climate/turn_on", "/api/services/climate/set_temperature"]
    assert res.error == "partial_write"
    assert res.data["completed"] == ["climate.turn_on"]
    assert res.data["failed"] == "climate.set_temperature"
    assert "Włączyłam klimę u Ani, ale nie udało się ustawić temperatury" in res.speak


@pytest.mark.asyncio
async def test_utterance_wins_over_entity_id_passthrough_and_sensitive_attrs_stripped(
    ha_env: None, home_cfg: Settings
) -> None:
    """L1: 'wyłącz telewizor u Bartka' with device='climate.ania' must NOT touch Ania's AC; strip token/picture attrs."""
    cfg = home_cfg
    fake_ha = FakeHomeAssistant()
    fake_ha.states["media_player.tv_bartek"]["state"] = "on"

    tok = current_session_context.set({"utterance": "wyłącz telewizor u Bartka"})
    try:
        res = await home(
            action="turn_off",
            device="climate.ania",
            settings=cfg,
            transport=httpx.MockTransport(fake_ha.handler),
            verify_delay_s=0.0,
        )
    finally:
        current_session_context.reset(tok)

    assert res.error is None
    assert res.data["entity_id"] == "media_player.tv_bartek"
    assert fake_ha.service_calls == [("media_player", "turn_off", {"entity_id": "media_player.tv_bartek"})]
    assert "entity_picture" not in res.data["attributes"]
    assert "access_token" not in res.data["attributes"]
    assert res.data["attributes"]["friendly_name"] == 'TV Bartka 55"'

    # Arbitrary unconfigured entity_id pass-through is rejected
    unlisted = resolve_device("switch.secret_garage_door", cfg.home.aliases, allowlist=cfg.home.write_allowlist)
    assert unlisted.entity_id is None
    assert unlisted.error == "no_match"


def test_czy_u_nas_dziala_does_not_resolve_to_nas_sensor(home_cfg: Settings) -> None:
    aliases = home_cfg.home.aliases
    res = resolve_device("", aliases, utterance="czy u nas działa")
    assert res.entity_id is None
    assert res.error == "no_match"


# HIGH (round 5): the single-candidate exception must not swallow an unknown device noun. With
# exactly one writable entity, an utterance naming a device the aliases don't know ("światło",
# "lampka", "grzejnik", "rolety", "muzyka") must hand off — it used to execute the sole climate
# (or media_player) entity because the noun was missing from the generic stem list.
_ROUND5_SINGLE_ENTITY_UNKNOWNS = [
    "wyłącz światło w kuchni",
    "włącz lampkę",
    "włącz grzejnik",
    "zamknij rolety",
    "wyłącz muzykę",
]


def _single_entity_ha(domain: str) -> FakeHomeAssistant:
    fake_ha = FakeHomeAssistant()
    eid = f"{domain}.jedyne"
    fake_ha.states.clear()
    fake_ha.states[eid] = {
        "entity_id": eid,
        "state": "off",
        "attributes": {
            "friendly_name": "Jedyne urządzenie",
            **({"temperature": None} if domain == "climate" else {"volume_level": 0.3}),
        },
    }
    return fake_ha


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", ["climate", "media_player"])
@pytest.mark.parametrize("utterance", _ROUND5_SINGLE_ENTITY_UNKNOWNS)
async def test_single_candidate_write_never_executes_on_an_unknown_device_noun(
    utterance: str, domain: str, ha_env: None
) -> None:
    """HIGH (round 5): 'włącz lampkę' with one climate/media entity must hand off, never write."""
    single = Settings(home={"write_allowlist": [f"{domain}.jedyne"]})
    fake_ha = _single_entity_ha(domain)
    tok = current_session_context.set({"utterance": utterance})
    try:
        res = await home(
            action="turn_off" if utterance.startswith("wyłącz") else "turn_on",
            device=f"{domain}.jedyne",
            settings=single,
            transport=httpx.MockTransport(fake_ha.handler),
            verify_delay_s=0.0,
        )
    finally:
        current_session_context.reset(tok)
    assert res.error == "no_match", (utterance, domain, res.error, res.speak)
    assert res.handoff is True
    assert "Nie znam takiego urządzenia" in res.speak
    assert fake_ha.service_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("utterance", "action", "domain", "kwargs"),
    [
        ("włącz to", "turn_on", "climate", {}),
        ("ustaw 22 stopnie", "set_temperature", "climate", {"temperature": 22.0}),
        ("ustaw temperaturę", "set_temperature", "climate", {"temperature": 22.0}),
        ("ustaw na chłodzenie", "set_hvac_mode", "climate", {"hvac_mode": "cool"}),
        ("wyłącz ją", "turn_off", "media_player", {}),
        ("ustaw głośność na 50", "volume_set", "media_player", {"level": 50}),
    ],
)
async def test_single_candidate_still_executes_on_verb_filler_and_value_utterances(
    utterance: str, action: str, domain: str, kwargs: dict, ha_env: None
) -> None:
    """Positive control: verb + pronouns/fillers (plus a climate value) keep the H1 exception."""
    single = Settings(home={"write_allowlist": [f"{domain}.jedyne"]})
    fake_ha = _single_entity_ha(domain)
    tok = current_session_context.set({"utterance": utterance})
    try:
        res = await home(
            action=action,
            device=f"{domain}.jedyne",
            settings=single,
            transport=httpx.MockTransport(fake_ha.handler),
            verify_delay_s=0.0,
            **kwargs,
        )
    finally:
        current_session_context.reset(tok)
    assert res.error is None, (utterance, action, domain, res.error, res.speak)
    assert len(fake_ha.service_calls) >= 1


# MED round 5: an utterance's own device noun forbids candidates of the other writable domain.
def test_utterance_device_noun_forbids_other_domain_candidates() -> None:
    """'włącz telewizor u Ani' must not ride the 'u Ani' room alias onto climate.ania: the noun
    names media_player, so a climate candidate is rejected, while the same-room media_player
    alias, when the household has one, resolves. With no TV alias at that place the TV is an
    unknown device next to a person place: the write refuses as unknown_device (F1) with the
    same 'Nie znam takiego urządzenia.' line the no_match rule used to speak."""
    aliases = {"klima u Ani": "climate.ania", "u Ani": "climate.ania"}
    res = resolve_device("włącz telewizor u Ani", aliases, utterance="włącz telewizor u Ani", action="turn_on")
    assert res.entity_id is None, (res.error, res.entity_id)
    assert res.error == "unknown_device"

    # The same-room entity of the right domain, when an alias exists, still resolves.
    aliases_tv = dict(aliases, **{"telewizor u Ani": "media_player.tv_ania"})
    ok = resolve_device("włącz telewizor u Ani", aliases_tv, utterance="włącz telewizor u Ani", action="turn_on")
    assert ok.error is None and ok.entity_id == "media_player.tv_ania"

    # And the reverse: a TV noun can't be hijacked for a climate write either.
    rev = resolve_device("ustaw klimę u Ani na 22 stopnie", aliases_tv, utterance="ustaw klimę u Ani na 22 stopnie", action="set_temperature")
    assert rev.error is None and rev.entity_id == "climate.ania"


def test_write_allowed_derives_the_domain_from_the_entity_id() -> None:
    """r5 6f: is_write_allowed must follow the entity_id's own domain prefix, never the `domain`
    argument (a stale or LLM-supplied one): a climate id never passes on a media_player.*
    wildcard, and a player id keeps its wildcard unlock even when the argument lies."""
    assert not is_write_allowed("climate.ania", "media_player", "turn_on", ["media_player.*"], [])
    assert is_write_allowed("media_player.tv_salon", "button", "press", ["media_player.*"], [])


@pytest.mark.asyncio
async def test_write_ask_back_never_offers_a_read_only_boiler(
    ha_env: None, home_cfg: Settings
) -> None:
    """r5 6g: write ask-back candidates come from _write_candidate_entities, where alias targets
    in writable domains (climate.boiler_heating via 'piec'/'kocioł') appeared even though
    is_write_allowed refuses every wildcard/unlisted write on them. The question must offer only
    writable entities."""
    fake_ha = FakeHomeAssistant()
    tok = current_session_context.set({"utterance": "wyłącz"})
    try:
        res = await home(
            action="turn_off",
            device="climate.ania",
            settings=home_cfg,
            transport=httpx.MockTransport(fake_ha.handler),
            verify_delay_s=0.0,
        )
    finally:
        current_session_context.reset(tok)
    assert res.error == "ambiguous_device"
    assert res.speak.endswith("?")
    assert "Kocioł" not in res.speak
    assert "piec" not in res.speak
    assert "kocioł" not in res.speak.lower()


def test_single_candidate_write_ask_back_is_an_open_question() -> None:
    """6h: 'Masz na myśli klimę w sypialni?' promised that a bare 'tak' would resolve to the
    candidate — but the pipeline keeps no pending-clarification state (the follow-up is classified
    fresh; the home tool has no cross-turn memory), so the promise dead-ended. The ask-back is
    now an open question that invites the repeat."""
    aliases = {"klima w sypialni": "climate.sypialnia", "telewizor w kuchni": "media_player.tv_kuchnia"}
    res = resolve_device(
        "klima w sypialni", aliases, utterance="włącz klimę wieczorem w domu", action="turn_on"
    )
    assert res.entity_id is None and res.error == "ambiguous_device"
    assert res.ask_back.startswith("Które urządzenie: klima w sypialni"), res.ask_back
    assert res.ask_back.endswith("?")
    assert "Masz na myśli" not in res.ask_back
    # Several scored candidates keep the already-open listing question.
    two = resolve_device(
        "klima", {"klima w sypialni": "climate.sypialnia", "klima w salonie": "climate.salon"},
        utterance="włącz klimę", action="turn_on",
    )
    assert two.ask_back.startswith("Które urządzenie masz na myśli"), two.ask_back


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("utterance", "kwargs", "speak_fragment"),
    [
        ("ustaw ciepłą wodę na 60", dict(action="set_temperature", device="ciepła woda", temperature=60), "Kocioł"),
        ("włącz bojler", dict(action="turn_on", device="bojler"), "Kocioł"),
        ("wyłącz konsolę", dict(action="turn_off", device="konsola"), "tylko do odczytu"),
    ],
)
async def test_write_to_a_named_read_only_device_is_refused_out_loud(
    utterance: str,
    kwargs: dict[str, Any],
    speak_fragment: str,
    ha_env: None,
    home_cfg: Settings,
) -> None:
    """A write aimed at a named read-only device (the boiler's hot water, a binary sensor) is
    refused out loud, never "Nie znam takiego urządzenia" plus a handoff that lets Hermes try."""
    fake_ha = FakeHomeAssistant()
    transport = httpx.MockTransport(fake_ha.handler)
    tok = current_session_context.set({"utterance": utterance})
    try:
        res = await home(settings=home_cfg, transport=transport, verify_delay_s=0.0, **kwargs)
    finally:
        current_session_context.reset(tok)
    assert res.error == "write_not_allowed", (utterance, res.error, res.speak)
    assert res.handoff is False
    assert speak_fragment in res.speak
    assert fake_ha.service_calls == []


# ---- Bug hunt: spoken-reply grammar and message leaks ------------------------------------------------

@pytest.mark.asyncio
async def test_water_heater_status_uses_the_label_gender(ha_env: None, home_cfg: Settings) -> None:
    """'Bojler jest włączona' is wrong Polish — bojler/zasobnik/piec are masculine; only 'ciepła
    woda' is feminine. The status reply must follow the spoken label's gender."""
    fake_ha = FakeHomeAssistant()
    fake_ha.states["water_heater.boiler_dhw"] = {
        "entity_id": "water_heater.boiler_dhw",
        "state": "eco",
        "attributes": {"current_temperature": 55.0, "temperature": 60, "friendly_name": "Ciepła woda"},
    }
    # The household's spoken form for the entity is masculine ("bojler") — as when a household
    # names its tank "bojler" in [spoken_forms].
    cfg = Settings(
        home={
            "household_file": home_cfg.home.household_file,
            "spoken_forms": {"water_heater.boiler_dhw": ("bojler", "bojlera", "bojlera")},
        }
    )
    transport = httpx.MockTransport(fake_ha.handler)

    bojler = await home(action="status", device="bojler", settings=cfg, transport=transport)
    assert bojler.error is None, bojler.speak
    assert "Bojler jest włączony" in bojler.speak, bojler.speak

    # Feminine labels keep feminine agreement (the same reply builder, feminine spoken form).
    from gaja.tools.home_tool import _speak_status

    state_obj = {"state": "eco", "attributes": {"current_temperature": 55.0, "temperature": 60}}
    fem = _speak_status(
        "water_heater.boiler_dhw", state_obj, {}, lang="pl",
        spoken_forms={"water_heater.boiler_dhw": ("ciepła woda", "ciepłą wodę", "ciepłej wody")},
    )
    assert "Ciepła woda jest włączona" in fem, fem

    # Masculine off-state too.
    fake_ha.states["water_heater.boiler_dhw"]["state"] = "off"
    off = await home(action="status", device="bojler", settings=cfg, transport=transport)
    assert "Bojler jest wyłączony" in off.speak, off.speak


@pytest.mark.asyncio
async def test_write_to_missing_entity_never_speaks_the_raw_entity_id(
    ha_env: None, home_cfg: Settings
) -> None:
    """A write aimed at an entity that Home Assistant does not know (404) must refuse in Polish
    without reading the raw entity_id out loud ('Nie znalazłam encji sensor.biuro_drukarka…')."""
    fake_ha = FakeHomeAssistant()  # climate.ania removed below -> the write's GET returns 404
    del fake_ha.states["climate.ania"]
    transport = httpx.MockTransport(fake_ha.handler)
    res = await home(action="turn_on", device="klima u Ani", settings=home_cfg, transport=transport)
    assert res.error == "entity_not_found"
    assert res.speak, "the refusal must be spoken"
    assert "climate.ania" not in res.speak
    assert res.speak.startswith("Nie znalazłam")


def test_status_label_falls_back_to_spoken_object_id_not_the_raw_entity_id() -> None:
    """An entity with no alias and no friendly_name must not read its raw entity_id out loud
    ('Media_player.tv_salon jest wyłączony') — the object id's words are the spoken fallback."""
    from gaja.tools.home_tool import _speak_status

    speak = _speak_status("media_player.tv_salon", {"state": "standby", "attributes": {}}, {}, lang="pl")
    assert "media_player" not in speak.lower() and "_" not in speak, speak
    assert "tv salon" in speak.lower(), speak


@pytest.mark.asyncio
async def test_write_naming_allowed_device_and_boiler_executes_allowed_and_refuses_boiler_aloud(
    ha_env: None, home_cfg: Settings
) -> None:
    """F3: "wyłącz piec i klimę u Ani" names two devices. The boiler (climate.boiler_heating,
    read-only by its id markers) must stay untouched, the allowed AC write must still execute,
    and the reply must say out loud that the boiler cannot be changed — never a silent drop."""
    fake_ha = FakeHomeAssistant()
    transport = httpx.MockTransport(fake_ha.handler)
    tok = current_session_context.set({"utterance": "wyłącz piec i klimę u Ani"})
    try:
        res = await home(action="turn_off", device="", settings=home_cfg, transport=transport, verify_delay_s=0.0)
    finally:
        current_session_context.reset(tok)
    # The allowed part happened; the boiler was never sent a service call.
    assert fake_ha.service_calls == [("climate", "turn_off", {"entity_id": "climate.ania"})]
    assert res.error is None, res.data
    assert "climate.boiler_heating" not in {c[2].get("entity_id") for c in fake_ha.service_calls}
    # The refusal is spoken, in the existing boiler wording.
    assert "Wyłączyłam klimę u Ani" in res.speak, res.speak
    assert "Kocioł jest tylko do odczytu" in res.speak, res.speak
    assert res.data.get("read_only_refused") == ["climate.boiler_heating"]


@pytest.mark.asyncio
async def test_write_naming_only_the_allowed_device_stays_free_of_boiler_refusal(
    ha_env: None, home_cfg: Settings
) -> None:
    """Control for F3: without a co-named read-only device the reply stays the plain success."""
    fake_ha = FakeHomeAssistant()
    transport = httpx.MockTransport(fake_ha.handler)
    tok = current_session_context.set({"utterance": "wyłącz klimę u Ani"})
    try:
        res = await home(action="turn_off", device="", settings=home_cfg, transport=transport, verify_delay_s=0.0)
    finally:
        current_session_context.reset(tok)
    assert res.error is None, res.data
    assert "Kocioł" not in res.speak, res.speak


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("utterance", "action"),
    [
        ("włącz zmywarkę u Ani", "turn_on"),
        ("wyłącz ekspres do kawy u Ani", "turn_off"),
        ("włącz piekarnik u rodziców", "turn_on"),
    ],
)
async def test_a_person_place_never_stands_in_for_an_unknown_device_write(
    utterance: str, action: str, ha_env: None, home_cfg: Settings
) -> None:
    """F1: the utterance names a device the household does not know next to a bare person-place
    alias ('u Ani'). The write must say it does not know that device and leave Home Assistant
    untouched — never switch the AC that happens to live at that person's place."""
    fake_ha = FakeHomeAssistant()
    transport = httpx.MockTransport(fake_ha.handler)
    tok = current_session_context.set({"utterance": utterance})
    try:
        res = await home(
            action=action, device=utterance, settings=home_cfg,
            transport=transport, verify_delay_s=0.0,
        )
    finally:
        current_session_context.reset(tok)
    assert res.error == "unknown_device", (utterance, res.error, res.speak)
    assert res.speak.startswith("Nie znam takiego urządzenia"), res.speak
    assert fake_ha.service_calls == [], (utterance, fake_ha.service_calls)
    # H1 parity: an unknown write still hands the turn to Hermes (it may know the device).
    assert res.handoff is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "utterance",
    ["czy u Ani działa pralka", "czy pralka u Ani działa", "jaka jest temperatura pralki u Ani"],
)
async def test_a_person_place_never_stands_in_for_an_unknown_device_read(
    utterance: str, ha_env: None, home_cfg: Settings
) -> None:
    """F3: a status question about an unknown device at a person place says it does not know
    that device — the AC's status is never spoken as the answer, and nothing is fetched."""
    def fail_if_called(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise AssertionError(f"no Home Assistant call expected for {utterance!r}")

    transport = httpx.MockTransport(fail_if_called)
    tok = current_session_context.set({"utterance": utterance})
    try:
        res = await home(
            action="status", device=utterance, settings=home_cfg, transport=transport
        )
    finally:
        current_session_context.reset(tok)
    assert res.error == "unknown_device", (utterance, res.error, res.speak)
    assert res.speak.startswith("Nie znam takiego urządzenia"), res.speak
    assert res.handoff is False, (utterance, res.speak)


@pytest.mark.asyncio
async def test_person_place_commands_over_known_devices_still_execute(
    ha_env: None, home_cfg: Settings
) -> None:
    """Controls for F1/F3: 'klima' is a known household device word, 'temperatura' a quantity
    word — these keep resolving through the person place, for writes and for reads."""
    fake_ha = FakeHomeAssistant()
    transport = httpx.MockTransport(fake_ha.handler)
    tok = current_session_context.set({"utterance": "włącz klimę u Ani"})
    try:
        res = await home(
            action="turn_on", device="włącz klimę u Ani", settings=home_cfg,
            transport=transport, verify_delay_s=0.0,
        )
    finally:
        current_session_context.reset(tok)
    assert res.error is None, (res.error, res.speak)
    assert fake_ha.service_calls == [("climate", "turn_on", {"entity_id": "climate.ania"})]

    tok = current_session_context.set({"utterance": "jaka jest temperatura u Ani"})
    try:
        read = await home(
            action="status", device="jaka jest temperatura u Ani",
            settings=home_cfg, transport=transport,
        )
    finally:
        current_session_context.reset(tok)
    assert read.error is None, (read.error, read.speak)
    assert read.speak.startswith("Klima u Ani"), read.speak


def test_resolve_device_rejects_person_place_hits_that_leave_an_unknown_content_noun(
    home_cfg: Settings,
) -> None:
    """Unit level: the bare 'u <person>' alias wins the subset match only when the remaining
    tokens are value/quantity words; an unknown content noun ('zmywarke', 'pralka', 'ekspres')
    makes the resolution refuse instead of guessing the person's entity."""
    aliases = home_cfg.home.aliases
    write = resolve_device("włącz zmywarkę u Ani", aliases, utterance="", action="turn_on")
    assert write.entity_id is None and write.error == "unknown_device"
    read = resolve_device("czy u Ani działa pralka", aliases, utterance="", action="status")
    assert read.entity_id is None and read.error == "unknown_device"
    # Known-device and quantity-word shapes keep resolving through the same place.
    assert resolve_device(
        "włącz klimę u Ani", aliases, utterance="", action="turn_on"
    ).entity_id == "climate.ania"
    assert resolve_device(
        "jaka jest temperatura u Ani", aliases, utterance="", action="status"
    ).entity_id == "climate.ania"


@pytest.mark.parametrize(
    "utterance",
    [
        "wyłącz klimę u Ani i telewizor w salonie",
        "wyłącz telewizor w salonie i klimę u Ani",
    ],
)
def test_two_known_devices_next_to_a_person_place_ask_back_not_unknown(utterance: str) -> None:
    """F1-R: a device word the household knows ('telewizor') is not an unknown noun. A write
    naming two known devices, one at a person's place, asks which one (as before F1) instead of
    claiming the household has no such device. Uses the documented example household, whose
    aliases carry no inflected 'klimę u Ani' that would resolve the AC directly."""
    example = Path(__file__).resolve().parent.parent / "docs" / "home.example.toml"
    aliases = Settings(home={"household_file": example}).home.aliases
    res = resolve_device(utterance, aliases, utterance=utterance, action="turn_off")
    assert res.entity_id is None
    assert res.error == "ambiguous_device", (res.error, res.ask_back)
    assert res.ask_back.startswith("Które urządzenie"), res.ask_back


@pytest.mark.asyncio
async def test_partial_write_is_symmetric_whatever_the_word_order_or_alias_length(
    ha_env: None, tmp_path: Path
) -> None:
    """F5: with the boiler alias LONGER than the AC alias ('kocioł gazowy w piwnicy' vs 'klima
    u Ani'), 'wyłącz kocioł gazowy w piwnicy i klimę u Ani' used to refuse the whole command
    and silently drop the allowed write. Whatever the order or alias length, the allowed AC
    write executes and the boiler refusal is spoken in the same reply; no service call ever
    reaches the boiler."""
    household = tmp_path / "home.toml"
    household.write_text(
        """
write_allowlist = ["climate.ania"]
boiler_id_substrings = ["boiler"]

[aliases]
"klima u Ani" = "climate.ania"
"u Ani" = "climate.ania"
"kocioł gazowy w piwnicy" = "climate.boiler_heating"

[spoken_forms]
"climate.ania" = ["klima u Ani", "klimę u Ani", "klimy u Ani"]
""",
        encoding="utf-8",
    )
    cfg = Settings(home={"household_file": household})
    for utterance in (
        "wyłącz kocioł gazowy w piwnicy i klimę u Ani",  # the read-only alias is longer: F5
        "wyłącz klimę u Ani i kocioł gazowy w piwnicy",  # the mirrored word order
    ):
        fake_ha = FakeHomeAssistant()
        transport = httpx.MockTransport(fake_ha.handler)
        tok = current_session_context.set({"utterance": utterance})
        try:
            res = await home(
                action="turn_off", device="", settings=cfg,
                transport=transport, verify_delay_s=0.0,
            )
        finally:
            current_session_context.reset(tok)
        assert res.error is None, (utterance, res.error, res.speak)
        assert fake_ha.service_calls == [("climate", "turn_off", {"entity_id": "climate.ania"})], (
            utterance, fake_ha.service_calls,
        )
        assert "Wyłączyłam klimę u Ani" in res.speak, (utterance, res.speak)
        assert "Kocioł jest tylko do odczytu" in res.speak, (utterance, res.speak)
        assert "climate.boiler_heating" not in {c[2].get("entity_id") for c in fake_ha.service_calls}


def test_status_agreement_follows_the_label_head_noun_ending() -> None:
    """F6: the gender of 'jest włączony/włączona/włączone' follows the label's head noun —
    derived from the nominative ending (consonant masculine, -a feminine, -o/-e/-um neuter),
    not a short prefix list. Abbreviations ('Podgrzewacz CWU') and numbers never decide; the
    few nouns that break the ending rule get an explicit exception."""
    from gaja.tools.home_tool import _label_gender, _speak_status

    assert _label_gender("Podgrzewacz CWU") == "m"
    assert _label_gender("Klimatyzator") == "m"
    assert _label_gender("Kocioł gazowy w piwnicy") == "m"
    assert _label_gender("Klima") == "f"
    assert _label_gender("Ciepła woda") == "f"  # adjective agrees: -a already says feminine
    assert _label_gender("Ogrzewanie") == "n"
    assert _label_gender("Światło") == "n"
    assert _label_gender("Centrum") == "n"  # -um
    assert _label_gender("Satelita") == "m"  # exception: masculine despite -a
    assert _label_gender("TX-NR6100") is None  # abbreviations/numbers never decide
    assert _label_gender("Nowy kocioł") == "m"  # -y adjective is skipped, the noun decides

    on = _speak_status(
        "water_heater.cwu",
        {"state": "on", "attributes": {"current_temperature": 50, "temperature": 50}},
        {"Podgrzewacz CWU": "water_heater.cwu"},
        "pl",
    )
    assert "Podgrzewacz CWU jest włączony" in on, on
    off = _speak_status(
        "climate.klimatyzator",
        {"state": "off", "attributes": {"current_temperature": 22.0}},
        {"Klimatyzator": "climate.klimatyzator"},
        "pl",
    )
    assert "Klimatyzator jest wyłączony" in off, off
    neuter = _speak_status(
        "climate.ogrzewanie",
        {"state": "heat", "attributes": {}},
        {"Ogrzewanie": "climate.ogrzewanie"},
        "pl",
    )
    assert "Ogrzewanie jest włączone" in neuter, neuter
    fem = _speak_status(
        "climate.klimatyzacja",
        {"state": "cool", "attributes": {"temperature": 22}},
        {"Klimatyzacja": "climate.klimatyzacja"},
        "pl",
    )
    assert "Klimatyzacja jest włączona" in fem, fem


def test_clamp_temperature_treats_null_limits_as_unknown() -> None:
    """F6: a climate entity whose min_temp/max_temp attributes are null has unknown limits;
    clamp_temperature must fall back to its default range instead of raising TypeError."""
    assert clamp_temperature(22, {"min_temp": None, "max_temp": None}) == (22, False)
    assert clamp_temperature(50, {"min_temp": None, "max_temp": None}) == (35, True)
    assert clamp_temperature(5, {"min_temp": None, "max_temp": None}) == (7, True)


@pytest.mark.asyncio
async def test_set_temperature_with_null_min_max_attributes_succeeds(
    ha_env: None, home_cfg: Settings
) -> None:
    """F6 end to end: "ustaw klimę w garażu na 22" against an entity reporting null limits must
    complete the write (unknown limits -> default clamp range), never raise TypeError."""
    fake_ha = FakeHomeAssistant()
    fake_ha.states["climate.garaz"] = {
        "entity_id": "climate.garaz",
        "state": "cool",
        "attributes": {
            "hvac_modes": ["off", "heat", "cool"],
            "min_temp": None,
            "max_temp": None,
            "target_temp_step": 1,
            "current_temperature": 19.0,
            "temperature": 20,
            "friendly_name": "Klima — Garaż",
        },
    }
    cfg = Settings(home={
        "aliases": {"klima w garażu": "climate.garaz"},
        "write_allowlist": ["climate.garaz"],
        "spoken_forms": {"climate.garaz": ("klima w garażu", "klimę w garażu", "klimy w garażu")},
    })
    transport = httpx.MockTransport(fake_ha.handler)
    tok = current_session_context.set({"utterance": "ustaw klimę w garażu na 22"})
    try:
        res = await home(
            action="set_temperature", device="klima w garażu", temperature=22,
            settings=cfg, transport=transport, verify_delay_s=0.0,
        )
    finally:
        current_session_context.reset(tok)
    assert res.error is None, (res.error, res.speak)
    assert ("climate", "set_temperature", {"entity_id": "climate.garaz", "temperature": 22}) in fake_ha.service_calls
    assert "dwadzieścia dwa stopnie" in res.speak, res.speak


def test_spoken_forms_fallback_declines_the_label() -> None:
    """F7: a label with no configured spoken forms must not be repeated in the nominative for
    every case ("Włączyłam klima u Ani") — the fallback derives acc/gen best-effort."""
    from gaja.tools.home_tool import _spoken_forms

    assert _spoken_forms("climate.ania", {"attributes": {}}, {"klima u Ani": "climate.ania"}) == (
        "klima u Ani",
        "klimę u Ani",
        "klimy u Ani",
    )
    assert _spoken_forms("water_heater.boiler_dhw", {"attributes": {}}, {"ciepła woda": "water_heater.boiler_dhw"}) == (
        "ciepła woda",
        "ciepłą wodę",
        "ciepłej wody",
    )


def test_spoken_forms_fallback_marks_unknown_forms_as_empty() -> None:
    """F7: when the helper cannot decline a label confidently, the slots stay empty so the
    callers pick a neutral sentence shape instead of speaking a wrong form."""
    from gaja.tools.home_tool import _spoken_forms

    assert _spoken_forms("climate.ania", {"attributes": {}}, {"pokój Ani": "climate.ania"}) == (
        "pokój Ani",
        "",
        "",
    )


@pytest.mark.asyncio
async def test_turn_on_speaks_the_declined_label_without_configured_forms(
    ha_env: None, home_cfg: Settings
) -> None:
    """F7 end to end: "włącz klimę w biurze" with only an alias (no [spoken_forms] entry) must
    speak the accusative ("Włączyłam klimę w biurze…"), never the nominative."""
    fake_ha = FakeHomeAssistant()
    fake_ha.states["climate.biuro"] = {
        "entity_id": "climate.biuro",
        "state": "off",
        "attributes": {
            "hvac_modes": ["off", "heat", "cool"],
            "min_temp": 7,
            "max_temp": 35,
            "target_temp_step": 1,
            "current_temperature": 21.0,
            "temperature": None,
            "friendly_name": "Klima — Biuro",
        },
    }
    cfg = Settings(home={
        "aliases": {"klima w biurze": "climate.biuro"},
        "write_allowlist": ["climate.biuro"],
    })
    transport = httpx.MockTransport(fake_ha.handler)
    tok = current_session_context.set({"utterance": "włącz klimę w biurze"})
    try:
        res = await home(action="turn_on", device="", settings=cfg, transport=transport, verify_delay_s=0.0)
    finally:
        current_session_context.reset(tok)
    assert res.error is None, (res.error, res.speak)
    assert res.speak.startswith("Włączyłam klimę w biurze"), res.speak
    assert "klima w biurze" not in res.speak, res.speak


@pytest.mark.asyncio
async def test_unknown_label_forms_speak_a_neutral_shape(ha_env: None, home_cfg: Settings) -> None:
    """F7: a label whose forms are unknown ("tv garaż") must not be spoken with a guessed case;
    the reply uses a neutral, case-free sentence shape instead."""
    fake_ha = FakeHomeAssistant()
    fake_ha.states["media_player.tv_garaz"] = {
        "entity_id": "media_player.tv_garaz",
        "state": "on",
        "attributes": {"friendly_name": "TV Garaż", "volume_level": 0.3},
    }
    cfg = Settings(home={
        "aliases": {"tv garaż": "media_player.tv_garaz"},
        "write_allowlist": ["media_player.*"],
    })
    transport = httpx.MockTransport(fake_ha.handler)

    tok = current_session_context.set({"utterance": "wyłącz tv garaż"})
    try:
        res = await home(action="turn_off", device="", settings=cfg, transport=transport, verify_delay_s=0.0)
    finally:
        current_session_context.reset(tok)
    assert res.error is None, (res.error, res.speak)
    assert res.speak == "Tv garaż: wyłączono.", res.speak

    fake_ha.service_calls.clear()
    tok = current_session_context.set({"utterance": "ustaw głośność tv garaż na 30"})
    try:
        res = await home(action="volume_set", device="", level=30, settings=cfg, transport=transport, verify_delay_s=0.0)
    finally:
        current_session_context.reset(tok)
    assert res.error is None, (res.error, res.speak)
    assert res.speak == "Ustawiono głośność na trzydzieści procent.", res.speak
