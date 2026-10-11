from __future__ import annotations

import pytest

from gaja.devices.wake import WakeMatcher, edit_distance, match_wake_phrase, normalize


@pytest.mark.parametrize(
    ("text", "rest"),
    [
        ("Hej Gaja", ""),
        ("hej gajo", ""),
        ("Hej, Gaja!", ""),
        ("Hej Gaja jaka pogoda", "jaka pogoda"),
        ("Hej Gaja, jaka jest pogoda w Krakowie?", "jaka jest pogoda w Krakowie?"),
        ("hej gajo, włącz światło", "włącz światło"),
        ("Hey Gaia", ""),
        ("hey gaia what time is it", "what time is it"),
        ("Ej Gaju", ""),
        ("ej gaya, ile to dwa plus dwa", "ile to dwa plus dwa"),
        ("Hejgaja, włącz światło", "włącz światło"),
        ("no hej Gaja co słychać", "co słychać"),
        ("HEJ GAJA", ""),
        ("Hej Gaję", ""),
        ("hej gaj", ""),  # one edit from a name that starts with g
    ],
)
def test_wake_phrase_positives(text: str, rest: str) -> None:
    match = match_wake_phrase(text)
    assert match.matched, text
    assert match.rest == rest
    assert not match.bare


@pytest.mark.parametrize(
    "text",
    [
        "hej maja",
        "Hej Maja, jaka pogoda",
        "hej kaja",
        "jej gaja",  # one edit from "hej" is not enough: the greeting list is explicit
        "to jest Gaja",  # the name is not the first word and has no greeting
        "ala ma kota, hej gaja",  # greeting too far from the start
        "dobranoc",
        "",
        "   ",
        "hej",
        "Hej, jaka jest pogoda",
        "Maja to ładne imię",
    ],
)
def test_wake_phrase_negatives(text: str) -> None:
    match = match_wake_phrase(text)
    assert not match.matched, text
    assert match.rest == text.strip()


def test_bare_name_as_first_word_is_accepted_by_default() -> None:
    """Documented policy: the device already heard the full phrase, so 'Gaja' as the first word is enough."""
    match = match_wake_phrase("Gaja, jaka jest pogoda?")
    assert match.matched and match.bare
    assert match.rest == "jaka jest pogoda?"
    # ...which also accepts the sentence "Gaja to ładne imię" (the price of the rule); not as a later word.
    assert match_wake_phrase("Gaja to ładne imię").matched
    assert not match_wake_phrase("to ładne imię Gaja").matched


def test_bare_name_can_be_turned_off() -> None:
    assert not match_wake_phrase("Gaja, jaka pogoda", accept_bare_name=False).matched
    assert match_wake_phrase("Hej Gaja, jaka pogoda", accept_bare_name=False).matched


def test_bare_name_is_exact_not_fuzzy() -> None:
    assert not match_wake_phrase("gaj jest zielony").matched
    assert not match_wake_phrase("gala wieczorna").matched


def test_extra_variants_from_config() -> None:
    matcher = WakeMatcher(extra_names=["Kaja"], extra_greetings=["cześć"])
    assert matcher.match("Cześć Gaja, która godzina").rest == "która godzina"
    assert matcher.match("hej Kaja").matched
    assert not WakeMatcher().match("cześć Gaja").matched


def test_normalize_and_edit_distance() -> None:
    assert normalize("Łódź Gaję") == "lodz gaje"
    assert edit_distance("gaja", "gaj") == 1
    assert edit_distance("gaja", "maja") == 1
    assert edit_distance("abc", "abc") == 0
