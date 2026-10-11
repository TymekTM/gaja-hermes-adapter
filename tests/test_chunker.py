from __future__ import annotations

import re

import pytest

from gaja.text.chunker import SentenceChunker, SpeakerPrefixStripper, split_sentences, strip_style_tags


def chunker(first_min_words: int = 6) -> SentenceChunker:
    return SentenceChunker(first_min_words=first_min_words, min_words=3, max_chars=220)


def run(text: str, *, first_min_words: int = 6) -> list[str]:
    c = chunker(first_min_words)
    return c.feed(text) + c.flush()


def test_splits_two_sentences() -> None:
    assert run("Dzień jest ładny dzisiaj. Idziemy na spacer po parku.") == [
        "Dzień jest ładny dzisiaj.",
        "Idziemy na spacer po parku.",
    ]


def test_abbreviation_does_not_end_sentence() -> None:
    assert run("Spotkanie jest o godz. dziesiątej rano w biurze.") == [
        "Spotkanie jest o godz. dziesiątej rano w biurze."
    ]


def test_np_inside_sentence_is_kept() -> None:
    assert run("Jest dużo rzeczy, np. książki i filmy. Tak.") == [
        "Jest dużo rzeczy, np. książki i filmy.",
        "Tak.",
    ]


def test_decimal_number_does_not_end_sentence() -> None:
    assert run("Temperatura wynosi 3.5 stopnia w nocy. Dobrze.") == [
        "Temperatura wynosi 3.5 stopnia w nocy.",
        "Dobrze.",
    ]


def test_ellipsis_merges_short_fragments() -> None:
    assert run("Hmm... to zależy. Dobrze.") == ["Hmm... to zależy.", "Dobrze."]


def test_first_chunk_cut_at_comma_after_enough_words() -> None:
    parts = run("Wczoraj wieczorem długo rozmawialiśmy o pogodzie, potem poszliśmy spać.")
    assert parts[0] == "Wczoraj wieczorem długo rozmawialiśmy o pogodzie,"
    assert parts[1] == "potem poszliśmy spać."


def test_comma_before_threshold_does_not_cut() -> None:
    assert run("Dzisiaj rano w domu, pijemy kawę i czytamy książkę.") == [
        "Dzisiaj rano w domu, pijemy kawę i czytamy książkę."
    ]


def capped(text: str, first_max_words: int) -> list[str]:
    c = SentenceChunker(first_min_words=3, min_words=3, max_chars=220, first_max_words=first_max_words)
    out: list[str] = []
    for word in re.findall(r"\S+\s*", text):
        out += c.feed(word)
    return out + c.flush()


def test_first_chunk_capped_at_word_limit_without_punctuation() -> None:
    parts = capped("Oczywiście sprawdzę to dla ciebie teraz i zaraz wrócę.", 5)
    assert parts[0] == "Oczywiście sprawdzę to dla ciebie"
    assert parts[1] == "teraz i zaraz wrócę."


def test_word_cap_skips_style_tags() -> None:
    parts = capped("[happy] Super sprawa jest taka, że zrobimy to wszystko razem.", 3)
    assert parts[0] == "[happy] Super sprawa jest"


def test_word_cap_does_not_split_a_reply_that_ends_within_limit() -> None:
    assert capped("Oczywiście sprawdzę to dla ciebie.", 8) == ["Oczywiście sprawdzę to dla ciebie."]


def test_word_cap_only_applies_to_first_chunk() -> None:
    parts = capped("Raz dwa trzy cztery pięć sześć siedem osiem.", 3)
    assert parts == ["Raz dwa trzy", "cztery pięć sześć siedem osiem."]


def test_style_tags_stay_with_their_sentence() -> None:
    assert run("[happy] Super, udało się! [calm] Teraz sprawdzę resztę.") == [
        "[happy] Super, udało się!",
        "[calm] Teraz sprawdzę resztę.",
    ]


def test_streaming_matches_whole_text() -> None:
    text = "[happy] Super, udało się! Ulica Kwiatowa jest zamknięta. Temperatura 3.5 stopnia, np. w nocy. Koniec."
    whole = run(text)
    c = chunker()
    streamed: list[str] = []
    for ch in text:
        streamed.extend(c.feed(ch))
    streamed.extend(c.flush())
    assert streamed == whole


def test_split_sentences_helper() -> None:
    assert split_sentences("Pierwsze zdanie tutaj. Drugie zdanie tutaj.") == [
        "Pierwsze zdanie tutaj.",
        "Drugie zdanie tutaj.",
    ]


def test_strip_style_tags() -> None:
    assert strip_style_tags("[happy] Hej [sigh] trudno") == " Hej  trudno"


def _strip_label(deltas: list[str]) -> str:
    stripper = SpeakerPrefixStripper()
    return "".join(stripper.feed(d) for d in deltas) + stripper.flush()


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        (["Gaja: Twoje pytanie jest niezrozumiałe."], "Twoje pytanie jest niezrozumiałe."),
        (["Ga", "ja", ":", " Nie rozumiem."], "Nie rozumiem."),
        (["**Gaja**: ", "Cześć!"], "Cześć!"),
        (["Assistant: Hi."], "Hi."),
        (["Gaja jest", " moim imieniem."], "Gaja jest moim imieniem."),
        (["Gaja, ", "czyli ja."], "Gaja, czyli ja."),
        (["Dzień dobry. Gaja: to ja."], "Dzień dobry. Gaja: to ja."),
        (["Ga"], "Ga"),
        (["Gaja:"], ""),
        (["", "Jasne."], "Jasne."),
    ],
)
def test_speaker_prefix_is_dropped_only_at_reply_start(deltas: list[str], expected: str) -> None:
    assert _strip_label(deltas) == expected
