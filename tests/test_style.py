from __future__ import annotations

from dataclasses import replace

from gaja.text.chunker import SentenceChunker
from gaja.text.style import Style, parse_styled


def test_tags_switch_emotion_for_following_text() -> None:
    segments = parse_styled("[happy] Super! [calm] Teraz.", Style())
    assert [(s.emotion, t) for s, t in segments] == [("happy", "Super!"), ("calm", "Teraz.")]


def test_rate_tags_and_events() -> None:
    segments = parse_styled("[fast] Szybko [laugh] haha", Style())
    assert segments[0][0].rate > 1.0
    assert segments[1][0].events == ("laugh",)


def test_unknown_tag_is_dropped_not_spoken() -> None:
    assert parse_styled("[xyzzy] Hej", Style()) == [(Style(), "Hej")]


def test_default_style_is_kept_without_tags() -> None:
    default = Style(emotion="sad", rate=0.9)
    assert parse_styled("Tekst bez znacznika.", default) == [(default, "Tekst bez znacznika.")]


def test_chunker_then_style_parse_keeps_per_sentence_style() -> None:
    chunker = SentenceChunker(first_min_words=6, min_words=3, max_chars=220)
    raw = chunker.feed("[happy] Super, udało się! [calm] Teraz sprawdzę resztę.") + chunker.flush()
    style = Style()
    results: list[tuple[str, str]] = []
    for chunk in raw:
        for segment_style, text in parse_styled(chunk, style):
            results.append((segment_style.emotion, text))
            style = replace(segment_style, events=())
    assert results == [("happy", "Super, udało się!"), ("calm", "Teraz sprawdzę resztę.")]
