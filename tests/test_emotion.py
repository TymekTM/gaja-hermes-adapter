from __future__ import annotations

import pytest

from gaja.hermes.progress import ProgressCoalescer, ProgressItem
from gaja.pipeline import _to_sentences
from gaja.text.emotion import classify_emotion, has_explicit_tag
from gaja.text.style import Style


@pytest.mark.parametrize(
    ("text", "emotion"),
    [
        ("Świetnie, dziękuję!", "happy"),
        ("Hurra, wygraliśmy!!", "excited"),
        ("Niestety nie mogę tego zrobić.", "sad"),
        ("Oddychaj powoli i spokojnie.", "calm"),
        ("To poważne ryzyko.", "serious"),
        ("Dwa plus dwa to cztery.", "neutral"),
        # REVIEW M9: inflected forms, and no "happy" for a plain acknowledgement.
        ("To fantastyczna wiadomość.", "excited"),
        ("Jest mi smutno.", "sad"),
        ("Gratulacje z okazji awansu.", "happy"),
        ("To bardzo poważna sprawa.", "serious"),
        ("Spokojnie, wszystko będzie dobrze.", "calm"),
        ("Dobrze, zaraz sprawdzę.", "neutral"),
        ("Cieszę się, że pytasz.", "happy"),
        ("Bardzo mi przykro z tego powodu.", "sad"),
        ("To była porażka.", "sad"),
        ("Odpocznij chwilę.", "calm"),
        ("Uważaj, to ryzykowne.", "serious"),
        ("Kupiłem mleko w supermarkecie.", "neutral"),
        ("Gotowe!", "neutral"),
        ("Super, udało się!", "excited"),
    ],
)
def test_classify_emotion(text: str, emotion: str) -> None:
    assert classify_emotion(text) == emotion


def test_explicit_tag_detection() -> None:
    assert has_explicit_tag("[happy] Hej")
    assert not has_explicit_tag("Hej, jak leci")


def test_auto_emotion_applies_to_untagged_sentences() -> None:
    sentences, _ = _to_sentences(["Hurra, wygraliśmy!!"], Style(), auto_emotion=True)
    assert sentences[0].style.emotion == "excited"


def test_auto_emotion_off_keeps_neutral() -> None:
    sentences, _ = _to_sentences(["Hurra, wygraliśmy!!"], Style(), auto_emotion=False)
    assert sentences[0].style.emotion == "neutral"


def test_explicit_tag_wins_over_auto_emotion() -> None:
    sentences, _ = _to_sentences(["[sad] Hurra, wygraliśmy!!"], Style(), auto_emotion=True)
    assert sentences[0].style.emotion == "sad"


def test_untagged_sentence_ignores_carried_explicit_style() -> None:
    sentences, _ = _to_sentences(["[sad] Niestety nie. ", "Dwa plus dwa to cztery."], Style(), auto_emotion=True)
    assert [s.style.emotion for s in sentences] == ["sad", "neutral"]


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_coalescer_flushes_first_then_at_interval() -> None:
    clock = _Clock()
    coalescer = ProgressCoalescer(0.5, clock=clock)
    assert coalescer.push("text_delta", "Ala ") == [ProgressItem("text_delta", "Ala ")]
    assert coalescer.push("text_delta", "ma ") == []
    assert coalescer.push("text_delta", "kota.") == []
    clock.now = 0.6
    assert coalescer.push("text_delta", " Koniec.") == [ProgressItem("text_delta", "ma kota. Koniec.")]
    assert not coalescer.pending


def test_coalescer_emits_immediate_types_with_pending_text_first() -> None:
    clock = _Clock()
    coalescer = ProgressCoalescer(0.5, clock=clock)
    coalescer.push("text_delta", "Start")
    assert coalescer.push("text_delta", " dalej") == []
    assert coalescer.push("tool_call", "Hermes uruchamia narzędzie: web") == [
        ProgressItem("text_delta", " dalej"),
        ProgressItem("tool_call", "Hermes uruchamia narzędzie: web"),
    ]


def test_coalescer_keeps_only_latest_status_per_type() -> None:
    clock = _Clock()
    coalescer = ProgressCoalescer(0.5, clock=clock)
    coalescer.push("step", "analizuje")
    assert coalescer.push("step", "analizuje dalej") == []
    assert coalescer.flush() == [ProgressItem("step", "analizuje dalej")]
    assert coalescer.flush() == []
