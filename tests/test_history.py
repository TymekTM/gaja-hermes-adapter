from __future__ import annotations

from gaja.llm.history import History


def test_messages_after_matches_add_then_messages_including_trim() -> None:
    history = History("sys", max_turns=2, max_chars=60)
    history.add("user", "pierwsze pytanie o pogodę w Krakowie")
    history.add("assistant", "Jest ciepło.")
    history.add("user", "a jutro?")
    history.add("assistant", "Podobnie.")
    preview = history.messages_after("user", "a pojutrze, na tydzień do przodu?")

    reference = History("sys", max_turns=2, max_chars=60)
    for turn in history.context():
        reference.add(turn["role"], turn["content"])
    reference.add("user", "a pojutrze, na tydzień do przodu?")

    assert preview == reference.messages()
    assert history.context()[-1]["content"] == "Podobnie."


def test_messages_after_blank_content_is_current_messages() -> None:
    history = History("sys", max_turns=4, max_chars=1000)
    history.add("user", "cześć")
    assert history.messages_after("user", "   ") == history.messages()


def test_trim_drops_to_half_the_budget_at_a_user_turn() -> None:
    history = History("sys", max_turns=100, max_chars=100)
    for i in range(5):
        history.add("user", f"pytanie {i} " + "x" * 8)
        history.add("assistant", f"odpowiedź {i} " + "y" * 6)
    context = history.context()
    assert context[0]["role"] == "user"
    assert sum(len(m["content"]) for m in context) <= 50


def test_prefix_is_stable_between_trims() -> None:
    history = History("sys", max_turns=100, max_chars=200)
    snapshots = []
    for i in range(12):
        history.add("user", f"pytanie numer {i}")
        history.add("assistant", f"odpowiedź numer {i}")
        snapshots.append(history.context())
    trims = sum(1 for a, b in zip(snapshots, snapshots[1:]) if b[: len(a)] != a)
    assert 1 <= trims <= 3


def test_turn_limit_trims_in_a_big_step() -> None:
    history = History("sys", max_turns=4, max_chars=10_000)
    for i in range(5):
        history.add("user", f"u{i}")
        history.add("assistant", f"a{i}")
    assert len(history.context()) <= 4
    assert history.context()[0]["role"] == "user"


def test_messages_with_pending_does_not_mutate() -> None:
    history = History("sys", max_turns=4, max_chars=1000)
    history.add("user", "cześć")
    preview = history.messages_with([("assistant", "hej"), ("user", "co tam")])
    assert [m["content"] for m in preview] == ["sys", "cześć", "hej", "co tam"]
    assert history.context() == [{"role": "user", "content": "cześć"}]
