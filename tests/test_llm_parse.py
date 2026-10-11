from __future__ import annotations

from gaja.llm.parse import NativeCallAccumulator, PythonicCallSplitter, ThinkFilter, ToolCall


def feed_each_char(filter_: ThinkFilter, text: str) -> str:
    return "".join(filter_.feed(ch) for ch in text) + filter_.flush()


def test_stray_open_tag_starts_a_new_reasoning_block() -> None:
    assert feed_each_char(ThinkFilter(starts_inside=False), "Hej <think>myślę</think> świat") == "Hej  świat"


def test_think_block_is_removed_even_when_split_across_deltas() -> None:
    assert feed_each_char(ThinkFilter(), "<think>krótkie rozumowanie</think>Cześć!") == "Cześć!"


def test_reasoning_without_opening_tag_is_dropped() -> None:
    assert feed_each_char(ThinkFilter(starts_inside=True), "ukryte myśli</think>Cześć!") == "Cześć!"


def test_plain_text_passes_through() -> None:
    assert feed_each_char(ThinkFilter(), "Dzień dobry, jak leci?") == "Dzień dobry, jak leci?"


def test_pythonic_tool_call_is_split_from_spoken_text() -> None:
    splitter = PythonicCallSplitter()
    source = 'Jasne. <|tool_call_start|>[handoff_to_hermes(task="sprawdź pogodę", reason="dane")]<|tool_call_end|>'
    spoken: list[str] = []
    calls: list[ToolCall] = []
    for ch in source:
        text, found = splitter.feed(ch)
        spoken.append(text)
        calls.extend(found)
    tail, tail_calls = splitter.finish()
    calls.extend(tail_calls)
    assert "".join(spoken) + tail == "Jasne. "
    assert calls == [ToolCall("handoff_to_hermes", {"task": "sprawdź pogodę", "reason": "dane"})]


def test_malformed_pythonic_call_is_dropped() -> None:
    splitter = PythonicCallSplitter()
    text, calls = splitter.feed("<|tool_call_start|>[broken(<|tool_call_end|>Hej")
    assert calls == []
    assert text == "Hej"

def test_held_partial_text_is_emitted_at_end() -> None:
    splitter = PythonicCallSplitter()
    text, _ = splitter.feed("3 <")
    tail, calls = splitter.finish()
    assert text + tail == "3 <"
    assert calls == []


def test_native_fragments_are_assembled_per_index() -> None:
    accumulator = NativeCallAccumulator()
    accumulator.feed([{"index": 0, "function": {"name": "handoff_to_hermes", "arguments": '{"task": "a'}}])
    accumulator.feed([{"index": 0, "function": {"arguments": '", "reason": "b"}'}}])
    assert accumulator.finish() == [ToolCall("handoff_to_hermes", {"task": "a", "reason": "b"})]
