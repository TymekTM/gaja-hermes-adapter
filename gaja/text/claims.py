"""Detectors for reply text that claims an action or a tool call the model never made.

The small chat model sometimes ends a turn with the announcement instead of the call:
"Jasne, przekazuję to Hermesowi… Wywołaj narzędzie handoff_to_hermes" as plain text, or a
past-tense "Zrobiłem to teraz" with no tool call behind it (bench/llm_noisy_eval.py, the
"claimed action without call" metric in docs/LLM_NOISY_EVAL.md). The pipeline checks every
streamed sentence with these matchers before the sentence reaches TTS (Session._enqueue, and
only on local replies): an announced handoff or a written-out tool call turns into a real
handoff_to_hermes carrying the user's utterance, and a first-person action claim is dropped,
so nothing is spoken that did not happen. Deliberately narrow (Polish and English, literal
tool names, first-person claims with a negation carve-out) and text-only: no cost and no
extra model call on a normal turn.
"""

from __future__ import annotations

import re

# Tool names that never belong in spoken text: written into a reply, they are the model
# announcing (or rehearsing) the call instead of making it. Bare words that double as ordinary
# chat (spotify, remember, forget, home, calculate) count only with call syntax attached.
_TOOL_NAME = (
    r"(?:handoff_to_hermes|set_speaker_volume|get_time|get_weather|get_sun|convert_units"
    r"|exchange_rate|wiki_summary|server_status|next_holiday|manage_timer)"
)
_TOOL_WORD = r"(?:spotify|remember|forget|home|calculate)"
TOOL_CALL_TEXT = re.compile(
    r"<\|tool_call_(?:start|end)\|>"
    rf"|\b{_TOOL_NAME}\b"
    rf"|\b{_TOOL_WORD}\s*\(",
    re.IGNORECASE,
)

# First-person past-tense action verbs; [ae]m covers both genders (zrobiłem/zrobiłam), and
# usuną-/usunę- keeps the vowel change. A negation ("nie zrobiłem", "nic nie usunąłem") is an
# honest sentence and must never match; "przypomniałem sobie" is recalling, not acting.
_ACTION_PL = (
    r"zrobił[ae]m|załatwił[ae]m|wykonał[ae]m|usunął[ae]m|usunęł[ae]m|skasował[ae]m|wykasował[ae]m|"
    r"zapisał[ae]m|zanotował[ae]m|zapamiętał[ae]m|wysłał[ae]m|przesłał[ae]m|przekazał[ae]m|"
    r"ustawił[ae]m|zaplanował[ae]m|zamówił[ae]m|zarezerwował[ae]m|"
    r"włączył[ae]m|wyłączył[ae]m|puścił[ae]m|wznowił[ae]m|zatrzymał[ae]m|"
    r"podgłośnił[ae]m|[sś]ciszył[ae]m|wyciszył[ae]m|przyciszył[ae]m|"
    r"anulował[ae]m|odwołał[ae]m|dodał[ae]m|sprawdził[ae]m|poszukał[ae]m|wyszukał[ae]m|"
    r"znalazł[ae]m|odtworzył[ae]m|przełączył[ae]m|zmienił[ae]m|zaktualizował[ae]m|"
    r"przypomniał[ae]m(?!\s+sobie)"
)
ACTION_CLAIM_PL = re.compile(
    r"(?<!\b[Nn]ie )(?<!\b[Nn]ic [Nn]ie )\b(?:" + _ACTION_PL + r")\b",
    re.IGNORECASE,
)
ACTION_CLAIM_EN = re.compile(
    r"\bi\s*(?:(?:have|'ve|just|already)\s+)+(?:done\s+it|deleted|removed|cleared|sent|saved|set"
    r"|turned|ordered|booked|scheduled|cancelled|canceled|fixed|added|created|updated|started"
    r"|stopped|played|checked|found|handled)\b"
    r"|\bit'?s\s+(?:all\s+)?done\b|\ball\s+set\b",
    re.IGNORECASE,
)


def mentions_tool_call(text: str) -> bool:
    """Whether the text announces or writes out a tool call instead of making one."""
    return TOOL_CALL_TEXT.search(text) is not None


def claims_action(text: str) -> bool:
    """Whether the text claims a completed first-person action ("Zrobiłem to teraz").

    Without a tool call such a sentence would speak a fact that is not true, so the pipeline
    drops it instead of synthesizing it.
    """
    return ACTION_CLAIM_PL.search(text) is not None or ACTION_CLAIM_EN.search(text) is not None
