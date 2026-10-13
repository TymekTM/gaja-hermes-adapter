from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gaja.llm.client import LLMClient
from gaja.llm.parse import ToolCall
from gaja.llm.prompt import load_text_block

PROMISE_PL = re.compile(
    r"\b(?:przeka[zż]\w*|zlec\w*|przesy?ł\w*|wysy?ł\w*|poprosz\w*|zapytam)\s+"
    r"(?:(?:to|ci|mu|tę|tego|zadanie|sprawę|temat|do|teraz|od\s+razu|już)\s+){0,3}hermes(?:owi|a|em|ie|u)?\b"
    r"|\bhermes\s+(?:już\s+)?(?:to\s+)?(?:sprawdza|sprawdzi|zajmie\s+się|się\s+tym\s+zajmie|jest\s+na\s+tym)"
    r"|\b(?:sprawdza|sprawdzi|zajmie\s+się|zajmuje\s+się)\s+(?:tym\s+|to\s+)?(?:już\s+)?hermes\b",
    re.IGNORECASE,
)
PROMISE_EN = re.compile(
    r"\b(?:hand|pass|send|forward)\w*\s+(?:(?:it|this|that|the\s+task)\s+)?(?:(?:over|off|along)\s+)?to\s+hermes\b"
    r"|\b(?:i'?ll|let\s+me|i\s+will)\s+ask\s+hermes\b|\bhermes\s+(?:is\s+on\s+it|will\s+(?:check|handle))",
    re.IGNORECASE,
)
# "Hermes" as the god, the brand or the subject of a question, not as the agent being addressed.
MYTHOLOGY = re.compile(
    r"mitolog|mytholog|\bbog(?:a|ów|iem|u)?\b|\bbóg\b|\bgod\b|olimp|olymp|zeus|posłaniec\s+bog|messenger\s+of|"
    r"grec|greek|herm[eè]s\s+paris|torebk|perfum|o\s+hermesie|about\s+hermes|kim\s+(?:jest|był)\s+hermes|"
    r"who\s+(?:is|was)\s+hermes",
    re.IGNORECASE,
)
EXPLICIT_HERMES = re.compile(
    r"\b(?:zapytaj|spytaj|popro[sś]|każ|zleć|przeka[zż]\w*|niech)\s+(?:\w+\s+){0,2}hermes\w*"
    r"|^\W*hermes(?:ie)?\s*[,:!]|\b(?:ask|tell|have|let)\s+hermes\b|\bhermes\s*,\s*(?:please|proszę)"
    r"|\b(?:w|przez|użyj|use|via|using|in)\s+hermes(?:ie|a|em)?\b",
    re.IGNORECASE,
)

# Imperatives only Hermes can carry out (write/send/store/control/search). In 3-class mode they win
# over TOOL and LOCAL, since the small model tends to file "napisz skrypt…" or "włącz klimatyzację…"
# under a tool. Creative asks ("napisz wierszyk") stay local.
HERMES_ACTION = re.compile(
    r"\b(?:napisz|wyślij|wyslij|odpisz|zapisz|zapamiętaj|zapamietaj|dodaj|usuń|usun|ustaw|włącz|wlacz|wyłącz|wylacz|"
    r"przyciemnij|rozjaśnij|zgaś|zgas|zamów|zamow|zarezerwuj|znajdź|znajdz|wyszukaj|poszukaj|porównaj|porownaj|"
    r"zaplanuj|przygotuj)\b"
    r"(?!\s+(?:mi\s+)?(?:\w+\s+)?(?:wiersz\w*|wierszyk\w*|haiku|limeryk\w*|rym\w*|żart\w*|zart\w*|dowcip\w*|zagadk\w*))"
    r"|\bprzypomnij\s+mi\b(?!\s+co)"
    r"|\b(?:write|draft)\s+(?:me\s+)?(?:an?\s+)?(?:\w+\s+){0,3}(?:script|program|function|code|e-?mail|letter|essay|contract|report)"
    r"|\b(?:send|remind\s+me|set\s+an?\s+(?:alarm|reminder|timer)|turn\s+(?:on|off)|schedule|book\s+a|order|"
    r"search\s+(?:the\s+web|online|for)|find\s+me)\b",
    re.IGNORECASE,
)

# A selected tool only runs if the utterance (or, for follow-ups like "a w Krakowie?", the previous
# user turn) mentions what the tool is about; otherwise the small model's pick is noise (e.g.
# "kiedy była bitwa pod Grunwaldem" -> get_weather) and the turn is answered locally.
TOOL_CUES: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "get_time": r"godzin|czas|dat[aęy]|dzie[nń]\b|dni\b|dzi[sś]|kt[oó]ry\s+(?:dzi|jest)|ile\s+dni|ile\s+(?:zosta[lł]o\s+)?dni|jaki\s+dzie[nń]|time|date|\bday\b|today|weekday|clock",
        "get_weather": r"pogod|deszcz|pada|padać|padac|temperatur|stopni|s[lł]o[nń]c|wiatr|[sś]nieg|burz|mr[oó]z|zimno|ciep[lł]o|"
        r"prognoz|parasol|pochmurn|weather|rain|snow|sun|temperature|forecast|degrees|wind|cold|warm|hot",
        "get_sun": r"wsch[oó]d|zach[oó]d|d[lł]ugo[sś][cć]\s+dnia|kiedy\s+wschodzi|kiedy\s+zachodzi|ile\s+trwa\s+dzie[nń]|sunrise|sunset|daylight",
# Digits only: the selector garbles spelled-out numbers ("dwadzieścia plus trzydzieści pięć" -> 20 + 33).
        "calculate": r"\d",
        "convert_units": r"\d|kilo|metr|mil[eia]?\b|stop[ay]|cal[ei]?\b|litr|galon|funt|uncj|stopni|celsjusz|fahrenheit|kelvin|"
        r"szklank|łyżk|lyzk|bajt|bit|byte|przelicz|ile\s+\w+\s+(?:to|ma)|convert|pound|ounce|inch|feet|foot|mile|gallon|cup|tbsp|tsp",
        "exchange_rate": r"kurs|walut|euro|dolar|funt|frank|z[lł]ot|pln|eur\b|usd|gbp|chf|currency|exchange|dollar|pound|franc|zloty",
        "wiki_summary": r"kim\s+(?:jest|by[lł])|kto\s+to|co\s+to|opowiedz|wikipedi|podsumuj|who\s+(?:is|was)|what\s+is|summary|tell\s+me\s+about",
        "server_status": r"serwer|server|gpu|ram\b|pami[eę]ć|obci[aą][zż]|uptime|dysk|procesor|cpu|us[lł]ug|system|maszyn|disk|load|memory|"
        r"karta\s+graficzn|karty\s+graficzn",
        "next_holiday": r"[sś]wi[eę]t|wolne|wolny|holiday|urlop|ferie|d[lł]ugi\s+weekend|bo[zż]\w*\s+narodzeni\w*|gwiazdk\w*|wigili\w*|wielkanoc\w*|nowy\s+rok|nowego\s+roku|sylwest\w*|trzech\s+kr[oó]l\w*|bo[zż]\w*\s+cia[lł]\w*|wszystkich\s+[sś]wi[eę]t\w*|niepodleg[lł]\w*|konstytucj\w*|maj[oó]wk\w*",
        "manage_timer": r"minutnik|stoper|przypomn|budzik|ile\s+zosta[lł]o|anuluj\s+minutnik|timer|reminder",
        "remember": r"zapami[eę]taj|pami[eę]taj|remember|zanotuj",
        "forget": r"zapomnij|wykasuj|usu[nń]\s+z\s+pami[eę]ci|forget",
    }.items()
}
# Home sensors and devices are Hermes', even when the words look like weather ("temperatura w salonie").
FAST_PATH_TIME = re.compile(
    r"^\s*(?:hej\s+gaja[,\s]+|gaja[,\s]+)?(?:przepraszam\s*,?\s*)?"
    r"(?:kt[oó]ra\s+(?:jest\s+)?(?:teraz\s+)?godzina(?:\s+jest)?|"
    r"jaka\s+(?:jest\s+)?(?:teraz\s+)?godzina|"
    r"podaj\s+(?:mi\s+)?godzin[eę]|"
    r"jaki\s+(?:mamy\s+)?czas|"
    r"what\s+(?:time\s+is\s+it|is\s+the\s+time))\s*[?!.]*\s*$",
    re.IGNORECASE,
)

FAST_PATH_DATE = re.compile(
    r"^\s*(?:hej\s+gaja[,\s]+|gaja[,\s]+)?(?:przepraszam\s*,?\s*)?"
    r"(?:jaki\s+(?:jest\s+|mamy\s+)?(?:dzi[sś]|dzisiaj)?\s*(?:jest\s+)?dzie[nń](?:\s+tygodnia)?|"
    r"jaka\s+(?:jest\s+)?(?:dzi[sś]|dzisiaj)?\s*data|"
    r"kt[oó]ry\s+(?:(?:dzi[sś]|dzisiaj)\s+jest|(?:jest\s+|mamy\s+)?(?:dzi[sś]|dzisiaj)(?:\s+dzie[nń])?)|"
    r"kt[oó]rego\s+(?:dzi[sś]|dzisiaj\s+)?mamy|"
    r"what\s+(?:day\s+is\s+today|is\s+(?:today'?s\s+date|the\s+date)))\s*[?!.]*\s*$",
    re.IGNORECASE,
)


def clock_fast_path(utterance: str) -> ToolCall | None:
    if addresses_hermes(utterance):
        return None
    if FAST_PATH_TIME.match(utterance):
        return ToolCall("get_time", {"kind": "time"})
    if FAST_PATH_DATE.match(utterance):
        return ToolCall("get_time", {"kind": "date"})
    return None


AT_HOME = re.compile(r"\bw\s+(?:domu|salonie|sypialni|kuchni|pokoju|[lł]azience|gara[zż]u|biurze)\b|\b(?:in|at)\s+(?:the\s+)?(?:house|home|living\s+room|bedroom|kitchen)\b", re.IGNORECASE)


def plausible(call: ToolCall, utterance: str, history: list[dict[str, Any]]) -> bool:
    cue = TOOL_CUES.get(call.name)
    if cue is None:
        return True
    previous = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
    return bool(cue.search(utterance) or (len(utterance.split()) <= 4 and cue.search(str(previous))))


def any_plausible(names: set[str], utterance: str, history: list[dict[str, Any]]) -> bool:
    """Whether ``plausible`` could accept a call to any of ``names``; if not, selecting is wasted time."""
    return any(plausible(ToolCall(name, {}), utterance, history) for name in names)


Kind = Literal["local", "tool", "hermes"]
ROUTER_MESSAGE_CHARS = 240
SELECT_TOKENS = 64


def promises_handoff(text: str) -> bool:
    return bool(PROMISE_PL.search(text) or PROMISE_EN.search(text)) and not MYTHOLOGY.search(text)


def addresses_hermes(text: str) -> bool:
    """The user explicitly asks Hermes (the agent) to do something."""
    return bool(EXPLICIT_HERMES.search(text)) and not MYTHOLOGY.search(text)


def router_prefix(system: str) -> str:
    return f"<|im_start|>system\n{system}<|im_end|>\n"


def _short(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= ROUTER_MESSAGE_CHARS else text[: ROUTER_MESSAGE_CHARS - 1] + "…"


def router_context(system: str, history: list[dict[str, Any]]) -> str:
    """Everything of the router prompt before the new utterance: system, recent history, user header.

    Priming this (not just the system prefix) keeps the history tokens off the critical path; only
    the utterance itself is evaluated when the turn is classified.
    """
    parts = [router_prefix(system)]
    for message in history:
        if message["role"] == "user":
            parts.append(f"<|im_start|>user\nZapytanie: {_short(message['content'])}<|im_end|>\n")
        elif message["role"] == "assistant":
            parts.append(f"<|im_start|>assistant\n{_short(message['content'])}<|im_end|>\n")
    parts.append("<|im_start|>user\n")
    return "".join(parts)


def build_router_prompt(system: str, utterance: str, history: list[dict[str, Any]]) -> str:
    return f"{router_context(system, history)}Zapytanie: {utterance}<|im_end|>\n<|im_start|>assistant\nKategoria:"


def class_masses(completion: dict[str, Any]) -> tuple[float, float, float]:
    """Normalised probability mass of the first answer token: (LOCAL, TOOL, HERMES) by first letter."""
    candidates = (completion.get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
    hermes = local = tool = 0.0
    for candidate in candidates:
        token = str(candidate.get("token", "")).strip().lower()
        mass = math.exp(float(candidate.get("logprob", -99.0)))
        if token.startswith("h"):
            hermes += mass
        elif token.startswith("l"):
            local += mass
        elif token.startswith("t"):
            tool += mass
    total = hermes + local + tool
    if total <= 0:
        return 1.0, 0.0, 0.0
    return local / total, tool / total, hermes / total


def p_hermes(completion: dict[str, Any]) -> float:
    """Normalised probability mass of the first answer token starting with H (HERMES) vs L (LOCAL)."""
    local, _, hermes = class_masses(completion)
    total = hermes + local
    return hermes / total if total > 0 else 0.0


def tool_grammar(tools: list[dict[str, Any]]) -> str:
    """GBNF for one pythonic call `name(arg=value, ...)` restricted to the given tools and their parameters."""
    rules = ['root ::= " "? call', 'str ::= "\\"" [^"\\n]{0,60} "\\""', 'num ::= "-"? [0-9]{1,9} ("." [0-9]{1,6})?']
    alternatives: list[str] = []
    for index, tool in enumerate(tools):
        function = tool.get("function", tool)
        name = function["name"]
        params = list((function.get("parameters") or {}).get("properties") or {})
        alternatives.append(f"t{index}")
        if not params:
            rules.append(f't{index} ::= "{name}()"')
            continue
        properties = (function.get("parameters") or {}).get("properties") or {}
        arguments: list[str] = []
        for param in params:
            enum = properties[param].get("enum") if isinstance(properties[param], dict) else None
            if enum:
                values = " | ".join(f'"\\"{value}\\""' for value in enum)
                arguments.append(f'"{param}=" ({values})')
            else:
                arguments.append(f'"{param}=" (str | num)')
        rules.append(f"a{index} ::= " + " | ".join(f"({argument})" for argument in arguments))
        rules.append(f't{index} ::= "{name}(" (a{index} (", " a{index}){{0,4}})? ")"')
    rules.insert(1, "call ::= " + " | ".join(alternatives))
    return "\n".join(rules)


def parse_call(text: str, allowed: set[str]) -> ToolCall | None:
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in allowed:
        return None
    arguments: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None or not isinstance(keyword.value, ast.Constant):
            return None
        arguments[keyword.arg] = keyword.value.value
    return ToolCall(node.func.id, arguments)


def build_profile_prompt(utterance: str, options: list[tuple[str, str]]) -> str:
    listing = "\n".join(f"{i}. {label}: {desc or label}" for i, (label, desc) in enumerate(options, 1))
    system = (
        "Wybierasz, który asystent wykona zadanie. Odpowiedz tylko numerem.\n"
        f"{listing}"
    )
    return (
        f"{router_prefix(system)}<|im_start|>user\nZadanie: {_short(utterance)}<|im_end|>\n"
        "<|im_start|>assistant\nNumer:"
    )


def option_masses(completion: dict[str, Any], count: int) -> list[float]:
    """Normalised probability of each numbered option (1-based tokens "1".."9") for the first token."""
    candidates = (completion.get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
    masses = [0.0] * count
    for candidate in candidates:
        token = str(candidate.get("token", "")).strip()
        if token.isdigit() and 1 <= int(token) <= count:
            masses[int(token) - 1] += math.exp(float(candidate.get("logprob", -99.0)))
    total = sum(masses)
    return [m / total for m in masses] if total > 0 else [1.0 / count] * count


@dataclass(frozen=True)
class Route:
    kind: Kind
    p_local: float = 1.0
    p_tool: float = 0.0
    p_hermes: float = 0.0
    reason: str = "router"


class Router:
    """One 1-token logprob call on the pinned router slot decides LOCAL / TOOL / HERMES.

    With ``tools`` empty this is the 2-class Router v2 (TOOL mass, if any, counts as local).
    For TOOL, ``select_tool`` continues the very same prompt (``Kategoria: TOOL\\nWywołanie:``) with a
    grammar, so the call is a short constrained generation on the already cached prefix.
    """

    def __init__(
        self,
        llm: LLMClient,
        system: str,
        *,
        threshold: float,
        history_turns: int,
        timeout_s: float,
        slot: int = -1,
        tools: list[dict[str, Any]] | None = None,
        tool_threshold: float = 0.5,
    ) -> None:
        self._llm = llm
        self._system = system
        self.threshold = threshold
        self.tool_threshold = tool_threshold
        self._history_turns = history_turns
        self._timeout_s = timeout_s
        self._slot = slot
        self._tools = list(tools or [])
        self._tool_names = {t.get("function", t)["name"] for t in self._tools}
        self._grammar = tool_grammar(self._tools) if self._tools else ""

    @classmethod
    def from_file(cls, llm: LLMClient, path: Path, **kwargs: Any) -> Router:
        return cls(llm, load_text_block(path), **kwargs)

    @property
    def tools_enabled(self) -> bool:
        return bool(self._tools)

    def _recent(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return history[-self._history_turns :] if self._history_turns > 0 else []

    def _prompt(self, utterance: str, history: list[dict[str, Any]]) -> str:
        return build_router_prompt(self._system, utterance, self._recent(history))

    async def classify(self, utterance: str, history: list[dict[str, Any]]) -> Route:
        if addresses_hermes(utterance):
            return Route("hermes", 0.0, 0.0, 1.0, reason="explicit")
        if self._tools and "get_time" in self._tool_names and clock_fast_path(utterance):
            return Route("tool", 0.0, 1.0, 0.0, reason="fast-path")
        completion = await self._llm.complete(self._prompt(utterance, history), timeout_s=self._timeout_s, slot=self._slot)
        local, tool, hermes = class_masses(completion)
        if not self._tools:
            local, tool = local + tool, 0.0
        mythology = MYTHOLOGY.search(utterance)
        if hermes >= self.threshold and not mythology:
            return Route("hermes", local, tool, hermes)
        is_local_override = bool(
            re.search(r"\bminutnik|\bprzypomnij\s+mi\s+o\s+\d|\bzapami[eę]taj|\bzapomnij\b", utterance, re.IGNORECASE)
        )
        if self._tools and not mythology and not is_local_override and (HERMES_ACTION.search(utterance) or AT_HOME.search(utterance)):
            return Route("hermes", local, tool, hermes, reason="action")
        if self._tools and tool >= self.tool_threshold:
            return Route("tool", local, tool, hermes)
        return Route("local", local, tool, hermes)

    async def select_tool(self, utterance: str, history: list[dict[str, Any]]) -> ToolCall | None:
        if not self._tools:
            return None
        fast = clock_fast_path(utterance) if "get_time" in self._tool_names else None
        if fast is not None:
            return fast
        if not any_plausible(self._tool_names, utterance, self._recent(history)):
            return None
        prompt = self._prompt(utterance, history) + " TOOL\nWywołanie:"
        text = await self._llm.generate(
            prompt, n_predict=SELECT_TOKENS, grammar=self._grammar, timeout_s=self._timeout_s * 2, slot=self._slot
        )
        call = parse_call(text, self._tool_names)
        if call is not None and not plausible(call, utterance, self._recent(history)):
            return None
        return call

    async def pick_profile(self, utterance: str, options: list[tuple[str, str]]) -> tuple[int, float]:
        """Which of the person's Hermes profiles (label, description) fits the task: (index, probability).

        One 1-token logprob call like ``classify``; the router slot is re-primed after the turn anyway.
        """
        prompt = build_profile_prompt(utterance, options)
        completion = await self._llm.complete(prompt, timeout_s=self._timeout_s, slot=self._slot)
        masses = option_masses(completion, len(options))
        best = max(range(len(options)), key=masses.__getitem__)
        return best, masses[best]

    async def p_hermes(self, utterance: str, history: list[dict[str, Any]]) -> float:
        return (await self.classify(utterance, history)).p_hermes

    async def prime(self, history: list[dict[str, Any]] | None = None) -> None:
        """Cache the prompt up to the next utterance; ``history`` is the chat context without system."""
        prompt = router_context(self._system, self._recent(history or []))
        await self._llm.prime(prompt, timeout_s=self._timeout_s, slot=self._slot)

    def hands_off(self, probability: float) -> bool:
        return probability >= self.threshold
