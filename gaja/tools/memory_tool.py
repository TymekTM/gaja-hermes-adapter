from __future__ import annotations

import re
from typing import Any

from gaja.memory import MemoryStore, get_memory_store, set_memory_store
from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import ToolResult

FIRST_TO_THIRD_VERBS_PL = [
    (r"\bnie\s+lubi[eę]\b", "nie lubi", "nie lubisz"),
    (r"\blubi[eę]\b", "lubi", "lubisz"),
    (r"\buwielbiam\b", "uwielbia", "uwielbiasz"),
    (r"\bwol[eę]\b", "woli", "wolisz"),
    (r"\bnie\s+znosz[eę]\b", "nie znosi", "nie znosisz"),
    (r"\bnie\s+mam\b", "nie ma", "nie masz"),
    (r"\bmam\b", "ma", "masz"),
    (r"\bmieszkam\b", "mieszka", "mieszkasz"),
    (r"\bpochodz[eę]\b", "pochodzi", "pochodzisz"),
    (r"\bpracuj[eę]\b", "pracuje", "pracujesz"),
    (r"\bucz[eę]\s+si[eę]\b", "uczy się", "uczysz się"),
    (r"\bstudiuj[eę]\b", "studiuje", "studiujesz"),
    (r"\bnazywam\s+si[eę]\b", "nazywa się", "nazywasz się"),
    (r"\bjestem\b", "jest", "jesteś"),
    (r"\bnie\s+jestem\b", "nie jest", "nie jesteś"),
    (r"\binteresuj[eę]\s+si[eę]\b", "interesuje się", "interesujesz się"),
    (r"\bpij[eę]\b", "pije", "pijesz"),
    (r"\bnie\s+pij[eę]\b", "nie pije", "nie pijesz"),
    (r"\bjem\b", "je", "jesz"),
    (r"\bjadam\b", "jada", "jadasz"),
    (r"\bchc[eę]\b", "chce", "chcesz"),
    (r"\bpotrzebuj[eę]\b", "potrzebuje", "potrzebujesz"),
    (r"\bje[zż]d[zż][eę]\b", "jeździ", "jeździsz"),
    (r"\bchodz[eę]\b", "chodzi", "chodzisz"),
    (r"\bgram\b", "gra", "grasz"),
    (r"\bposiadam\b", "posiada", "posiadasz"),
    (r"\bwstaj[eę]\b", "wstaje", "wstajesz"),
    (r"\bczuj[eę]\s+si[eę]\b", "czuje się", "czujesz się"),
]

POSSESSIVES_PL = [
    (r"\bmojego\b", "użytkownika", "twojego"),
    (r"\bmojemu\b", "użytkownikowi", "twojemu"),
    (r"\bmoim\b", "użytkownika", "twoim"),
    (r"\bmoj[a]\b", "użytkownika", "twoja"),
    (r"\bmoj[ą]\b", "użytkownika", "twoją"),
    (r"\bmoj[e]\b", "użytkownika", "twoje"),
    (r"\bm[oó]j\b", "użytkownika", "twój"),
    (r"\bmoi\b", "użytkownika", "twoi"),
    (r"\bmoich\b", "użytkownika", "twoich"),
]


def normalize_memory_fact_pl(fact: str) -> tuple[str, str]:
    text = fact.strip()
    text = re.sub(r"^(?:że|ze)\s+", "", text, flags=re.I)

    # If already normalized to 3rd person about user:
    if re.search(r"^\s*u[zż]ytkownik\b", text, re.I):
        conf_tail = re.sub(r"^\s*u[zż]ytkownik\s+", "", text, flags=re.I)
        for _, third, second in FIRST_TO_THIRD_VERBS_PL:
            if re.search(rf"\b{third}\b", conf_tail, re.I):
                conf_tail = re.sub(rf"\b{third}\b", second, conf_tail, flags=re.I)
                break
        return text, f"Zapamiętałam, że {conf_tail}."

    stored = text
    confirm = text

    # Possessive noun phrase at the beginning: "moja siostra ma na imię Ola"
    m_poss = re.match(r"^(moj[aąe]|m[oó]j|moi|moich)\s+(\w+)(.*)", stored, re.I)
    if m_poss:
        poss, noun, rest = m_poss.group(1), m_poss.group(2), m_poss.group(3)
        stored_phrase = f"{noun.capitalize()} użytkownika{rest}"
        tw_form = "twoja"
        for pat, _, tf in POSSESSIVES_PL:
            if re.match(pat, poss, re.I):
                tw_form = tf
                break
        return stored_phrase, f"Zapamiętałam, że {tw_form} {noun}{rest}."

    has_verb = False
    for pat, third, second in FIRST_TO_THIRD_VERBS_PL:
        if re.search(pat, stored, flags=re.I):
            stored = re.sub(pat, third, stored, flags=re.I)
            confirm = re.sub(pat, second, confirm, flags=re.I)
            has_verb = True
            break

    for pat, u_form, tw_form in POSSESSIVES_PL:
        stored = re.sub(pat, u_form, stored, flags=re.I)
        confirm = re.sub(pat, tw_form, confirm, flags=re.I)

    stored = re.sub(r"^\s*ja\s+", "", stored, flags=re.I)
    confirm = re.sub(r"^\s*ja\s+", "", confirm, flags=re.I)

    if has_verb:
        return f"Użytkownik {stored}", f"Zapamiętałam, że {confirm}."

    # Not a first-person statement (e.g. "Kasia lubi kawę"): store it as said.
    return stored[:1].upper() + stored[1:], f"Zapamiętałam: {confirm}."


def normalize_memory_fact_en(fact: str) -> tuple[str, str]:
    text = fact.strip()
    text = re.sub(r"^that\s+", "", text, flags=re.I)

    if re.search(r"^\s*user\b", text, re.I):
        return text, f"I remembered: {text}."

    if re.match(r"^i\s+like\b", text, re.I):
        rest = re.sub(r"^i\s+like\s+", "", text, flags=re.I)
        return f"User likes {rest}", f"I remembered that you like {rest}."

    if re.match(r"^i\s+have\b", text, re.I):
        rest = re.sub(r"^i\s+have\s+", "", text, flags=re.I)
        return f"User has {rest}", f"I remembered that you have {rest}."

    if re.match(r"^i\s+am\b", text, re.I):
        rest = re.sub(r"^i\s+am\s+", "", text, flags=re.I)
        return f"User is {rest}", f"I remembered that you are {rest}."

    if re.match(r"^my\s+", text, re.I):
        rest = re.sub(r"^my\s+", "", text, flags=re.I)
        return f"User's {rest}", f"I remembered that your {rest}."

    return f"User: {text}", f"I remembered: {text}."


def normalize_memory_fact(fact: str, lang: str = "pl") -> tuple[str, str]:
    if lang == "en":
        return normalize_memory_fact_en(fact)
    return normalize_memory_fact_pl(fact)


async def remember(fact: str, lang: str = "pl") -> ToolResult:
    """Store a personal fact about the user or their preferences in local memory."""
    store = get_memory_store()
    text = fact.strip()
    if not text:
        err = "Brak treści do zapamiętania"
        return ToolResult(data={"error": err}, speak="Nie podałeś, co mam zapamiętać.", error=err)

    stored_fact, raw_speak = normalize_memory_fact(text, lang)
    created = store.add_fact(stored_fact)
    speak = normalize_for_tts(raw_speak, lang)
    return ToolResult(data=created, speak=speak)


async def forget(fact: str, lang: str = "pl") -> ToolResult:
    """Remove a personal fact from local memory."""
    store = get_memory_store()
    text = fact.strip()
    if not text:
        err = "Brak treści do usunięcia"
        return ToolResult(data={"error": err}, speak="Nie podałeś, co mam zapomnieć.", error=err)

    deleted = store.forget_by_query(text)
    if deleted:
        if lang == "en":
            raw_speak = "I forgot that."
        else:
            raw_speak = "Zapomniałam o tym."
    else:
        if lang == "en":
            raw_speak = "I could not find that fact in memory."
        else:
            raw_speak = "Nie znalazłam takiego faktu w pamięci."

    speak = normalize_for_tts(raw_speak, lang)
    return ToolResult(data={"deleted": [f["id"] for f in deleted]}, speak=speak)
