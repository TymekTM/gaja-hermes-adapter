from __future__ import annotations

from pathlib import Path
import pytest

from gaja.memory import MemoryStore, get_memory_store, set_memory_store
from gaja.pipeline import Services, SystemPrompt
from gaja.config import Settings
from gaja.tools.memory_tool import remember, forget


def test_memory_store_persistence(tmp_path: Path):
    mem_file = tmp_path / "memory.json"
    store1 = MemoryStore(mem_file)
    f1 = store1.add_fact("Moja siostra ma na imię Ola")
    f2 = store1.add_fact("Lubię kawę bez cukru")

    assert len(store1.list_facts()) == 2
    assert mem_file.exists()

    # Reload from file
    store2 = MemoryStore(mem_file)
    facts = store2.list_facts()
    assert len(facts) == 2
    assert any("Ola" in f["text"] for f in facts)
    assert any("kawę" in f["text"] for f in facts)


def test_memory_store_delete_and_forget(tmp_path: Path):
    mem_file = tmp_path / "memory.json"
    store = MemoryStore(mem_file)
    f1 = store.add_fact("Moja siostra ma na imię Ola")
    f2 = store.add_fact("Mój pies wabi się Burek")

    # Forget by query keywords
    forgotten = store.forget_by_query("siostry")
    assert len(forgotten) == 1
    assert forgotten[0]["id"] == f1["id"]
    assert len(store.list_facts()) == 1

    # Delete by explicit id
    assert store.delete_fact(f2["id"]) is True
    assert len(store.list_facts()) == 0
    assert store.delete_fact("missing") is False


def test_memory_store_format_prompt_block(tmp_path: Path):
    mem_file = tmp_path / "memory.json"
    store = MemoryStore(mem_file)
    assert store.format_prompt_block() == ""

    store.add_fact("Fakt numer jeden")
    block = store.format_prompt_block()
    assert "Fakty o rozmówcy:" in block
    assert "- Fakt numer jeden" in block

    # Capping at max_chars
    for i in range(50):
        store.add_fact(f"Długi fakt powtarzany wielokrotnie w pętli numer {i}")
    capped_block = store.format_prompt_block(max_chars=200)
    assert len(capped_block) <= 200


@pytest.mark.asyncio
async def test_remember_and_forget_tools(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.json")
    set_memory_store(store)

    res_rem = await remember("Moja mama ma urodziny 5 maja")
    assert res_rem.error is None
    assert "Zapamiętałam" in res_rem.speak
    assert len(store.list_facts()) == 1

    res_forg = await forget("mama")
    assert res_forg.error is None
    assert "Zapomniałam" in res_forg.speak
    assert len(store.list_facts()) == 0


def test_services_system_prompt_injection(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.json")
    store.add_fact("Użytkownik ma na imię Ala")

    base_prompt = "Jesteś Gaja, asystentka głosowa."
    services = Services(
        cfg=Settings(),
        stt=None,
        tts=None,
        llm=None,
        prompt=SystemPrompt(text=base_prompt, tools=[]),
        hermes=None,
        metrics=None,
        new_vad=None,
        memory_store=store,
    )
    full_prompt = services.get_system_prompt()
    assert full_prompt.startswith(base_prompt)
    assert "Fakty o rozmówcy:" in full_prompt
    assert "- Użytkownik ma na imię Ala" in full_prompt


def test_memory_store_on_change_updates_live_session(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.json")
    base_prompt = "Jesteś Gaja."
    services = Services(
        cfg=Settings(),
        stt=None,
        tts=None,
        llm=None,
        prompt=SystemPrompt(text=base_prompt, tools=[]),
        hermes=None,
        metrics=None,
        new_vad=None,
        memory_store=store,
    )
    class FakeOutbox:
        async def send_json(self, p): pass
        async def send_bytes(self, b): pass

    from gaja.pipeline import Session
    session = Session(services, FakeOutbox())
    store.on_change = lambda: [s.update_system_prompt() for s in services.sessions]

    msgs = session._history.messages()
    assert "Fakty o rozmówcy:" not in msgs[0]["content"]

    store.add_fact("Użytkownik lubi jazz")
    msgs2 = session._history.messages()
    assert "Fakty o rozmówcy:" in msgs2[0]["content"]
    assert "- Użytkownik lubi jazz" in msgs2[0]["content"]


def test_normalize_memory_fact_person_and_confirmation():
    from gaja.tools.memory_tool import normalize_memory_fact

    st1, conf1 = normalize_memory_fact("lubię herbatę z cytryną", "pl")
    assert st1 == "Użytkownik lubi herbatę z cytryną"
    assert conf1 == "Zapamiętałam, że lubisz herbatę z cytryną."

    st2, conf2 = normalize_memory_fact("moja siostra ma na imię Ola", "pl")
    assert st2 == "Siostra użytkownika ma na imię Ola"
    assert "twoja siostra" in conf2

    st3, conf3 = normalize_memory_fact("mieszkam w Krakowie", "pl")
    assert st3 == "Użytkownik mieszka w Krakowie"
    assert conf3 == "Zapamiętałam, że mieszkasz w Krakowie."

    st4, conf4 = normalize_memory_fact("Użytkownik lubi herbatę z cytryną", "pl")
    assert st4 == "Użytkownik lubi herbatę z cytryną"
    assert conf4 == "Zapamiętałam, że lubisz herbatę z cytryną."


@pytest.mark.asyncio
async def test_remember_tool_normalizes_to_third_person(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.json")
    set_memory_store(store)

    res = await remember("lubię herbatę z cytryną")
    assert res.error is None
    assert res.speak == "Zapamiętałam, że lubisz herbatę z cytryną."
    facts = store.list_facts()
    assert len(facts) == 1
    assert facts[0]["text"] == "Użytkownik lubi herbatę z cytryną"
