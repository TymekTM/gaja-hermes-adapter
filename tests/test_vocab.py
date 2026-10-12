"""Tests for custom vocabulary, post-STT correction and vocabulary HTTP API (Part B)."""
from __future__ import annotations

import json
from pathlib import Path
from starlette.testclient import TestClient

from gaja.config import Settings
from gaja.server import create_app, origin_allowed
from gaja.stt.vocab import VocabularyManager, VocabTerm, VocabSuggestion, create_vocab_router


def test_default_vocab_loads(tmp_path: Path) -> None:
    default_toml = Path("prompts/vocab_default.toml")
    vm = VocabularyManager(
        default_path=default_toml,
        user_path=tmp_path / "user_vocab.toml",
        suggestions_path=tmp_path / "suggestions.json",
    )
    terms = {t.term: t for t in vm.all_terms}
    assert "Kubernetes" in terms
    assert "Docker" in terms
    assert "Grafana" in terms
    assert "Gaja" in terms
    assert terms["Kubernetes"].builtin is True
    assert "kubernetis" in terms["Kubernetes"].aliases


def test_vocab_user_file_save_and_load(tmp_path: Path) -> None:
    user_file = tmp_path / "vocab.toml"
    sug_file = tmp_path / "suggestions.json"
    vm = VocabularyManager(
        default_path=Path("prompts/vocab_default.toml"),
        user_path=user_file,
        suggestions_path=sug_file,
    )

    # Put a custom user term
    res = vm.put_terms([{"term": "OpenObserve", "aliases": ["open observe"], "lang": "en"}])
    terms = {t["term"]: t for t in res["terms"]}
    assert "OpenObserve" in terms
    assert terms["OpenObserve"].get("builtin") is not True

    # Reload from disk
    vm2 = VocabularyManager(
        default_path=Path("prompts/vocab_default.toml"),
        user_path=user_file,
        suggestions_path=sug_file,
    )
    terms2 = {t.term: t for t in vm2.all_terms}
    assert "OpenObserve" in terms2
    assert terms2["OpenObserve"].builtin is False
    assert terms2["Kubernetes"].builtin is True


def test_vocab_hotwords(tmp_path: Path) -> None:
    vm = VocabularyManager(
        default_path=Path("prompts/vocab_default.toml"),
        user_path=tmp_path / "user_vocab.toml",
        suggestions_path=tmp_path / "suggestions.json",
        max_hotwords=5,
    )
    hw = vm.hotwords()
    items = [x.strip() for x in hw.split(",")]
    assert len(items) <= 5
    assert "Gaja" in items


def test_post_stt_correction_exact_and_alias(tmp_path: Path) -> None:
    vm = VocabularyManager(
        default_path=Path("prompts/vocab_default.toml"),
        user_path=tmp_path / "user_vocab.toml",
        suggestions_path=tmp_path / "suggestions.json",
    )
    assert vm.correct("odpal mi kubernetis") == "odpal mi Kubernetes"
    assert vm.correct("zainstaluj doker") == "zainstaluj Docker"
    assert vm.correct("zainstaluj tejlskil") == "zainstaluj Tailscale"
    assert vm.correct("uruchom homeassistant") == "uruchom Home Assistant"


def test_post_stt_correction_polish_inflections(tmp_path: Path) -> None:
    vm = VocabularyManager(
        default_path=Path("prompts/vocab_default.toml"),
        user_path=tmp_path / "user_vocab.toml",
        suggestions_path=tmp_path / "suggestions.json",
    )
    # Inflected terms
    assert vm.correct("odpal deploy na kubernetesa") == "odpal deploy na Kubernetesa"
    assert vm.correct("pracuję z kubernetesem") == "pracuję z Kubernetesem"
    assert vm.correct("zarządzaj dokerem") == "zarządzaj Dockerem"
    assert vm.correct("w kontenerze dockerze") == "w kontenerze Dockerze"
    assert vm.correct("sprawdź logi w grafanie") == "sprawdź logi w Grafanie"
    assert vm.correct("wykresy z grafany") == "wykresy z Grafany"
    assert vm.correct("zrób git pusha") == "zrób Git pusha"
    assert vm.correct("utwórz pull requesta") == "utwórz pull requesta"

    # Inflected aliases
    assert vm.correct("odpal kubernetisa") == "odpal Kubernetesa"
    assert vm.correct("zarządzaj dokerem") == "zarządzaj Dockerem"


def test_post_stt_correction_negative_examples(tmp_path: Path) -> None:
    """Must never touch short common Polish words or unrelated natural sentences."""
    vm = VocabularyManager(
        default_path=Path("prompts/vocab_default.toml"),
        user_path=tmp_path / "user_vocab.toml",
        suggestions_path=tmp_path / "suggestions.json",
    )
    negatives = [
        "nie ma go w domu",
        "to jest bardzo dobre",
        "pójdę do sklepu po chleb",
        "jaki jest dzisiaj dzień",
        "kiedy będzie obiad",
        "zadzwoń do mamy",
        "co tam u ciebie",
        "włącz światło w salonie",
        "wyłącz telewizor w kuchni",
        "czy jutro będzie padać deszcz",
        "zrób mi kawę z mlekiem",
        "ile to jest dwa plus dwa",
    ]
    for sent in negatives:
        assert vm.correct(sent) == sent


def test_conservative_fuzzy_matching(tmp_path: Path) -> None:
    sug_file = tmp_path / "suggestions.json"
    vm = VocabularyManager(
        default_path=Path("prompts/vocab_default.toml"),
        user_path=tmp_path / "user_vocab.toml",
        suggestions_path=sug_file,
        fuzzy=True,
    )
    # Near-miss on long word: 'kubernetess' (1 char insertion) -> Kubernetes
    corrected = vm.correct("odpal kubernetess")
    assert corrected == "odpal Kubernetes"

    # Near-miss should record suggestion
    assert len(vm._suggestions) == 1
    assert vm._suggestions[0].term == "Kubernetes"
    assert "kubernetess" in vm._suggestions[0].heard
    assert vm._suggestions[0].reason == "near-miss"


def test_auto_learning_suggestions(tmp_path: Path) -> None:
    sug_file = tmp_path / "suggestions.json"
    vm = VocabularyManager(
        default_path=Path("prompts/vocab_default.toml"),
        user_path=tmp_path / "user_vocab.toml",
        suggestions_path=sug_file,
    )

    # User correction: "chodziło o X"
    vm.observe("chodziło o ArgoCD", previous_turn_text="odpal argocd")
    assert any(s.term == "ArgoCD" for s in vm._suggestions)

    # Unknown token
    vm.observe("zainstaluj ClickHouse na serwerze")
    assert any(s.term == "ClickHouse" for s in vm._suggestions)


def test_vocab_http_api(tmp_path: Path) -> None:
    user_file = tmp_path / "vocab.toml"
    sug_file = tmp_path / "suggestions.json"

    # Pre-populate one suggestion
    sug_file.write_text(
        json.dumps([{"id": "s1", "term": "ArgoCD", "heard": ["argocd"], "count": 2, "reason": "near-miss"}]),
        encoding="utf-8",
    )

    vm = VocabularyManager(
        default_path=Path("prompts/vocab_default.toml"),
        user_path=user_file,
        suggestions_path=sug_file,
    )

    allowed = ["http://127.0.0.1:8790", "https://localhost:8790"]
    router = create_vocab_router(vm, lambda o: origin_allowed(o, allowed))

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # 1. GET /api/vocab
    resp = client.get("/api/vocab")
    assert resp.status_code == 200
    data = resp.json()
    assert "terms" in data and "suggestions" in data
    assert any(t["term"] == "Kubernetes" and t.get("builtin") is True for t in data["terms"])
    assert len(data["suggestions"]) == 1
    assert data["suggestions"][0]["id"] == "s1"
    assert data["suggestions"][0]["term"] == "ArgoCD"

    # 2. PUT /api/vocab (replaces user terms)
    put_resp = client.put("/api/vocab", json={"terms": [{"term": "Traefik", "aliases": ["trafik"], "lang": "en"}]})
    assert put_resp.status_code == 200
    put_data = put_resp.json()
    assert any(t["term"] == "Traefik" and t.get("builtin") is not True for t in put_data["terms"])
    # Built-in terms still present
    assert any(t["term"] == "Kubernetes" and t.get("builtin") is True for t in put_data["terms"])

    # 3. POST /api/vocab/suggestions/s1/approve
    appr_resp = client.post("/api/vocab/suggestions/s1/approve")
    assert appr_resp.status_code == 200
    appr_data = appr_resp.json()
    assert len(appr_data["suggestions"]) == 0
    assert any(t["term"] == "ArgoCD" for t in appr_data["terms"])

    # 4. POST /api/vocab/suggestions/nonexistent/approve -> 404
    err_resp = client.post("/api/vocab/suggestions/nonexistent/approve")
    assert err_resp.status_code == 404

    # 5. Add another suggestion and test reject
    vm._add_suggestion("BadTerm", "badterm", "near-miss")
    s_id = vm._suggestions[0].id
    rej_resp = client.post(f"/api/vocab/suggestions/{s_id}/reject")
    assert rej_resp.status_code == 200
    rej_data = rej_resp.json()
    assert not any(s["id"] == s_id for s in rej_data["suggestions"])

    # 6. Origin check
    # Valid origin
    ok_orig = client.get("/api/vocab", headers={"Origin": "http://127.0.0.1:8790"})
    assert ok_orig.status_code == 200

    # Invalid origin
    bad_orig = client.get("/api/vocab", headers={"Origin": "http://evil.com"})
    assert bad_orig.status_code == 403



def test_observe_sentence_initial_and_filtering(tmp_path: Path) -> None:
    sug_file = tmp_path / "suggestions.json"
    vm = VocabularyManager(
        default_path=Path("prompts/vocab_default.toml"),
        user_path=tmp_path / "user_vocab.toml",
        suggestions_path=sug_file,
    )

    # Sentence-initial capitalized words from Whisper must be ignored
    vm.observe("Jaka jest dzisiaj pogoda")
    vm.observe("Włącz światło w salonie")
    vm.observe("Powiedz jaki mamy dzień. Sprawdź to dokładnie.")
    assert len(vm._suggestions) == 0

    # Non-initial technical term with mixed case or symbol is captured
    vm.observe("chcę wdrożyć ClickHouse na produkcji")
    assert any(s.term == "ClickHouse" for s in vm._suggestions)

    # Latin word seen only once is not suggested
    vm.observe("użyj edytora Neovim")
    assert not any(s.term == "Neovim" for s in vm._suggestions)
    # Seen 3 times across turns -> suggested
    vm.observe("znowu ten Neovim")
    vm.observe("otwórz plik w Neovim")
    assert any(s.term == "Neovim" for s in vm._suggestions)


def test_observe_repetition_acoustic_and_time_guard(tmp_path: Path) -> None:
    sug_file = tmp_path / "suggestions.json"
    vm = VocabularyManager(
        default_path=Path("prompts/vocab_default.toml"),
        user_path=tmp_path / "user_vocab.toml",
        suggestions_path=sug_file,
    )

    # Unrelated sentences with 1 word diff (Krakowie vs Warszawie) -> ratio < 0.5 -> ignored
    vm.observe("pogoda w Warszawie", previous_turn_text="pogoda w Krakowie", timestamp=10.0, previous_turn_time=5.0)
    assert not any(s.term == "Warszawie" for s in vm._suggestions)

    # Acoustically similar correction (kubernetisa vs kubernetes) within 30s -> accepted
    vm.observe("odpal kubernetes", previous_turn_text="odpal kubernetisa", timestamp=15.0, previous_turn_time=10.0)
    assert any(s.term == "Kubernetes" for s in vm._suggestions)

    # Stale turns (>30s apart) -> ignored
    vm.observe("odpal kubernetes", previous_turn_text="odpal kubernetisa", timestamp=100.0, previous_turn_time=10.0)
    # Should not add another or update if empty
