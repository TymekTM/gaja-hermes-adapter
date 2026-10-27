from pathlib import Path
import tempfile
import os
import pytest

import gaja.config as config_mod
from gaja.config import Settings, ServerConfig, HermesConfig, TTSConfig, VADConfig


@pytest.fixture(autouse=True)
def no_host_local_config(monkeypatch, tmp_path) -> None:
    """Ignore the checkout's own config.local.toml so the defaults are tested."""
    monkeypatch.setattr(config_mod, "LOCAL_CONFIG_PATH", tmp_path / "absent.toml")


def test_config_defaults() -> None:
    settings = Settings()
    assert settings.server.port == 8790
    assert settings.hermes.command == ["hermes", "acp"]
    assert all(".ts.net" not in o for o in settings.server.allowed_origins)


def test_spotify_and_home_config_defaults() -> None:
    settings = Settings()
    assert settings.spotify.enabled is True
    assert settings.spotify.token_path == Path("data/spotify_token.json")
    assert settings.spotify.device_name == ""
    assert settings.spotify.timeout_s == pytest.approx(3.0)
    assert settings.spotify.volume_step == 15
    assert settings.spotify.market == "PL"

    # NEW-6: no household data in the repo. Defaults are empty: no aliases, no writes at all.
    assert settings.home.enabled is True
    assert settings.home.timeout_s == pytest.approx(3.0)
    assert settings.home.household_file == Path("data/home.toml")
    assert settings.home.write_allowlist == []
    assert settings.home.aliases == {}
    assert settings.home.spoken_forms == {}
    assert settings.home.boiler_id_substrings == []


def _household_file(tmpdir: str) -> Path:
    path = Path(tmpdir) / "home.toml"
    path.write_text(
        """
write_allowlist = ["climate.salon", "media_player.*"]
boiler_id_substrings = ["boiler"]

[aliases]
"klima w salonie" = "climate.salon"
"u Ani" = "climate.ania"

[spoken_forms]
"climate.salon" = ["klima w salonie", "klimę w salonie", "klimy w salonie"]
""",
        encoding="utf-8",
    )
    return path


def test_home_household_file_merges_over_empty_defaults(tmp_path) -> None:
    """NEW-6: the real household data (aliases, write allowlist, spoken forms, boiler markers)
    loads from [home] household_file at Settings construction; the repo keeps none of it."""
    settings = Settings(home={"household_file": _household_file(str(tmp_path))})
    assert settings.home.aliases == {"klima w salonie": "climate.salon", "u Ani": "climate.ania"}
    assert settings.home.write_allowlist == ["climate.salon", "media_player.*"]
    assert settings.home.spoken_forms == {"climate.salon": ("klima w salonie", "klimę w salonie", "klimy w salonie")}
    assert settings.home.boiler_id_substrings == ["boiler"]
    # An empty allowlist means no writes: wildcards unlock nothing that is not listed.
    from gaja.tools.home_tool import is_write_allowed

    empty = Settings()
    assert not is_write_allowed("climate.salon", "climate", "set_temperature", empty.home.write_allowlist)


def test_home_household_file_missing_or_invalid_falls_back_to_no_writes(tmp_path) -> None:
    missing = Settings(home={"household_file": tmp_path / "absent.toml"})
    assert missing.home.aliases == {} and missing.home.write_allowlist == []

    broken = tmp_path / "broken.toml"
    broken.write_text("[aliases]\n\"klima w salonie\" = [not toml", encoding="utf-8")
    invalid = Settings(home={"household_file": broken})
    assert invalid.home.aliases == {} and invalid.home.write_allowlist == []


def test_home_example_toml_loads_verbatim_with_markers_and_allowlist() -> None:
    """H3/M1a: docs/home.example.toml had boiler_id_substrings after [spoken_forms], so TOML
    parsed it as a spoken form and the boiler had no marker. Loaded verbatim through the real
    loader the file must yield the boiler markers and the write allowlist it documents."""
    settings = Settings(home={"household_file": Path("docs/home.example.toml")})
    assert settings.home.boiler_id_substrings == ["boiler"]
    assert settings.home.write_allowlist == ["climate.salon", "climate.sypialnia", "climate.ania", "media_player.*"]
    assert settings.home.aliases["klima w salonie"] == "climate.salon"
    assert settings.home.spoken_forms["climate.salon"] == ("klima w salonie", "klimę w salonie", "klimy w salonie")


def test_home_household_markers_are_normalised_and_empties_dropped(tmp_path, caplog) -> None:
    """H3/M1b: markers are matched case/whitespace-insensitively ('Boiler', ' boiler') and an
    empty marker is dropped with a warning instead of silently never matching."""
    household = tmp_path / "home.toml"
    household.write_text(
        'boiler_id_substrings = ["Boiler", " boiler ", "", "Vitodens"]\n',
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="gaja.config"):
        settings = Settings(home={"household_file": household})
    assert settings.home.boiler_id_substrings == ["boiler", "vitodens"]
    assert "empty boiler_id_substrings" in caplog.text


def test_home_household_misplaced_keys_fail_closed(tmp_path, caplog) -> None:
    """H3/M1c: boiler_id_substrings/write_allowlist nested inside a table (or a non-string alias
    value) warn by key and empty the write allowlist for the load — the boiler never ends up
    writable because the file was mis-edited."""
    household = tmp_path / "home.toml"
    household.write_text(
        """
write_allowlist = ["climate.salon"]

[aliases]
"klima w salonie" = "climate.salon"
"zla wartosc" = ["climate.inaczej"]

[spoken_forms]
boiler_id_substrings = ["boiler"]
""",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="gaja.config"):
        settings = Settings(home={"household_file": household})
    assert settings.home.write_allowlist == []  # fail closed: no writes from this load
    assert settings.home.boiler_id_substrings == []  # the misplaced marker key never loads
    assert settings.home.aliases == {"klima w salonie": "climate.salon"}
    assert "spoken_forms.boiler_id_substrings" in caplog.text
    assert "aliases.zla wartosc" in caplog.text
    # Warnings name the misplaced keys, never their values.
    assert "climate.inaczej" not in caplog.text
    assert "climate.salon" not in caplog.text


def test_home_household_misplaced_key_clears_a_config_seeded_allowlist(tmp_path, caplog) -> None:
    """Floor pin for the fail-closed branch (H3/M1c round 5): the write allowlist may come from
    [home] config itself, and a misplaced household key ('write_allowlist' hidden under a table,
    i.e. a mis-edit while saving the file) must clear THAT list too — never merge on top of it."""
    household = tmp_path / "home.toml"
    household.write_text(
        """
[spoken_forms]
write_allowlist = ["media_player.kuchnia"]

[aliases]
"klima w salonie" = "climate.salon"
""",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="gaja.config"):
        settings = Settings(
            home={
                "household_file": household,
                "write_allowlist": ["climate.salon", "media_player.*"],
            }
        )
    assert settings.home.write_allowlist == []  # the config-seeded list is cleared for this load
    assert settings.home.aliases == {"klima w salonie": "climate.salon"}
    assert "spoken_forms.write_allowlist" in caplog.text


def test_home_household_dropped_and_misplaced_keys_warn_once_without_values(tmp_path, caplog) -> None:
    """L1: merge_home_household logs a warning once per dropped or misplaced key, never logging values."""
    household = tmp_path / "home.toml"
    household.write_text(
        """
extra_top_key = "secret_top_value"

[devices]
tv = "secret_device_value"

[aliases]
"klima" = "climate.salon"

[spoken_forms]
bad_form = "secret_form_string"
"climate.salon" = ["klima w salonie"]
""",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="gaja.config"):
        settings = Settings(home={"household_file": household})
    assert settings.home.aliases == {"klima": "climate.salon"}
    # Dropped keys are warned
    assert "extra_top_key" in caplog.text
    assert "devices" in caplog.text
    assert "spoken_forms.bad_form" in caplog.text
    # Values must never be logged
    assert "secret_top_value" not in caplog.text
    assert "secret_device_value" not in caplog.text
    assert "secret_form_string" not in caplog.text
    # Logged once per key
    assert caplog.text.count("'extra_top_key'") == 1
    assert caplog.text.count("'devices'") == 1
    assert caplog.text.count("'spoken_forms.bad_form'") == 1


def test_config_local_override(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        local_cfg = Path(tmpdir) / "config.local.toml"
        local_cfg.write_text("""
[server]
port = 9999
allowed_origins = ["https://custom.example.com"]

[hermes]
command = ["/custom/bin/hermes", "acp"]
""")
        monkeypatch.setenv("GAJA_LOCAL_CONFIG", str(local_cfg))
        import gaja.config as config_mod
        monkeypatch.setattr(config_mod, "LOCAL_CONFIG_PATH", local_cfg)

        settings = Settings()
        assert settings.server.port == 9999
        assert settings.server.allowed_origins == ["https://custom.example.com"]
        assert settings.hermes.command == ["/custom/bin/hermes", "acp"]
        # Non-overridden fields should retain defaults
        assert settings.server.host == "127.0.0.1"


def test_config_env_override(monkeypatch) -> None:
    monkeypatch.setenv("GAJA_SERVER__PORT", "8888")
    settings = Settings()
    assert settings.server.port == 8888


def test_wakeword_default_cap_holds_days_of_continuous_recording() -> None:
    # the recorder's "Nagrywanie ciągłe" writes ~115 MB per hour; the default cap must cover at least 10 days of it
    assert Settings().wakeword.max_total_mb >= 10 * 24 * 115


def test_vad_backend_default_and_override() -> None:
    assert Settings().vad.backend == "onnx"
    settings = Settings(vad={"backend": "torch"})
    assert settings.vad.backend == "torch"


def test_tts_preload_languages_default_and_override() -> None:
    assert TTSConfig().preload_languages is None  # pydantic default: all defaults eager
    assert Settings().tts.preload_languages == ["pl"]  # repo config.toml: paradee stays lazy
    settings = Settings(tts={"preload_languages": None})
    assert settings.tts.preload_languages is None
