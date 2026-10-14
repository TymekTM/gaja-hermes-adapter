from pathlib import Path
import tempfile
import os
import pytest

import gaja.config as config_mod
from gaja.config import Settings, ServerConfig, HermesConfig


@pytest.fixture(autouse=True)
def no_host_local_config(monkeypatch, tmp_path) -> None:
    """Ignore the checkout's own config.local.toml so the defaults are tested."""
    monkeypatch.setattr(config_mod, "LOCAL_CONFIG_PATH", tmp_path / "absent.toml")


def test_config_defaults() -> None:
    settings = Settings()
    assert settings.server.port == 8790
    assert settings.hermes.command == ["hermes", "acp"]
    assert all(".ts.net" not in o for o in settings.server.allowed_origins)

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
