from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("GAJA_CONFIG", ROOT / "config.toml"))
LOCAL_CONFIG_PATH = Path(os.environ.get("GAJA_LOCAL_CONFIG", ROOT / "config.local.toml"))


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8790
    web_dir: Path = Path("web")
    # Browser origins allowed to open /ws (exact scheme://host[:port]). Requests without Origin are allowed.
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8790",
            "https://localhost:8790",
            "http://127.0.0.1:8790",
            "https://127.0.0.1:8790",
        ]
    )


class LLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8791"
    model: str = "lfm2.5-2.6b"
    temperature: float = 0.3
    max_tokens: int = 200
    timeout_s: float = 60.0
    tool_strategy: Literal["native", "pythonic"] = "native"
    system_prompt_path: Path = Path("prompts/system_pl.md")
    history_max_turns: int = 12
    history_max_chars: int = 3000
    # llama-server slot ids (-np 2): the router and the chat reply keep separate prompt caches. -1 = server's choice.
    router_slot: int = 1
    chat_slot: int = 0
    # After each turn, evaluate the updated history into the chat slot (off the critical path), so the
    # next reply prefills only the new user message. False = only the session-start preload.
    prime_after_turn: bool = True


class VADConfig(BaseModel):
    threshold: float = 0.5
    min_silence_ms: int = 500
    spec_silence_ms: int = 0
    pre_roll_ms: int = 300
    min_speech_ms: int = 250
    max_utterance_s: float = 60.0
    # Speech needed before speech_start (barge-in) fires; stricter while Gaja is talking (echo guard).
    barge_in_ms: int = 250
    barge_in_strict_ms: int = 400
    strict_threshold: float = 0.7
    # Smart Turn v3 semantic endpointing
    smart_turn: bool = False
    smart_turn_model_path: Path | None = None
    smart_turn_check_ms: int = 250
    smart_turn_threshold: float = 0.5
    max_incomplete_silence_ms: int = 1200


class FasterWhisperConfig(BaseModel):
    model: str = "small"
    device: str = "cuda"
    compute_type: str = "int8"
    beam_size: int = 1


class STTConfig(BaseModel):
    default_language: Literal["pl", "en"] = "pl"
    engine: Literal["whistle", "faster-whisper"] = "faster-whisper"
    faster_whisper: FasterWhisperConfig = Field(default_factory=FasterWhisperConfig)


class PiperConfig(BaseModel):
    model_path: Path = Path("models/pl_PL-gosia-medium.onnx")
    threads: int = 2


class ParadeeConfig(BaseModel):
    threads: int = 1


class SupertonicConfig(BaseModel):
    steps: int = 6
    voice: str = "M4"
    speed: float = 1.05
    threads: int = 2
    output_sample_rate: int | None = 22050


class TTSConfig(BaseModel):
    engine_by_language: dict[str, str] = Field(default_factory=lambda: {"pl": "piper", "en": "paradee"})
    piper: PiperConfig = Field(default_factory=PiperConfig)
    piper_darkman: PiperConfig = Field(
        default_factory=lambda: PiperConfig(model_path=Path("models/pl_PL-darkman-medium.onnx"))
    )
    paradee: ParadeeConfig = Field(default_factory=ParadeeConfig)
    supertonic: SupertonicConfig = Field(default_factory=SupertonicConfig)


class ChunkerConfig(BaseModel):
    first_chunk_min_words: int = 3
    first_chunk_max_words: int = 0
    min_words: int = 3
    max_chars: int = 220


class RouterConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.97
    # 3-class router only: route to a local tool iff P(tool) >= tool_threshold (and not HERMES).
    tool_threshold: float = 0.5
    history_turns: int = 2
    timeout_s: float = 2.0
    prompt_path: Path = Path("prompts/router_pl.md")


class ToolsConfig(BaseModel):
    # Local tools (gaja/tools) behind the 3-class router; needs a 3-class router prompt.
    enabled: bool = False
    default_location: str = "Katowice"
    timeout_s: float = 4.0


class TextConfig(BaseModel):
    # Run gaja.text.normalize.normalize_for_tts on each sentence right before TTS.
    normalize_tts: bool = True


class StyleConfig(BaseModel):
    auto_emotion: bool = True
    # Post-hoc WSOLA pitch shift of the synthesised audio. Off: on real voices it sounds phasey/metallic,
    # so emotion is carried by rate, the engine's own expressiveness (Piper noise_scale) and pauses.
    pitch_shift: bool = False


class HermesConfig(BaseModel):
    backend: Literal["acp", "stub"] = "acp"
    command: list[str] = Field(default_factory=lambda: ["hermes", "acp"])
    cwd: str = "~"
    timeout_s: float = 300.0
    stub_delay_s: float = 1.5
    direct_max_chars: int = 160
    progress_interval_s: float = 0.5


class MetricsConfig(BaseModel):
    turns_log: Path = Path("logs/turns.jsonl")
    window: int = 200


class WakewordConfig(BaseModel):
    # Wake-word sample recorder (web/record.html, gaja/wakeword_samples.py).
    recordings_dir: Path = Path("data/wakeword/recordings")
    max_total_mb: int = 8192


class DevicesConfig(BaseModel):
    """Optional listener for IoT voice devices (ESP32, protocol v1.8; docs/DEVICE.md). Off by default."""

    enabled: bool = False
    listen_host: str = "0.0.0.0"
    port: int = 8792
    # Per-device tokens (TOML, `[devices.<id>] token = "..."`); keep it outside the repo, mode 600.
    tokens_file: Path = Path("~/.config/gaja/devices.toml")
    # Peers outside these networks are refused (defence in depth if the port is ever forwarded).
    allowed_networks: list[str] = Field(
        default_factory=lambda: [
            "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10",
            "169.254.0.0/16", "::1/128", "fc00::/7", "fe80::/10",
        ]
    )
    max_connections: int = 8
    hello_timeout_s: float = 5.0
    # Optional TLS (wss) for the listener.
    tls_certfile: Path | None = None
    tls_keyfile: Path | None = None
    # Wake verification: the first verify_ms of the stream (pre-roll included) is transcribed and must
    # start with hej/hey/ej + gaja/gajo/...; otherwise the device gets wake_rejected.
    verify_wake: bool = True
    verify_ms: int = 2500
    verify_hotwords: str = "Hej Gaja, Gajo"
    # A lone "Gaja"/"Gajo" as the very first word is accepted (the device already heard the full phrase).
    accept_bare_name: bool = True
    extra_wake_names: list[str] = Field(default_factory=list)
    extra_wake_greetings: list[str] = Field(default_factory=list)
    # How long a wake may stay undecided / an accepted wake may stay without speech.
    wake_timeout_s: float = 8.0
    listen_timeout_s: float = 8.0
    # A wake while Gaja is busy needs at least this on-device score (her own voice says "Gaja").
    barge_in_min_score: float = 0.9
    # Reply to a bare wake phrase: "spoken" (TTS), "earcon" (short beep audio) or "none".
    ack: Literal["spoken", "earcon", "none"] = "spoken"
    ack_text: dict[str, str] = Field(default_factory=lambda: {"pl": "Słucham?", "en": "Yes?"})
    # After a reply the mic stays open this long for a follow-up (0 = off).
    follow_up_ms: int = 6000
    listen_grace_s: float = 1.0
    # Preferred output sample rate when the device offers it (else the lowest offered rate >= 16 kHz).
    preferred_out_rate: int = 16000
    out_chunk_ms: int = 100


class VocabConfig(BaseModel):
    enabled: bool = True
    default_path: Path = Path("prompts/vocab_default.toml")
    user_path: Path = Path("data/vocab.toml")
    suggestions_path: Path = Path("data/vocab_suggestions.json")
    max_hotwords: int = 50
    fuzzy: bool = True


class AdminConfig(BaseModel):
    """Admin panel (/admin). The password hash and session key live in credentials_file (mode 600),
    created with scripts/admin_password.py; without that file the panel refuses every login."""

    enabled: bool = True
    credentials_file: Path = Path("~/.config/gaja/admin.toml")
    session_hours: float = 12.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GAJA_", env_nested_delimiter="__", extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    style: StyleConfig = Field(default_factory=StyleConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    hermes: HermesConfig = Field(default_factory=HermesConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    wakeword: WakewordConfig = Field(default_factory=WakewordConfig)
    devices: DevicesConfig = Field(default_factory=DevicesConfig)
    vocab: VocabConfig = Field(default_factory=VocabConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        sources = [init_settings, env_settings]
        if LOCAL_CONFIG_PATH.exists():
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=LOCAL_CONFIG_PATH))
        sources.append(TomlConfigSettingsSource(settings_cls, toml_file=CONFIG_PATH))
        return tuple(sources)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path
