from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("GAJA_CONFIG", ROOT / "config.toml"))
LOCAL_CONFIG_PATH = Path(os.environ.get("GAJA_LOCAL_CONFIG", ROOT / "config.local.toml"))

# Languages the adapter speaks: STT default/fallback, per-turn TTS routing and the router prompts.
# Every language here needs a TTS engine in [tts] engine_by_language (validated below).
LANGUAGES = ("pl", "en")

log = logging.getLogger(__name__)


class HomeConfig(BaseModel):
    """Home Assistant REST API tool (credentials HASS_URL and HASS_TOKEN come from the environment only).

    The household's real data — spoken aliases, the write allowlist, spoken reply forms and the
    boiler id markers — lives in ``household_file`` (default ``data/home.toml``, gitignored and
    never in the repo; see ``docs/home.example.toml`` for the schema). Without that file the
    aliases are empty and the allowlist is empty, which means no writes at all (NEW-6).
    """

    enabled: bool = True
    timeout_s: float = 3.0
    household_file: Path = Path("data/home.toml")
    write_allowlist: list[str] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    # Spoken reply forms per entity: (nominative, accusative, genitive), e.g. "Włączyłam klimę w salonie".
    spoken_forms: dict[str, tuple[str, str, str]] = Field(default_factory=dict)
    # entity_id substrings that mark heating appliances as read-only (wildcards never unlock them).
    boiler_id_substrings: list[str] = Field(default_factory=list)


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
    # VAD backend: "onnx" = onnxruntime + numpy state (no torch, ~84 MB vs ~480 MB for "torch",
    # identical outputs); "torch" = the silero_vad wrapper (needs torch also for the TTS G2P chain).
    backend: Literal["onnx", "torch"] = "onnx"


class FasterWhisperConfig(BaseModel):
    model: str = "small"
    device: str = "cuda"
    compute_type: str = "int8"
    beam_size: int = 1


class ParakeetConfig(BaseModel):
    model_dir: Path = Path("~/.cache/gaja-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
    num_threads: int = 4
    hotwords_score: float = 3.0
    max_active_paths: int = 4
    prefix_below_s: float = 2.6


class STTConfig(BaseModel):
    default_language: Literal["pl", "en"] = "pl"
    engine: Literal["whistle", "faster-whisper", "parakeet"] = "faster-whisper"
    # Engine for device transcripts that decide a wake ("hej gaja"); None = the same as engine.
    wake_engine: Literal["whistle", "faster-whisper", "parakeet"] | None = None
    faster_whisper: FasterWhisperConfig = Field(default_factory=FasterWhisperConfig)
    parakeet: ParakeetConfig = Field(default_factory=ParakeetConfig)


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
    # Languages whose default engines load at startup; None = all defaults (today's behavior).
    # With ["pl"] the paradee stack (en: misaki+spacy+torch, ~710 MB) loads on the first en turn
    # instead (~12 s once per boot).
    preload_languages: list[str] | None = None
    piper: PiperConfig = Field(default_factory=PiperConfig)
    piper_darkman: PiperConfig = Field(
        default_factory=lambda: PiperConfig(model_path=Path("models/pl_PL-darkman-medium.onnx"))
    )
    paradee: ParadeeConfig = Field(default_factory=ParadeeConfig)
    supertonic: SupertonicConfig = Field(default_factory=SupertonicConfig)

    @model_validator(mode="after")
    def _every_language_has_an_engine(self) -> "TTSConfig":
        # Session._engine_for indexes this dict per turn: a missing language raised KeyError
        # mid-turn. Fail at load instead, with the language(s) that need an engine.
        missing = [lang for lang in LANGUAGES if lang not in self.engine_by_language]
        if missing:
            raise ValueError(
                f"tts.engine_by_language has no engine for {', '.join(repr(lang) for lang in missing)}: "
                f"every language in LANGUAGES ({', '.join(LANGUAGES)}) needs one "
                f"(configured: {', '.join(sorted(self.engine_by_language)) or 'none'})"
            )
        return self


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


class SpotifyConfig(BaseModel):
    """Household Spotify Web API tool (own OAuth2 PKCE login stored in token_path, never Hermes's).

    ``token_source = "hermes"`` borrows a short-lived access token from Hermes instead
    (``scripts/hermes_spotify_token.py`` under Hermes's own interpreter): Spotify rotates
    refresh tokens, so only Hermes's code may refresh — Gaja never sees the refresh token.
    """

    enabled: bool = True
    # "file": own PKCE login (token_path). "hermes": borrow an access token from Hermes.
    token_source: Literal["file", "hermes"] = "file"
    # Hermes interpreter for the broker (empty = resolve from the shebang of `hermes`).
    hermes_python: str = ""
    # Hermes root for the broker's HERMES_HOME (empty = ~/.hermes; a person's profile P
    # resolves to <hermes_home>/profiles/P at runtime).
    hermes_home: str = ""
    client_id: str = ""
    token_path: Path = Path("data/spotify_token.json")
    device_name: str = ""
    timeout_s: float = 3.0
    volume_step: int = 15
    market: str = "PL"


class TextConfig(BaseModel):
    # Run gaja.text.normalize.normalize_for_tts on each sentence right before TTS.
    normalize_tts: bool = True


class StyleConfig(BaseModel):
    auto_emotion: bool = True
    # Post-hoc WSOLA pitch shift of the synthesised audio. Off: on real voices it sounds phasey/metallic,
    # so emotion is carried by rate, the engine's own expressiveness (Piper noise_scale) and pauses.
    pitch_shift: bool = False


class HermesGatewayConfig(BaseModel):
    """Hermes gateway HTTP API (one process, every profile under /p/<profile>/). Off until set up."""

    enabled: bool = False
    url: str = "http://127.0.0.1:8642"
    # Profile keys are read at runtime from <hermes_home>/.env and <hermes_home>/profiles/<name>/.env.
    hermes_home: str = "~/.hermes"
    key_var: str = "API_SERVER_KEY"
    # A person's Hermes session is replaced after this much idle time (and on "nowy temat").
    session_idle_min: float = 30.0


class HermesConfig(BaseModel):
    backend: Literal["acp", "stub"] = "acp"
    command: list[str] = Field(default_factory=lambda: ["hermes", "acp"])
    cwd: str = "~"
    timeout_s: float = 300.0
    stub_delay_s: float = 1.5
    direct_max_chars: int = 160
    progress_interval_s: float = 0.5
    speaker_mcp: bool = True
    gateway: HermesGatewayConfig = Field(default_factory=HermesGatewayConfig)


class HouseholdConfig(BaseModel):
    """People, their Hermes profiles and bound phones/devices (edited in the admin panel)."""

    path: Path = Path("data/household.json")
    # The router's pick between a person's profiles must reach this probability, else the default one.
    profile_threshold: float = 0.6


class MetricsConfig(BaseModel):
    turns_log: Path = Path("logs/turns.jsonl")
    window: int = 200
    # Device turns that decoded to nothing keep their audio here (newest N; 0 = off), see gaja/audio/keep.py.
    empty_turn_audio_dir: Path = Path("logs/empty_turns")
    empty_turn_audio_keep: int = Field(default=0, ge=0, le=500)
    # Diagnosis only: keep every device turn (not just empty ones) under the same cap.
    turn_audio_keep_all: bool = False


class WakewordConfig(BaseModel):
    # Wake-word sample recorder (web/record.html, gaja/wakeword_samples.py).
    recordings_dir: Path = Path("data/wakeword/recordings")
    max_total_mb: int = 32768  # about 280 h of 16 kHz PCM16 (115 MB/h): enough for days of continuous ambient recording


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
    # Full-duplex (caps.aec) devices: let the user's voice interrupt a reply (server-side echo
    # canceller + VAD barge-in). Off: such a device is served like a half-duplex one (listen_end
    # after the utterance, barge-in by wake word only, which its on-device model hears through the
    # reply). On the Atom VoiceS3R the echo is ~19 dB above the user's voice and clips, so no linear
    # canceller removes it and Gaja barges in on herself (measured 2026-10-21, docs/TODO.md).
    speech_barge_in: bool = False
    # Reply to a bare wake phrase: "spoken" (TTS), "earcon" (short beep audio) or "none".
    ack: Literal["spoken", "earcon", "none"] = "spoken"
    ack_text: dict[str, str] = Field(default_factory=lambda: {"pl": "Słucham?", "en": "Yes?"})
    # After a reply the mic stays open this long for a follow-up (0 = off).
    follow_up_ms: int = 6000
    listen_grace_s: float = 1.0
    # If the connection dies before a user-turn reply finishes (no audio_end), repeat it on reconnect within this window (0 = off).
    reply_recovery_s: float = 30.0
    # Preferred output sample rate when the device offers it (else the lowest offered rate >= 16 kHz).
    preferred_out_rate: int = 16000
    out_chunk_ms: int = 100
    # Reply audio is paced to the device (ResamplingOutbox): at most this far ahead of playback, after a
    # 0.5 s burst at 2x real time. 0 sends as fast as TTS produces (overflowed the Wi-Fi link: gaps, drops).
    downlink_lead_s: float = 3.0
    # Quiet hours: unforced playback on speakers is refused during this window (HH:MM-HH:MM, local time). Empty disables.
    quiet_hours: str = "22:00-07:00"
    # Extra user sounds for /play (WAV/MP3/OGG/FLAC, flat files). Re-read on every request, so files can
    # be dropped in without a restart; a user file overrides a built-in of the same name.
    sounds_dir: Path = Path("data/sounds")

    @field_validator("quiet_hours")
    @classmethod
    def check_quiet_hours(cls, v: str) -> str:
        from gaja.devices.quiet_hours import validate_quiet_hours

        return validate_quiet_hours(v)


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
    spotify: SpotifyConfig = Field(default_factory=SpotifyConfig)
    home: HomeConfig = Field(default_factory=HomeConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    hermes: HermesConfig = Field(default_factory=HermesConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    wakeword: WakewordConfig = Field(default_factory=WakewordConfig)
    devices: DevicesConfig = Field(default_factory=DevicesConfig)
    vocab: VocabConfig = Field(default_factory=VocabConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    household: HouseholdConfig = Field(default_factory=HouseholdConfig)

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        sources = [init_settings, env_settings]
        if LOCAL_CONFIG_PATH.exists():
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=LOCAL_CONFIG_PATH))
        sources.append(TomlConfigSettingsSource(settings_cls, toml_file=CONFIG_PATH))
        return tuple(sources)

    @model_validator(mode="after")
    def _merge_home_household(self) -> "Settings":
        merge_home_household(self)
        return self


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


# Household-file keys that must stay top-level: nested under a table they parse as dotted keys
# ("spoken_forms.boiler_id_substrings") or as members of a table dict, and used to be dropped
# silently (H3/M1c).
_MISPLACED_HOME_KEYS: frozenset[str] = frozenset({"boiler_id_substrings", "write_allowlist"})


def _misplaced_home_keys(raw: dict, prefix: str = "") -> list[str]:
    """Names of boiler_id_substrings/write_allowlist keys hidden inside tables or dotted keys."""
    found: list[str] = []
    for key, value in raw.items():
        full = f"{prefix}{key}"
        leaf = key.rsplit(".", 1)[-1]
        if leaf in _MISPLACED_HOME_KEYS and (prefix or key != leaf):
            found.append(full)  # inside a table ("spoken_forms.boiler_id_substrings") or a dotted key
        if isinstance(value, dict):
            found.extend(_misplaced_home_keys(value, prefix=f"{full}."))
    return found


def merge_home_household(settings: Settings) -> None:
    """Merge the household's home data (``[home] household_file``) over ``settings.home`` (NEW-6).

    The file (default ``data/home.toml``, gitignored) holds the real entity ids, spoken aliases,
    write allowlist, spoken reply forms and boiler id markers — nothing household-specific lives in
    the repo. A missing file or invalid TOML leaves the config values untouched: with the empty
    defaults that means no aliases and no writes. Schema (see docs/home.example.toml):

        boiler_id_substrings = ["boiler"]              # read-only heating appliance markers (TOP LEVEL)
        write_allowlist = ["climate.salon", ...]       # fnmatch patterns; absent/empty = no writes
        [aliases]                                       # spoken phrase -> entity_id
        "klima w salonie" = "climate.salon"
        [spoken_forms]                                  # entity_id -> [nominative, accusative, genitive]
        "climate.salon" = ["klima w salonie", "klimę w salonie", "klimy w salonie"]

    ``boiler_id_substrings`` / ``write_allowlist`` nested inside a table, or a non-string alias
    value, means the file was misplaced after a ``[table]`` header: each is warned about by key
    and the write allowlist is emptied for the load (fail closed, H3/M1c).
    """
    path = settings.home.household_file
    try:
        raw = tomllib.loads(resolve(path).read_text(encoding="utf-8"))
    except OSError:
        log.debug("home household file %s not present; home aliases/write allowlist stay as configured", path)
        return
    except tomllib.TOMLDecodeError as exc:
        log.warning("home household file %s is not valid TOML (%s); ignoring it", path, exc)
        return

    warned_keys: set[str] = set()

    def _warn_key(key: str, reason: str = "misplaced/malformed") -> None:
        if key not in warned_keys:
            warned_keys.add(key)
            log.warning("home household file %s: ignoring %s key '%s'", path, reason, key)

    misplaced: list[str] = sorted(_misplaced_home_keys(raw))
    aliases = raw.get("aliases")
    bad_alias_values: list[str] = []
    if isinstance(aliases, dict):
        bad_alias_values = sorted(str(k) for k, v in aliases.items() if not isinstance(v, str))
    fail_closed = bool(misplaced or bad_alias_values)
    if fail_closed:
        for key in misplaced + [f"aliases.{k}" for k in bad_alias_values]:
            _warn_key(key, reason="misplaced/malformed")
        log.warning("home household file %s: fail closed, the write allowlist is emptied for this load", path)
        settings.home.write_allowlist.clear()

    # Warn once per unknown top-level key (L1)
    _KNOWN_HOME_KEYS = frozenset({"boiler_id_substrings", "write_allowlist", "aliases", "spoken_forms"})
    for key in sorted(raw.keys() - _KNOWN_HOME_KEYS):
        _warn_key(key, reason="unknown/dropped")

    aliases = raw.get("aliases")
    if isinstance(aliases, dict):
        for alias, eid in aliases.items():
            if isinstance(alias, str) and isinstance(eid, str) and alias.strip() and eid.strip():
                settings.home.aliases[alias] = eid
            else:
                _warn_key(f"aliases.{alias}", reason="malformed")
    elif aliases is not None:
        _warn_key("aliases", reason="malformed")

    allowlist = raw.get("write_allowlist")
    if not fail_closed:
        if isinstance(allowlist, list) and all(isinstance(v, str) for v in allowlist):
            for p in allowlist:
                if p and p.strip():
                    settings.home.write_allowlist.append(p.strip().lower())
                else:
                    log.warning("home household file %s: ignoring an empty write_allowlist entry", path)
        elif allowlist is not None:
            _warn_key("write_allowlist", reason="malformed")

    spoken = raw.get("spoken_forms")
    if isinstance(spoken, dict):
        for eid, forms in spoken.items():
            if f"spoken_forms.{eid}" in warned_keys or eid in _MISPLACED_HOME_KEYS:
                continue
            if not isinstance(eid, str) or not eid.strip() or not isinstance(forms, list) or not forms or not all(isinstance(f, str) and f.strip() for f in forms):
                _warn_key(f"spoken_forms.{eid}", reason="malformed")
                continue
            padded = [forms[i % len(forms)] for i in range(3)]
            settings.home.spoken_forms[eid.lower()] = (padded[0], padded[1], padded[2])
    elif spoken is not None:
        _warn_key("spoken_forms", reason="malformed")

    boiler = raw.get("boiler_id_substrings")
    if isinstance(boiler, list) and all(isinstance(v, str) for v in boiler):
        for marker in boiler:
            norm = marker.strip().lower()
            if not norm:
                log.warning("home household file %s: ignoring an empty boiler_id_substrings entry", path)
                continue
            if norm not in settings.home.boiler_id_substrings:
                settings.home.boiler_id_substrings.append(norm)
    elif boiler is not None:
        _warn_key("boiler_id_substrings", reason="malformed")
