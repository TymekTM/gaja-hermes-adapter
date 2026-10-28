"""Read and edit the adapter settings for the admin panel.

Edits go to config.local.toml (tomlkit keeps its comments); config.toml stays the documented base.
Field help comes from the comment lines directly above each key in config.toml.
"""

from __future__ import annotations

import math
import os
import re
import tomllib
import types
import typing
from pathlib import Path
from typing import Any, Callable, Literal

import tomlkit
from pydantic import BaseModel

from gaja import config as config_module
from gaja.config import Settings

# Edited elsewhere (or never from a browser): the panel shows them read-only.
READ_ONLY = {
    "admin",
    "devices.tokens_file",
    "server.web_dir",
    "household.path",
    "spotify.token_path",
    "home.household_file",
    "home.write_allowlist",
    "home.aliases",
    "home.spoken_forms",
    "home.boiler_id_substrings",
}
# Editable, but a wrong value can lock the panel out; the UI asks for confirmation.
DANGER = {"server.host", "server.port", "server.allowed_origins", "hermes.command", "hermes.backend"}

_SECTION = re.compile(r"^\[([A-Za-z0-9_.]+)\]\s*$")
_KEY = re.compile(r"^([A-Za-z0-9_]+)\s*=")


def _paths() -> tuple[Path, Path]:
    return config_module.CONFIG_PATH, config_module.LOCAL_CONFIG_PATH


def describe_config(text: str) -> dict[str, str]:
    """Map "section.key" (and "section" for headers) to the comment block written right above it."""
    out: dict[str, str] = {}
    section = ""
    pending: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            pending.append(line.lstrip("#").strip())
            continue
        if not line:
            pending = []
            continue
        if m := _SECTION.match(line):
            section = m.group(1)
            if pending:
                out[section] = " ".join(pending)
        elif (m := _KEY.match(line)) and section:
            if pending:
                out[f"{section}.{m.group(1)}"] = " ".join(pending)
        pending = []
    return out


def _kind(annotation: Any) -> tuple[str, bool, list[Any] | None]:
    """(kind, optional, choices) for a field annotation."""
    optional = False
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        optional = len(args) < len(typing.get_args(annotation))
        annotation = args[0] if len(args) == 1 else annotation
        origin = typing.get_origin(annotation)
    if origin is Literal:
        return "choice", optional, list(typing.get_args(annotation))
    if origin is list:
        return "list", optional, None
    if origin is dict:
        return "map", optional, None
    for kind, typ in (("bool", bool), ("int", int), ("float", float), ("path", Path), ("str", str)):
        if annotation is typ:
            return kind, optional, None
    return "str", optional, None


def _walk(model: type[BaseModel], prefix: str = "") -> list[tuple[str, Any]]:
    fields: list[tuple[str, Any]] = []
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            fields.extend(_walk(ann, f"{path}."))
        else:
            fields.append((path, ann))
    return fields


def _get(data: Any, path: str) -> tuple[bool, Any]:
    for part in path.split("."):
        if not isinstance(data, dict) or part not in data:
            return False, None
        data = data[part]
    return True, data


def _read_only(path: str) -> bool:
    return any(path == p or path.startswith(p + ".") for p in READ_ONLY)


def _local_doc() -> tomlkit.TOMLDocument:
    _, local = _paths()
    return tomlkit.parse(local.read_text()) if local.exists() else tomlkit.document()


def snapshot(running: Settings) -> dict[str, Any]:
    """Every editable field with its effective, saved-local, default and running value."""
    base_path, _ = _paths()
    base_text = base_path.read_text()
    help_text = describe_config(base_text)
    base = tomllib.loads(base_text)
    local = _local_doc().unwrap()
    effective = Settings().model_dump(mode="json")
    running_dump = running.model_dump(mode="json")
    code_defaults = Settings.model_construct().model_dump(mode="json")
    sections: dict[str, str] = {}
    fields = []
    for path, ann in _walk(Settings):
        kind, optional, choices = _kind(ann)
        section = path.rsplit(".", 1)[0]
        sections.setdefault(section, help_text.get(section, ""))
        has_local, local_value = _get(local, path)
        has_base, base_value = _get(base, path)
        value = _get(effective, path)[1]
        fields.append(
            {
                "path": path,
                "section": section,
                "kind": kind,
                "optional": optional,
                "choices": choices,
                "help": help_text.get(path, ""),
                "value": value,
                "default": base_value if has_base else _get(code_defaults, path)[1],
                "overridden": has_local,
                "local": local_value if has_local else None,
                "running": _get(running_dump, path)[1],
                "restart_pending": value != _get(running_dump, path)[1],
                "read_only": _read_only(path),
                "danger": path in DANGER,
            }
        )
    return {"sections": sections, "fields": fields}


_TYPE_OK: dict[str, tuple[Callable[[Any], bool], str]] = {
    "bool": (lambda v: isinstance(v, bool), "a boolean"),
    "int": (lambda v: isinstance(v, int) and not isinstance(v, bool), "an integer"),
    "float": (
        lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
        "a finite number",
    ),
    "choice": (lambda v: isinstance(v, str), "one of the choices"),
    "list": (lambda v: isinstance(v, list), "a list"),
    "map": (lambda v: isinstance(v, dict), "a table"),
    "path": (lambda v: isinstance(v, str), "a string"),
    "str": (lambda v: isinstance(v, str), "a string"),
}


def _reject_wrong_type(patch: dict[str, Any]) -> None:
    """Refuse a raw value whose JSON type does not match the field (e.g. `true` for an int).
    Pydantic coerces leniently after the merge, so without this check the boolean/string would be
    written to config.local.toml as-is."""
    annotations = dict(_walk(Settings))
    for path, value in patch.items():
        if value is None:  # None removes the override; every field accepts that
            continue
        check, expected = _TYPE_OK[_kind(annotations[path])[0]]
        if not check(value):
            raise ValueError(f"{path} expects {expected}, got {type(value).__name__}")


def apply_patch(patch: dict[str, Any]) -> list[str]:
    """Write {"section.key": value} to config.local.toml; None removes the override.

    The merged result is validated as Settings before anything is written (ValueError on bad input).
    Returns the changed paths.
    """
    known = {path for path, _ in _walk(Settings)}
    for path in patch:
        if path not in known:
            raise ValueError(f"unknown setting {path!r}")
        if _read_only(path):
            raise ValueError(f"{path} is read-only in the panel")
    _reject_wrong_type(patch)
    doc = _local_doc()
    for path, value in patch.items():
        *parents, key = path.split(".")
        table: Any = doc
        for part in parents:
            if part not in table:
                if value is None:
                    break
                table[part] = tomlkit.table()
            table = table[part]
        else:
            if value is None:
                table.pop(key, None)
            else:
                table[key] = value
    _prune_empty(doc)
    base_path, local_path = _paths()
    merged = _deep_merge(tomllib.loads(base_path.read_text()), doc.unwrap())
    try:
        Settings(**merged)
    except Exception as exc:  # pydantic ValidationError and friends
        raise ValueError(str(exc)) from exc
    _atomic_write(local_path, tomlkit.dumps(doc))
    return list(patch)


def _prune_empty(table: Any) -> None:
    for key in list(table.keys()):
        value = table[key]
        if isinstance(value, dict):
            _prune_empty(value)
            if not value:
                del table[key]


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
