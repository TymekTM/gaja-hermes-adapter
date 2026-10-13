"""Household: people, the Hermes profiles they may use, bound phones/devices, the shared house note.

Stored in data/household.json and edited only through the admin panel. Each person has a default
Hermes profile plus optional extra profiles; Gaja picks among them per request (explicit name first,
then the router). Unknown voices use the general profile. Sensitive profiles are never used for a
voice turn: only typed text from a phone bound to that person reaches them.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

log = logging.getLogger(__name__)

ID_RE = r"^[a-z0-9][a-z0-9_-]{0,63}$"
GUEST = "guest"


class Profile(BaseModel):
    """A Hermes profile Gaja may hand tasks to (`id` is the Hermes profile name, "default" = ~/.hermes)."""

    id: str = Field(pattern=ID_RE)
    label: str = Field(min_length=1, max_length=40)
    # Used by the router to pick between a person's profiles ("lista zadań, przypomnienia, plan dnia").
    description: str = Field(default="", max_length=300)
    # Spoken names that force this profile ("sierżancie", "sierżant"); the label is always one.
    aliases: list[str] = Field(default_factory=list)
    sensitive: bool = False


class Person(BaseModel):
    id: str = Field(pattern=ID_RE)
    name: str = Field(min_length=1, max_length=40)
    default_profile: str
    # Extra profiles besides the default one.
    profiles: list[str] = Field(default_factory=list)
    language: Literal["pl", "en"] | None = None
    tts: str | None = None

    def allowed(self) -> list[str]:
        return [self.default_profile, *[p for p in self.profiles if p != self.default_profile]]


class Client(BaseModel):
    """A phone or browser that said hello with this client_id; `person` is set in the panel."""

    person: str | None = None
    label: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0


class Room(BaseModel):
    """A shared voice device (Atom). `person` makes everyone on it that person (no voice check)."""

    name: str = ""
    person: str | None = None


class Household(BaseModel):
    people: list[Person] = Field(default_factory=list)
    profiles: list[Profile] = Field(
        default_factory=lambda: [Profile(id="default", label="Gaja", description="od wszystkiego")]
    )
    # Profile for unknown voices (and people without their own profile yet).
    general_profile: str = "default"
    clients: dict[str, Client] = Field(default_factory=dict)
    devices: dict[str, Room] = Field(default_factory=dict)
    # Shared context for every conversation (who lives here, rooms, habits).
    note: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def _check_refs(self) -> Household:
        ids = [p.id for p in self.profiles]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate profile id")
        people = [p.id for p in self.people]
        if len(set(people)) != len(people) or GUEST in people:
            raise ValueError("duplicate or reserved person id")
        known = set(ids)
        if self.general_profile not in known:
            raise ValueError(f"general profile {self.general_profile!r} is not defined")
        for person in self.people:
            missing = [p for p in person.allowed() if p not in known]
            if missing:
                raise ValueError(f"{person.name}: unknown profiles {missing}")
        for ref in [c.person for c in self.clients.values()] + [r.person for r in self.devices.values()]:
            if ref is not None and ref not in people:
                raise ValueError(f"unknown person {ref!r}")
        return self

    def person(self, person_id: str | None) -> Person | None:
        return next((p for p in self.people if p.id == person_id), None)

    def profile(self, profile_id: str) -> Profile | None:
        return next((p for p in self.profiles if p.id == profile_id), None)


@dataclass(frozen=True)
class Identity:
    """Who a turn comes from. `person` None = unknown voice / guest."""

    person: Person | None
    source: Literal["client", "device", "voice", "unknown"]

    @property
    def key(self) -> str:
        return self.person.id if self.person is not None else GUEST

    @property
    def name(self) -> str | None:
        return self.person.name if self.person is not None else None


UNKNOWN = Identity(None, "unknown")


class HouseholdStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.on_change: list[Callable[[], None]] = []
        self._data = self._load()

    def _load(self) -> Household:
        if not self.path.exists():
            return Household()
        try:
            return Household.model_validate_json(self.path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("could not read %s; starting with an empty household", self.path)
            return Household()

    @property
    def data(self) -> Household:
        return self._data

    def replace(self, data: Household) -> Household:
        """Validate and store a whole new household (the panel sends the full document)."""
        data = Household.model_validate(data.model_dump())
        with self._lock:
            # Client bookkeeping (first/last seen) is ours; keep it for clients the panel still lists.
            for cid, client in data.clients.items():
                old = self._data.clients.get(cid)
                if old is not None:
                    client.first_seen, client.last_seen = old.first_seen, old.last_seen
            self._data = data
            self._save()
        self._changed()
        return data

    def touch_client(self, client_id: str, label: str, now: float) -> None:
        with self._lock:
            client = self._data.clients.get(client_id)
            if client is None:
                client = self._data.clients[client_id] = Client(label=label, first_seen=now)
            elif label and not client.label:
                client.label = label
            client.last_seen = now
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(self._data.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def _changed(self) -> None:
        for fn in self.on_change:
            try:
                fn()
            except Exception:
                log.exception("household on_change callback failed")

    # ---- per-turn decisions ------------------------------------------------------------

    def identify(self, *, client_id: str | None = None, device_id: str | None = None) -> Identity:
        """Identity from a bound phone or a device bound to one person (voice ID comes later)."""
        h = self._data
        if client_id is not None:
            client = h.clients.get(client_id)
            person = h.person(client.person) if client is not None else None
            if person is not None:
                return Identity(person, "client")
        if device_id is not None:
            room = h.devices.get(device_id)
            person = h.person(room.person) if room is not None else None
            if person is not None:
                return Identity(person, "device")
        return UNKNOWN

    def usable_profiles(self, identity: Identity, *, voice: bool) -> list[Profile]:
        """Profiles this turn may use, the person's default first. Voice turns skip sensitive ones;
        typed turns reach them only from the person's own bound phone."""
        h = self._data
        ids = identity.person.allowed() if identity.person is not None else [h.general_profile]
        trusted_text = identity.source == "client" and not voice
        out = []
        for pid in ids:
            profile = h.profile(pid)
            if profile is not None and (not profile.sensitive or trusted_text):
                out.append(profile)
        if not out:
            general = h.profile(h.general_profile)
            out = [general] if general is not None and not general.sensitive else []
        return out


def fold(text: str) -> str:
    """Lowercase, strip diacritics (ł -> l) for alias matching."""
    text = text.lower().replace("ł", "l")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


# The assistant's own name starts most utterances ("Hej Gaja, …"); it never selects a profile.
ASSISTANT_NAMES = {"gaja", "gajo", "gaju", "gai", "gaje", "gaji", "gajy"}


def explicit_profile(utterance: str, profiles: list[Profile]) -> Profile | None:
    """A profile addressed by name in the utterance ("Sierżancie, co mam dziś?", "zapytaj sierżanta…").

    Names of 5+ letters match any word starting with the name minus its last 2 letters (Polish
    inflection: sierżancie, sierżanta, sierżant); shorter names must match a whole word."""
    words = [w for w in re.findall(r"\w+", fold(utterance)) if w not in ASSISTANT_NAMES]
    for profile in profiles:
        for name in [profile.label, *profile.aliases]:
            stem = fold(name).strip()
            if " " in stem or len(stem) < 3 or stem in ASSISTANT_NAMES:
                continue
            if len(stem) >= 5:
                hit = any(w.startswith(stem[:-2]) for w in words)
            else:
                hit = stem in words
            if hit:
                return profile
    return None
