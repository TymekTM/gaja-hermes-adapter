from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from gaja.config import Settings
from gaja.tools.base import ToolResult
from gaja.tools.calc_tool import calculate
from gaja.tools.convert_tool import convert_units
from gaja.tools.currency_tool import exchange_rate
from gaja.tools.holiday_tool import next_holiday
from gaja.tools.home_tool import home, home_configured
from gaja.tools.memory_tool import forget, remember
from gaja.tools.spotify_tool import spotify
from gaja.tools.sun_tool import get_sun
from gaja.tools.system_tool import server_status
from gaja.tools.time_tool import get_time
from gaja.tools.timer_tool import manage_timer
from gaja.tools.volume_tool import set_speaker_volume
from gaja.tools.weather_tool import WEATHER_TOOL_TIMEOUT_S, get_weather
from gaja.tools.wiki_tool import wiki_summary

log = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    description_pl: str
    description_en: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    timeout_s: float | None = None

    def schema(self, lang: str = "pl") -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description_pl if lang == "pl" else self.description_en,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._register_default_tools()

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self, lang: str = "pl") -> list[dict[str, Any]]:
        return [tool.schema(lang) for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        lang: str = "pl",
        timeout_s: float = 4.0,
    ) -> ToolResult:
        tool = self.get(name)
        if not tool:
            err = f"Nieznane narzędzie: {name}"
            speak = "Przepraszam, nie potrafię wykonać tej operacji." if lang == "pl" else "Sorry, I cannot perform this action."
            return ToolResult(data={"error": err}, speak=speak, error="unknown_tool")

        effective_timeout = tool.timeout_s if tool.timeout_s is not None else timeout_s
        try:
            handler_kwargs = dict(arguments)
            # Only inject lang if handler accepts it or needs it
            if "lang" not in handler_kwargs:
                handler_kwargs["lang"] = lang

            res = tool.handler(**handler_kwargs)
            if asyncio.iscoroutine(res):
                res = await asyncio.wait_for(res, timeout=effective_timeout)

            if isinstance(res, ToolResult):
                return res
            return ToolResult(data={"result": res}, speak=str(res))
        except TimeoutError:
            err = f"Przekroczono limit czasu ({effective_timeout}s) dla {name}"
            speak = "Sprawdzanie tych informacji zajęło zbyt dużo czasu." if lang == "pl" else "Checking this information timed out."
            log.warning("Tool %s timed out after %.1fs", name, effective_timeout)
            return ToolResult(data={"error": err}, speak=speak, error="timeout")
        except Exception as exc:
            err = f"Błąd wykonania {name}: {exc}"
            speak = "Wystąpił błąd podczas sprawdzania informacji." if lang == "pl" else "An error occurred while running the tool."
            log.exception("Tool %s execution failed: %s", name, exc)
            return ToolResult(data={"error": err}, speak=speak, error=str(exc))

    def _register_default_tools(self) -> None:
        self.register(
            ToolSpec(
                name="get_time",
                description_pl="Zwraca aktualną datę, godzinę, dzień tygodnia lub oblicza dzień tygodnia i liczbę dni do podanej daty (np. 3 listopada, 24 grudnia).",
                description_en="Returns current date, time, weekday, or calculates weekday and days until a target date.",
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Strefa czasowa, np. Europe/Warsaw",
                            "default": "Europe/Warsaw",
                        },
                        "target_date": {
                            "type": "string",
                            "description": "Data lub termin do obliczenia dnia tygodnia i dni do daty, np. 3 listopada, 24 grudnia",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["time", "date", "all"],
                            "description": "Zakres odpowiedzi: tylko godzina ('time'), tylko data/dzień ('date'), lub wszystko ('all').",
                            "default": "all",
                        },
                    },
                },
                handler=get_time,
            )
        )

        self.register(
            ToolSpec(
                name="get_weather",
                description_pl="Pobiera aktualną lub prognozowaną pogodę dla danej miejscowości (temperatura, odczuwalna, opady, wiatr, warunki, wschód/zachód słońca).",
                description_en="Fetches current or forecasted weather for a location (temperature, conditions, precipitation, wind, sunrise/sunset).",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Nazwa miejscowości w mianowniku (np. Kraków, Gdańsk, Wrocław). Pomiń, jeśli nie podano.",
                        },
                        "when": {
                            "type": "string",
                            "enum": ["now", "today", "tomorrow", "weekend"],
                            "description": "Termin: now (aktualna), today (dzisiaj), tomorrow (jutro), weekend (nadchodząca sobota i niedziela).",
                            "default": "now",
                        },
                    },
                },
                handler=get_weather,
                # The budget lives next to the HTTP timing constants in weather_tool, which the
                # geocoding deadline also uses: geocode candidates (one shared deadline) + the
                # forecast leg, each leg retried once on a 503. Never hand-copy this number.
                timeout_s=WEATHER_TOOL_TIMEOUT_S,
            )
        )

        self.register(
            ToolSpec(
                name="get_sun",
                description_pl="Oblicza offline wschód i zachód słońca oraz długość dnia dla danej miejscowości.",
                description_en="Computes offline sunrise, sunset, and day length for a location.",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Nazwa miejscowości w mianowniku (np. Kraków, Gdańsk, Wrocław). Pomiń, jeśli nie podano.",
                        },
                        "date": {
                            "type": "string",
                            "description": "Termin: today lub data YYYY-MM-DD.",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["all", "sunrise", "sunset", "day_length"],
                            "description": "Zakres odpowiedzi: all, sunrise, sunset lub day_length.",
                            "default": "all",
                        },
                    },
                },
                handler=get_sun,
            )
        )

        self.register(
            ToolSpec(
                name="calculate",
                description_pl="Bezpiecznie oblicza wyrażenie arytmetyczne (+, -, *, /, **, %, pierwiastki, procenty). Używaj do obliczeń matematycznych.",
                description_en="Safely calculates mathematical expressions (+, -, *, /, **, %, sqrt, percentages).",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Wyrażenie matematyczne np. 15 * 240 / 100, sqrt(144), 2**8, 12 * 5",
                        }
                    },
                    "required": ["expression"],
                },
                handler=calculate,
            )
        )

        self.register(
            ToolSpec(
                name="convert_units",
                description_pl="Przelicza jednostki miary (długość, masa, temperatura, objętość, jednostki kuchenne, rozmiary danych, prędkość, czas).",
                description_en="Converts units of measurement (length, mass, temperature, volume, cooking units, data sizes, speed, time).",
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "Wartość liczbowa do przeliczenia",
                        },
                        "from_unit": {
                            "type": "string",
                            "description": "Jednostka źródłowa np. km, kg, C, mil, funt, szklanka, MB, GB",
                        },
                        "to_unit": {
                            "type": "string",
                            "description": "Jednostka docelowa np. mil, funt, F, km, kg, ml, GB, MB",
                        },
                    },
                    "required": ["value", "from_unit", "to_unit"],
                },
                handler=convert_units,
            )
        )

        self.register(
            ToolSpec(
                name="exchange_rate",
                description_pl="Sprawdza kursy walut i przelicza waluty według tabeli NBP (Narodowy Bank Polski). Np. euro, dolary, funty na PLN.",
                description_en="Checks exchange rates and converts currencies using National Bank of Poland (NBP) table A.",
                parameters={
                    "type": "object",
                    "properties": {
                        "currency": {
                            "type": "string",
                            "description": "Kod lub nazwa waluty źródłowej np. EUR, USD, euro, dolar",
                        },
                        "amount": {
                            "type": "number",
                            "description": "Kwota do przeliczenia, domyślnie 1",
                            "default": 1.0,
                        },
                        "to": {
                            "type": "string",
                            "description": "Waluta docelowa, domyślnie PLN",
                            "default": "PLN",
                        },
                    },
                    "required": ["currency"],
                },
                handler=exchange_rate,
            )
        )

        self.register(
            ToolSpec(
                name="wiki_summary",
                description_pl="Zwraca krótkie 1-2 zdaniowe podsumowanie z Wikipedii o osobie, pojęciu, miejscu lub wydarzeniu.",
                description_en="Fetches a short 1-2 sentence Wikipedia summary for a person, concept, place, or event.",
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Temat lub hasło do sprawdzenia w Wikipedii, np. Maria Skłodowska-Curie, Warszawa, czarna dziura",
                        },
                        "lang": {
                            "type": "string",
                            "enum": ["pl", "en"],
                            "description": "Język artykułu (pl lub en)",
                            "default": "pl",
                        },
                    },
                    "required": ["topic"],
                },
                handler=wiki_summary,
            )
        )

        self.register(
            ToolSpec(
                name="server_status",
                description_pl="Sprawdza stan tego serwera: czas działania (uptime), obciążenie systemu (loadavg), zużycie RAM i GPU oraz stan usług asystentki.",
                description_en="Checks server health: uptime, load average, RAM, GPU memory, and service status.",
                parameters={"type": "object", "properties": {}},
                handler=server_status,
            )
        )

        self.register(
            ToolSpec(
                name="next_holiday",
                description_pl="Sprawdza najbliższe polskie święto lub ile dni zostało do konkretnego święta (np. Boże Narodzenie, Wielkanoc).",
                description_en="Computes next statutory Polish public holiday or days until a specific holiday.",
                parameters={
                    "type": "object",
                    "properties": {
                        "holiday": {
                            "type": "string",
                            "description": "Nazwa konkretnego święta np. Boże Narodzenie, Wielkanoc",
                        }
                    },
                },
                handler=next_holiday,
            )
        )

        self.register(
            ToolSpec(
                name="manage_timer",
                description_pl="Ustawia, sprawdza stan lub anuluje minutnik oraz przypomnienia.",
                description_en="Sets, checks status of, or cancels timers and reminders.",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["set", "status", "cancel"],
                            "description": "Akcja: set (ustaw), status (ile zostało), cancel (anuluj).",
                            "default": "set",
                        },
                        "duration_min": {
                            "type": "number",
                            "description": "Czas w minutach dla minutnika np. 10",
                        },
                        "time": {
                            "type": "string",
                            "description": "Godzina dla przypomnienia np. 18:30",
                        },
                        "label": {
                            "type": "string",
                            "description": "Etykieta lub treść przypomnienia np. pizza, zadzwonić do mamy",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["timer", "reminder"],
                            "description": "Typ: timer (minutnik) lub reminder (przypomnienie)",
                            "default": "timer",
                        },
                    },
                },
                handler=manage_timer,
            )
        )

        self.register(
            ToolSpec(
                name="set_speaker_volume",
                description_pl=(
                    "Zmienia głośność głośnika Gaji na tym urządzeniu. Użyj, gdy użytkownik prosi o"
                    " zrobienie ciszej/głośniej, ustawienie głośności, wyciszenie lub maksimum."
                    " action=set z level (0-100, 0 wycisza), action=step z delta (ujemna = ciszej,"
                    " dodatnia = głośniej, jeden krok to około 12), action=mute wycisza, action=max"
                    " ustawia maksimum."
                ),
                description_en=(
                    "Changes the Gaja speaker volume on the current device. Use when the user asks to"
                    " make it quieter/louder, set the volume, mute, or go to maximum."
                    " action=set with level (0-100, 0 mutes), action=step with delta (negative = quieter,"
                    " positive = louder, one step is about 12), action=mute mutes, action=max sets maximum."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["set", "step", "mute", "max"],
                            "description": "Akcja: set (ustaw poziom), step (relative), mute (wycisz), max (maksimum).",
                            "default": "set",
                        },
                        "level": {
                            "type": "number",
                            "description": "Poziom głośności 0-100 dla action=set (np. 30, 50, 100).",
                        },
                        "delta": {
                            "type": "number",
                            "description": "Zmiana w punktach dla action=step, np. -12 (ciszej) lub 24 (dużo głośniej).",
                        },
                    },
                },
                handler=set_speaker_volume,
            )
        )

        self.register(
            ToolSpec(
                name="spotify",
                description_pl=(
                    "Steruje odtwarzaniem muzyki na Spotify w domu (np. 'puść Dawida Podsiadło na Spotify',"
                    " 'włącz Sanah na Spotify', 'zatrzymaj muzykę', 'wznów muzykę', 'następna piosenka',"
                    " 'poprzedni utwór', 'muzykę głośniej', 'Spotify ciszej', 'głośność muzyki na 40',"
                    " 'co teraz gra na Spotify')."
                ),
                description_en=(
                    "Controls household Spotify playback on the active device: play/resume, pause, next,"
                    " previous, volume_up, volume_down, volume_set (0-100), now_playing, or play a query."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "play",
                                "pause",
                                "next",
                                "previous",
                                "volume_up",
                                "volume_down",
                                "volume_set",
                                "now_playing",
                            ],
                            "description": "Akcja: play (odtwórz lub wznów), pause, next, previous, volume_up, volume_down, volume_set, now_playing.",
                            "default": "play",
                        },
                        "query": {
                            "type": "string",
                            "description": "Wykonawca, utwór lub playlista dla action='play', np. 'Dawid Podsiadło', 'Sanah'. Pomiń przy wznowieniu.",
                        },
                        "level": {
                            "type": "number",
                            "description": "Poziom głośności Spotify 0-100 dla action='volume_set'.",
                        },
                    },
                },
                handler=spotify,
                timeout_s=12.0,
            )
        )

        self.register(
            ToolSpec(
                name="home",
                description_pl=(
                    "Sprawdza stan lub steruje klimatyzacją, telewizorami i amplitunerem w Home Assistant"
                    " (np. 'jaka temperatura w sypialni', 'czy klima w salonie chodzi', 'włącz klimę w salonie',"
                    " 'ustaw klimatyzację w sypialni na 22 stopnie', 'włącz chłodzenie w salonie',"
                    " 'wyłącz telewizor w salonie', 'wycisz amplituner', 'jaki stan pieca')."
                    " Nie obsługuje świateł ani rolet."
                ),
                description_en=(
                    "Reads status or controls Home Assistant air conditioners (climate.*) and media players"
                    " (TVs, Onkyo receiver), plus read-only boiler sensors. Does not control lights or blinds."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "status",
                                "turn_on",
                                "turn_off",
                                "set_temperature",
                                "set_hvac_mode",
                                "volume_up",
                                "volume_down",
                                "volume_set",
                                "mute",
                                "unmute",
                            ],
                            "description": "Akcja: status (odczyt), turn_on, turn_off, set_temperature, set_hvac_mode, volume_up, volume_down, volume_set, mute, unmute.",
                            "default": "status",
                        },
                        "device": {
                            "type": "string",
                            "description": "Nazwa urządzenia lub pokoju z wypowiedzi, np. 'klima w salonie', 'w sypialni', 'telewizor w salonie', 'amplituner', 'piec'.",
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Docelowa temperatura w stopniach Celsjusza dla klimatyzacji, np. 21, 22.5.",
                        },
                        "hvac_mode": {
                            "type": "string",
                            "enum": ["cool", "heat", "fan_only", "dry", "auto", "off"],
                            "description": "Tryb klimatyzacji: cool (chłodzenie), heat (grzanie), fan_only (nawiew), dry (osuszanie), auto, off.",
                        },
                        "level": {
                            "type": "number",
                            "description": "Poziom głośności 0-100 dla telewizora lub amplitunera przy action='volume_set'.",
                        },
                    },
                },
                handler=home,
                timeout_s=15.0,
            )
        )

        self.register(
            ToolSpec(
                name="remember",
                description_pl="Zapamiętuje fakt lub preferencję o użytkowniku w 3. osobie (np. Użytkownik lubi herbatę z cytryną).",
                description_en="Stores a personal fact about the user in local memory.",
                parameters={
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": "Treść faktu do zapamiętania np. moja siostra ma na imię Ola",
                        }
                    },
                    "required": ["fact"],
                },
                handler=remember,
            )
        )

        self.register(
            ToolSpec(
                name="forget",
                description_pl="Usuwa zapamiętany fakt o użytkowniku z lokalnej pamięci.",
                description_en="Removes a stored fact about the user from local memory.",
                parameters={
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": "Słowo kluczowe lub treść faktu do zapomnienia",
                        }
                    },
                    "required": ["fact"],
                },
                handler=forget,
            )
        )


def build_runtime_registry(settings: Settings) -> ToolRegistry:
    """Build a per-server ToolRegistry honouring `[spotify]`, `[home]`, and HA env vars."""
    registry = ToolRegistry()
    if settings.spotify.enabled:
        spec = registry.get("spotify")
        if spec is not None:
            spec.handler = partial(spotify, settings=settings)
            spec.timeout_s = max(12.0, settings.spotify.timeout_s * 4.0)
    else:
        registry.unregister("spotify")

    if settings.home.enabled and home_configured():
        spec = registry.get("home")
        if spec is not None:
            spec.handler = partial(home, settings=settings)
            spec.timeout_s = max(15.0, settings.home.timeout_s * 4.0 + 2.0)
    else:
        registry.unregister("home")

    return registry


default_registry = ToolRegistry()
