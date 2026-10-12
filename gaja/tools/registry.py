from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from gaja.tools.base import ToolResult
from gaja.tools.calc_tool import calculate
from gaja.tools.convert_tool import convert_units
from gaja.tools.currency_tool import exchange_rate
from gaja.tools.holiday_tool import next_holiday
from gaja.tools.memory_tool import forget, remember
from gaja.tools.sun_tool import get_sun
from gaja.tools.system_tool import server_status
from gaja.tools.time_tool import get_time
from gaja.tools.timer_tool import manage_timer
from gaja.tools.weather_tool import get_weather
from gaja.tools.wiki_tool import wiki_summary

log = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    description_pl: str
    description_en: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

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

        try:
            handler_kwargs = dict(arguments)
            # Only inject lang if handler accepts it or needs it
            if "lang" not in handler_kwargs:
                handler_kwargs["lang"] = lang

            res = tool.handler(**handler_kwargs)
            if asyncio.iscoroutine(res):
                res = await asyncio.wait_for(res, timeout=timeout_s)

            if isinstance(res, ToolResult):
                return res
            return ToolResult(data={"result": res}, speak=str(res))
        except TimeoutError:
            err = f"Przekroczono limit czasu ({timeout_s}s) dla {name}"
            speak = "Sprawdzanie tych informacji zajęło zbyt dużo czasu." if lang == "pl" else "Checking this information timed out."
            log.warning("Tool %s timed out after %.1fs", name, timeout_s)
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
                            "description": "Nazwa miejscowości, domyślnie Katowice.",
                        },
                        "when": {
                            "type": "string",
                            "enum": ["now", "today", "tomorrow"],
                            "description": "Termin: now (aktualna), today (dzisiaj), tomorrow (jutro).",
                            "default": "now",
                        },
                    },
                },
                handler=get_weather,
            )
        )

        self.register(
            ToolSpec(
                name="get_sun",
                description_pl="Oblicza offline wschód i zachód słońca oraz długość dnia dla danej miejscowości (domyślnie Katowice).",
                description_en="Computes offline sunrise, sunset, and day length for Katowice or a location.",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Nazwa miejscowości, domyślnie Katowice.",
                        },
                        "date": {
                            "type": "string",
                            "description": "Termin: today lub data YYYY-MM-DD.",
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
                name="remember",
                description_pl="Zapamiętuje fakt o użytkowniku lub jego rodzinie w lokalnej pamięci.",
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


default_registry = ToolRegistry()
