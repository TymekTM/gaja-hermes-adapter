from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from gaja.tools.base import ToolResult
from gaja.tools.calc_tool import calculate
from gaja.tools.convert_tool import convert_units
from gaja.tools.currency_tool import exchange_rate
from gaja.tools.holiday_tool import next_holiday
from gaja.tools.system_tool import server_status
from gaja.tools.time_tool import get_time
from gaja.tools.weather_tool import get_weather
from gaja.tools.wiki_tool import wiki_summary

log = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    description_pl: str
    description_en: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[ToolResult]]

    def schema(self, lang: str = "pl") -> dict[str, Any]:
        desc = self.description_en if lang == "en" else self.description_pl
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
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
        return [spec.schema(lang) for spec in self._tools.values()]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        lang: str = "pl",
        timeout_s: float = 2.5,
    ) -> ToolResult:
        spec = self.get(name)
        if spec is None:
            err = f"Nieznane narzędzie: {name}"
            speak = f"Nie znam narzędzia {name}." if lang == "pl" else f"Unknown tool {name}."
            return ToolResult(data={"error": err}, speak=speak, error=err)

        args = dict(arguments or {})
        handler_sig = inspect.signature(spec.handler)

        call_kwargs: dict[str, Any] = {}
        for param in handler_sig.parameters.values():
            if param.name == "lang":
                call_kwargs["lang"] = lang
            elif param.name in args:
                call_kwargs[param.name] = args[param.name]

        try:
            return await asyncio.wait_for(spec.handler(**call_kwargs), timeout=timeout_s)
        except asyncio.TimeoutError:
            err = f"Przekroczono limit czasu ({timeout_s}s) dla {name}"
            speak = "Przekroczono limit czasu wykonania." if lang == "pl" else "Tool execution timed out."
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
                description_pl="Zwraca aktualną datę, godzinę, dzień tygodnia i strefę czasową. Używaj przy pytaniach o aktualny czas, datę, jaki dziś dzień.",
                description_en="Returns current date, time, weekday, and timezone (Europe/Warsaw).",
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Strefa czasowa, np. Europe/Warsaw",
                            "default": "Europe/Warsaw",
                        }
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
                            "description": "Nazwa miejscowości, np. Warszawa, Kraków, Londyn.",
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
                description_pl="Przelicza jednostki miary (długość, masa, temperatura, objętość, prędkość, czas).",
                description_en="Converts units of measurement (length, mass, temperature, volume, speed, time).",
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "Wartość liczbowa do przeliczenia",
                        },
                        "from_unit": {
                            "type": "string",
                            "description": "Jednostka źródłowa np. km, kg, C, mil, funt",
                        },
                        "to_unit": {
                            "type": "string",
                            "description": "Jednostka docelowa np. mil, funt, F, km, kg",
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
                description_pl="Sprawdza najbliższe polskie święto państwowe lub kościelne (dzień ustawowo wolny od pracy) i ile dni do niego zostało.",
                description_en="Computes the next statutory Polish public holiday and days remaining.",
                parameters={"type": "object", "properties": {}},
                handler=next_holiday,
            )
        )


default_registry = ToolRegistry()
