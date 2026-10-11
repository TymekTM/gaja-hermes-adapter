from __future__ import annotations

import ast
import math
import re
from typing import Any

from gaja.text.normalize import normalize_for_tts
from gaja.tools.base import ToolResult, format_number_pl

ALLOWED_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}

ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "pow": pow,
}


def _preprocess_expression(expr: str) -> str:
    s = expr.strip()
    # Normalize unicode multiplication / division
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/")
    # Replace ^ with **
    s = s.replace("^", "**")

    # Polish & English percentage phrases:
    # "15% z 240", "15 % z 240", "15% of 240", "15 procent z 240", "15 procent od 240"
    pct_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:%|procent|percent)\s*(?:z|od|of|\*)\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    s = pct_pattern.sub(r"((\1 * \2) / 100)", s)

    # Standalone percentages: "20%" -> "(20/100)"
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1 / 100)", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(?:procent|percent)", r"(\1 / 100)", s)

    # Polish square root phrases: "pierwiastek z 16" -> "sqrt(16)"
    s = re.sub(r"pierwiastek\s+(?:z\s+|kwadratowy\s+z\s+)?(\d+(?:\.\d+)?)", r"sqrt(\1)", s, flags=re.IGNORECASE)

    # "12 x 5" where x is between digits
    s = re.sub(r"(\d+)\s*[xX]\s*(\d+)", r"\1 * \2", s)

    return s


def _eval_node(node: ast.AST, node_count: list[int]) -> Any:
    node_count[0] += 1
    if node_count[0] > 100:
        raise ValueError("Zbyt złożone wyrażenie")

    if isinstance(node, ast.Expression):
        return _eval_node(node.body, node_count)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Niedozwolona stała: {node.value}")

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in ALLOWED_UNARY_OPERATORS:
            operand = _eval_node(node.operand, node_count)
            return ALLOWED_UNARY_OPERATORS[op_type](operand)
        raise ValueError(f"Niedozwolony operator jednoargumentowy: {op_type}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            left = _eval_node(node.left, node_count)
            right = _eval_node(node.right, node_count)
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Dzielenie przez zero")
            if op_type == ast.Pow:
                if abs(right) > 1000 or (isinstance(left, (int, float)) and abs(left) > 1000 and right > 10):
                    raise ValueError("Wartość potęgi zbyt duża")
            return ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Niedozwolony operator dwuargumentowy: {op_type}")

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCTIONS:
            fn = ALLOWED_FUNCTIONS[node.func.id]
            args = [_eval_node(arg, node_count) for arg in node.args]
            if fn == math.sqrt and args and args[0] < 0:
                raise ValueError("Pierwiastek z liczby ujemnej")
            return fn(*args)
        raise ValueError("Niedozwolona funkcja")

    raise ValueError(f"Niedozwolony element składni: {type(node).__name__}")


async def calculate(expression: str, lang: str = "pl") -> ToolResult:
    """Safely evaluate mathematical expressions using AST."""
    if not expression or not expression.strip():
        err = "Brak wyrażenia do obliczenia"
        return ToolResult(data={"error": err}, speak="Nie podano wyrażenia do obliczenia.", error=err)

    raw_expr = expression.strip()
    try:
        clean_expr = _preprocess_expression(raw_expr)
        tree = ast.parse(clean_expr, mode="eval")
        node_count = [0]
        val = _eval_node(tree, node_count)

        if isinstance(val, float) and val.is_integer():
            val = int(val)
        elif isinstance(val, float):
            val = round(val, 6)

        data = {"expression": raw_expr, "result": val}
        if lang == "en":
            raw_speak = f"The result is {val}."
        else:
            raw_speak = f"Wynik to {format_number_pl(val)}."
        speak = normalize_for_tts(raw_speak, lang)
        return ToolResult(data=data, speak=speak)
    except ZeroDivisionError:
        err = "Dzielenie przez zero"
        speak = "Nie można dzielić przez zero." if lang == "pl" else "Cannot divide by zero."
        return ToolResult(data={"error": err}, speak=speak, error=err)
    except Exception as exc:
        err = f"Błąd obliczeń: {exc}"
        speak = "Nie udało się wykonać tych obliczeń." if lang == "pl" else "Could not perform the calculation."
        return ToolResult(data={"error": err}, speak=speak, error=err)
