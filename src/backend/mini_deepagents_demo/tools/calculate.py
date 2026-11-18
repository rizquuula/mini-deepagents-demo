"""The ``calculate`` tool: safe arithmetic on one expression.

The tool parses the expression with ``ast`` and walks the tree itself. It
never calls ``eval`` or ``exec``. It allows only numbers, the arithmetic
operators, and parentheses, and it caps the size of each number.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping
from typing import Final, TypeAlias

from mini_deepagents import ToolSpec

__all__ = ["CalculateTool"]

Number: TypeAlias = int | float

MAX_EXPRESSION_CHARS: Final = 200
MAX_EXPONENT: Final = 100
MAX_INT_BITS: Final = 4000

SPEC: Final = ToolSpec(
    name="calculate",
    description="Evaluate an arithmetic expression with + - * / // % ** and parentheses.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
)

_BINARY_OPS: Final[Mapping[type[ast.operator], Callable[[Number, Number], Number]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: Final[Mapping[type[ast.unaryop], Callable[[Number], Number]]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculateTool:
    """A ``Tool`` that evaluates one arithmetic expression."""

    @property
    def spec(self) -> ToolSpec:
        return SPEC

    async def run(self, arguments: Mapping[str, object]) -> str:
        """Return the value of ``arguments["expression"]`` as text.

        Raises:
            ValueError: The expression is missing, is not a string, is too
                long, uses a syntax other than arithmetic, or makes a number
                over the size limits.
            ZeroDivisionError: The expression divides by zero.
            OverflowError: A float result is out of range.
        """
        if "expression" not in arguments:
            raise ValueError("expression is required")
        expression = arguments["expression"]
        if not isinstance(expression, str):
            raise ValueError("expression must be a string")
        return _format(evaluate(expression))


def evaluate(expression: str) -> Number:
    """Return the value of an arithmetic expression. See ``CalculateTool.run``."""
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression is too long; max {MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError("expression is not valid syntax") from exc
    return _evaluate(tree.body)


def _evaluate(node: ast.expr) -> Number:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value  # type: ignore[no-any-return]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            _check_power(left, right)
        return _check_size(_BINARY_OPS[type(node.op)](left, right))
    raise ValueError(f"unsupported syntax: {_syntax_name(node)}")


def _check_power(base: Number, exponent: Number) -> None:
    if abs(exponent) > MAX_EXPONENT:
        raise ValueError(f"exponent is too large; max {MAX_EXPONENT}")
    if isinstance(base, int) and isinstance(exponent, int):
        if base.bit_length() * exponent > MAX_INT_BITS:
            raise ValueError(f"result is too large; max {MAX_INT_BITS} bits")


def _check_size(value: object) -> Number:
    if not isinstance(value, (int, float)):
        raise ValueError("result is not a real number")
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise ValueError(f"result is too large; max {MAX_INT_BITS} bits")
    return value


def _syntax_name(node: ast.expr) -> str:
    op = getattr(node, "op", None)
    return type(op).__name__ if op is not None else type(node).__name__


def _format(value: Number) -> str:
    return str(value) if isinstance(value, int) else format(value, ".12g")
