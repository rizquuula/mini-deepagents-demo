from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from mini_deepagents_demo.config import Settings
from mini_deepagents_demo.errors import ConfigError
from mini_deepagents_demo.tools import create_tools
from mini_deepagents_demo.tools.calculate import CalculateTool


def calculate(arguments: Mapping[str, object]) -> str:
    return asyncio.run(CalculateTool().run(arguments))


def test_spec_names_the_tool_and_requires_an_expression_string() -> None:
    spec = CalculateTool().spec

    assert spec.name == "calculate"
    assert spec.parameters == {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    }


@pytest.mark.parametrize(
    ("expression", "result"),
    [
        ("1234 * 5678", "7006652"),
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("10 - 12", "-2"),
        ("7 / 2", "3.5"),
        ("7 // 2", "3"),
        ("-7 % 3", "2"),
        ("2 ** 10", "1024"),
        ("2 ** -1", "0.5"),
        ("-(+3)", "-3"),
        ("1 / 3", "0.333333333333"),
        ("0.1 + 0.2", "0.3"),
        ("2 ** 0.5", "1.41421356237"),
        ("10 ** 30", "1000000000000000000000000000000"),
    ],
)
def test_arithmetic_returns_the_result_as_text(expression: str, result: str) -> None:
    assert calculate({"expression": expression}) == result


@pytest.mark.parametrize(
    "expression",
    [
        "x",
        "abs(-1)",
        "(1).real",
        "'a'",
        "True",
        "1 < 2",
        "1j",
        "(-8) ** 0.5",
        "1 & 2",
        "not 1",
        "1 +",
    ],
    ids=[
        "name",
        "call",
        "attribute",
        "string",
        "bool",
        "comparison",
        "complex",
        "complex-result",
        "bit-and",
        "not",
        "syntax-error",
    ],
)
def test_anything_but_arithmetic_is_rejected(expression: str) -> None:
    with pytest.raises(ValueError):
        calculate({"expression": expression})


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("1+" * 100 + "1", "too long"),
        ("2 ** 101", "exponent"),
        ("9**9**9**9", "exponent"),
        ("(10 ** 50) ** 90", "too large"),
        ("*".join(["10 ** 99"] * 13), "too large"),
    ],
    ids=["length", "exponent", "power-tower", "power-bits", "product-bits"],
)
def test_expressions_over_the_limits_are_rejected(expression: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        calculate({"expression": expression})


@pytest.mark.parametrize("expression", ["1 / 0", "1 // 0", "1 % 0"])
def test_division_by_zero_raises(expression: str) -> None:
    with pytest.raises(ZeroDivisionError):
        calculate({"expression": expression})


@pytest.mark.parametrize(
    "arguments",
    [{}, {"expression": 5}, {"expr": "1 + 1"}],
    ids=["missing", "not-a-string", "wrong-name"],
)
def test_a_missing_or_non_string_expression_is_rejected(
    arguments: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError, match="expression"):
        calculate(arguments)


# --- Registry ---------------------------------------------------------------


def test_registry_builds_the_named_tools() -> None:
    tools = create_tools(["calculate"], Settings(api_key="k"))

    assert [spec.name for spec in tools.specs()] == ["calculate"]


def test_registry_builds_an_empty_set_for_no_names() -> None:
    assert len(create_tools([], Settings(api_key="k"))) == 0


def test_registry_rejects_an_unknown_tool_and_lists_known_names() -> None:
    with pytest.raises(ConfigError, match="calculate"):
        create_tools(["no-such-tool"], Settings(api_key="k"))
