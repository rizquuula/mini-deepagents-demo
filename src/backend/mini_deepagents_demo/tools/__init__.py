"""The registry of tools that the model may call.

To add a tool, add one module in this package and one entry in ``TOOLS``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType

from mini_deepagents import Tool, ToolSet

from mini_deepagents_demo.config import Settings
from mini_deepagents_demo.errors import ConfigError
from mini_deepagents_demo.tools.calculate import CalculateTool

__all__ = ["TOOLS", "ToolFactory", "create_tools"]

ToolFactory = Callable[[Settings], Tool]

TOOLS: Mapping[str, ToolFactory] = MappingProxyType(
    {
        "calculate": lambda _settings: CalculateTool(),
    }
)


def create_tools(names: Iterable[str], settings: Settings) -> ToolSet:
    """Build the tool set that ``names`` selects, in the given order.

    Raises:
        ConfigError: No tool has one of ``names``. The message lists the known names.
        ToolConfigError: ``names`` holds a name twice.
    """
    tools: list[Tool] = []
    for name in names:
        try:
            factory = TOOLS[name]
        except KeyError:
            known = ", ".join(sorted(TOOLS))
            raise ConfigError(f"unknown tool {name!r}; known: {known}") from None
        tools.append(factory(settings))
    return ToolSet(tools)
