"""Storage adapters for ``ChatRepository``, selected by name.

To add a backend, add one module that implements ``ChatRepository`` and add
one entry to ``BACKENDS``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias

from mini_deepagents import ChatRepository

from mini_deepagents_demo.adapters.storage.sqlite import SqliteChatRepository
from mini_deepagents_demo.config import Settings
from mini_deepagents_demo.errors import ConfigError

RepositoryFactory: TypeAlias = Callable[[Settings], ChatRepository]

BACKENDS: Final[Mapping[str, RepositoryFactory]] = MappingProxyType(
    {
        "sqlite": lambda settings: SqliteChatRepository(settings.db_path),
    }
)


def create_repository(name: str, settings: Settings) -> ChatRepository:
    """Build the repository that ``BACKENDS`` registers under ``name``.

    Raises:
        ConfigError: No backend has ``name``. The message lists the known names.
    """
    try:
        factory = BACKENDS[name]
    except KeyError:
        known = ", ".join(sorted(BACKENDS))
        raise ConfigError(
            f"unknown storage backend {name!r}; known backends: {known}"
        ) from None
    return factory(settings)
