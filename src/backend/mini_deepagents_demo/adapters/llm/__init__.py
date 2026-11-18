"""The registry of chat model providers.

To add a provider, add one adapter module and one entry in ``PROVIDERS``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from mini_deepagents_demo.adapters.llm.opencode import OpenCodeChatModel
from mini_deepagents_demo.chat import ChatModel
from mini_deepagents_demo.config import Settings
from mini_deepagents_demo.errors import ConfigError

ChatModelFactory = Callable[[Settings], ChatModel]

PROVIDERS: Mapping[str, ChatModelFactory] = MappingProxyType(
    {
        "opencode": lambda s: OpenCodeChatModel(s.base_url, s.api_key, s.model),
    }
)


def create_chat_model(name: str, settings: Settings) -> ChatModel:
    """Build the chat model that ``name`` selects.

    Raises:
        ConfigError: No provider has ``name``. The message lists the known names.
    """
    try:
        factory = PROVIDERS[name]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise ConfigError(f"unknown model provider {name!r}; known: {known}") from None
    return factory(settings)
