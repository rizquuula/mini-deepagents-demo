"""The error hierarchy of the app.

Adapters translate vendor errors into these types, so no vendor exception
reaches the domain or the HTTP layer.
"""

from __future__ import annotations


class MiniDeepAgentsError(Exception):
    """Base for every error this app raises on purpose."""


class ConfigError(MiniDeepAgentsError):
    """The configuration is missing or invalid."""


class SessionNotFoundError(MiniDeepAgentsError):
    """No chat session has the requested id."""


class ModelError(MiniDeepAgentsError):
    """The model call failed or returned no usable reply."""


class ContextOverflowError(MiniDeepAgentsError):
    """The messages to send do not fit the token budget."""

    def __init__(
        self,
        message: str,
        *,
        estimated_tokens: int | None = None,
        budget_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.estimated_tokens = estimated_tokens
        self.budget_tokens = budget_tokens
