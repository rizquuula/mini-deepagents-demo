"""The errors that only the demo app raises.

The mini-deepagents framework holds the other errors, such as
``SessionNotFoundError``, ``ModelError``, and ``ContextOverflowError``.
Import them from ``mini_deepagents``.
"""

from __future__ import annotations

from mini_deepagents import MiniDeepAgentsError


class ConfigError(MiniDeepAgentsError):
    """The configuration is missing or invalid."""
