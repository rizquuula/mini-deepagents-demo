"""Load the app settings from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_API_KEY_VAR = "OPENCODE_API_KEY"


class ConfigError(Exception):
    """A setting is missing or not valid. The message never holds the API key."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://opencode.ai/inference/go/openai/v1"
    model: str = "deepseek-v4.1-flash"
    context_window: int = 8192
    max_output_tokens: int = 1024
    db_path: Path = Path("chat.db")

    @property
    def input_budget(self) -> int:
        """Return the token budget for the prompt: the window minus the output reserve."""
        return self.context_window - self.max_output_tokens

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``env`` (``os.environ`` when None).

        A missing or empty optional variable uses the default.

        Raises:
            ConfigError: The key is missing, an int does not parse or is not
                positive, or the output reserve does not fit the window.
        """
        source = os.environ if env is None else env
        api_key = source.get(_API_KEY_VAR, "")
        if not api_key:
            raise ConfigError(f"{_API_KEY_VAR} is missing or empty")

        defaults = cls(api_key=api_key)
        context_window = _positive_int(source, "CONTEXT_WINDOW", defaults.context_window)
        max_output_tokens = _positive_int(
            source, "MAX_OUTPUT_TOKENS", defaults.max_output_tokens
        )
        if max_output_tokens >= context_window:
            raise ConfigError(
                f"MAX_OUTPUT_TOKENS ({max_output_tokens}) must be less than "
                f"CONTEXT_WINDOW ({context_window})"
            )

        db_path = source.get("DB_PATH")
        return cls(
            api_key=api_key,
            base_url=source.get("OPENCODE_BASE_URL") or defaults.base_url,
            model=source.get("MODEL") or defaults.model,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            db_path=Path(db_path) if db_path else defaults.db_path,
        )


def _positive_int(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be greater than 0")
    return value
