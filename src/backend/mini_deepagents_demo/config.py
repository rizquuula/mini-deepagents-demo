"""Load the app settings from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mini_deepagents_demo.errors import ConfigError

__all__ = ["ConfigError", "Settings"]

_API_KEY_VAR = "OPENCODE_API_KEY"
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://opencode.ai/inference/go/openai/v1"
    model: str = "deepseek-v4.1-flash"
    context_window: int = 8192
    max_output_tokens: int = 1024
    db_path: Path = Path("chat.db")
    log_level: str = "INFO"
    log_file: Path = Path("logs/app.log")
    llm_provider: str = "opencode"
    storage_backend: str = "sqlite"

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
                positive, the output reserve does not fit the window, or
                ``LOG_LEVEL`` is not a known level.
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

        log_level = (source.get("LOG_LEVEL") or defaults.log_level).upper()
        if log_level not in _LOG_LEVELS:
            raise ConfigError(f"LOG_LEVEL must be one of {', '.join(_LOG_LEVELS)}")

        db_path = source.get("DB_PATH")
        log_file = source.get("LOG_FILE")
        return cls(
            api_key=api_key,
            base_url=source.get("OPENCODE_BASE_URL") or defaults.base_url,
            model=source.get("MODEL") or defaults.model,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            db_path=Path(db_path) if db_path else defaults.db_path,
            log_level=log_level,
            log_file=Path(log_file) if log_file else defaults.log_file,
            llm_provider=source.get("LLM_PROVIDER") or defaults.llm_provider,
            storage_backend=source.get("STORAGE_BACKEND") or defaults.storage_backend,
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
