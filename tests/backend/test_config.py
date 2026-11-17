from __future__ import annotations

from pathlib import Path

import pytest

from mini_deepagents.config import ConfigError, Settings

API_KEY = "sk-test-secret-123"


def test_only_the_key_set_uses_every_default() -> None:
    settings = Settings.from_env({"OPENCODE_API_KEY": API_KEY})

    assert settings.api_key == API_KEY
    assert settings.base_url == "https://opencode.ai/inference/go/openai/v1"
    assert settings.model == "deepseek-v4.1-flash"
    assert settings.context_window == 8192
    assert settings.max_output_tokens == 1024
    assert settings.db_path == Path("chat.db")
    assert settings.input_budget == 7168
    assert settings.log_level == "INFO"
    assert settings.log_file == Path("logs/app.log")
    assert settings.llm_provider == "opencode"
    assert settings.storage_backend == "sqlite"


def test_empty_optional_vars_use_the_default() -> None:
    settings = Settings.from_env(
        {
            "OPENCODE_API_KEY": API_KEY,
            "MODEL": "",
            "CONTEXT_WINDOW": "",
            "DB_PATH": "",
            "LOG_LEVEL": "",
            "LOG_FILE": "",
            "LLM_PROVIDER": "",
            "STORAGE_BACKEND": "",
        }
    )

    assert settings == Settings(api_key=API_KEY)


def test_every_var_set_overrides_every_default() -> None:
    settings = Settings.from_env(
        {
            "OPENCODE_API_KEY": API_KEY,
            "OPENCODE_BASE_URL": "http://localhost:9000/v1",
            "MODEL": "other-model",
            "CONTEXT_WINDOW": "4096",
            "MAX_OUTPUT_TOKENS": "512",
            "DB_PATH": "/tmp/other.db",
            "LOG_LEVEL": "debug",
            "LOG_FILE": "/tmp/other.log",
            "LLM_PROVIDER": "other-provider",
            "STORAGE_BACKEND": "other-backend",
        }
    )

    assert settings == Settings(
        api_key=API_KEY,
        base_url="http://localhost:9000/v1",
        model="other-model",
        context_window=4096,
        max_output_tokens=512,
        db_path=Path("/tmp/other.db"),
        log_level="DEBUG",
        log_file=Path("/tmp/other.log"),
        llm_provider="other-provider",
        storage_backend="other-backend",
    )
    assert settings.input_budget == 3584


@pytest.mark.parametrize("env", [{}, {"OPENCODE_API_KEY": ""}], ids=["missing", "empty"])
def test_missing_or_empty_key_raises(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError, match="OPENCODE_API_KEY"):
        Settings.from_env(env)


@pytest.mark.parametrize(
    "overrides",
    [
        {"CONTEXT_WINDOW": "big"},
        {"MAX_OUTPUT_TOKENS": "1.5"},
        {"CONTEXT_WINDOW": "0"},
        {"MAX_OUTPUT_TOKENS": "-1"},
        {"CONTEXT_WINDOW": "1024", "MAX_OUTPUT_TOKENS": "1024"},
        {"CONTEXT_WINDOW": "1000", "MAX_OUTPUT_TOKENS": "2000"},
        {"LOG_LEVEL": "verbose"},
    ],
    ids=[
        "non-int",
        "non-int-float",
        "zero",
        "negative",
        "output-equals-window",
        "output-above-window",
        "unknown-log-level",
    ],
)
def test_invalid_settings_raise_without_the_key(overrides: dict[str, str]) -> None:
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env({"OPENCODE_API_KEY": API_KEY, **overrides})

    assert API_KEY not in str(excinfo.value)
