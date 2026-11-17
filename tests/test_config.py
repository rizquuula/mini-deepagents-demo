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


def test_empty_optional_vars_use_the_default() -> None:
    settings = Settings.from_env(
        {"OPENCODE_API_KEY": API_KEY, "MODEL": "", "CONTEXT_WINDOW": "", "DB_PATH": ""}
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
        }
    )

    assert settings == Settings(
        api_key=API_KEY,
        base_url="http://localhost:9000/v1",
        model="other-model",
        context_window=4096,
        max_output_tokens=512,
        db_path=Path("/tmp/other.db"),
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
    ],
    ids=[
        "non-int",
        "non-int-float",
        "zero",
        "negative",
        "output-equals-window",
        "output-above-window",
    ],
)
def test_invalid_token_settings_raise_without_the_key(overrides: dict[str, str]) -> None:
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env({"OPENCODE_API_KEY": API_KEY, **overrides})

    assert API_KEY not in str(excinfo.value)
