from __future__ import annotations

from pathlib import Path

import pytest

from mini_deepagents_demo.config import ConfigError, Settings
from mini_deepagents_demo.tools import create_tools

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
    assert settings.enabled_tools == ("calculate",)
    assert settings.tool_overhead_tokens == 256


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
            "TOOL_OVERHEAD_TOKENS": "",
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
            "ENABLED_TOOLS": "other-tool",
            "TOOL_OVERHEAD_TOKENS": "300",
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
        enabled_tools=("other-tool",),
        tool_overhead_tokens=300,
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
        {"TOOL_OVERHEAD_TOKENS": "-1"},
        {"TOOL_OVERHEAD_TOKENS": "many"},
    ],
    ids=[
        "non-int",
        "non-int-float",
        "zero",
        "negative",
        "output-equals-window",
        "output-above-window",
        "unknown-log-level",
        "negative-tool-overhead",
        "non-int-tool-overhead",
    ],
)
def test_invalid_settings_raise_without_the_key(overrides: dict[str, str]) -> None:
    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env({"OPENCODE_API_KEY": API_KEY, **overrides})

    assert API_KEY not in str(excinfo.value)


@pytest.mark.parametrize(
    ("raw", "names"),
    [
        ("", ()),
        (" , ,", ()),
        (" calculate ", ("calculate",)),
        ("calculate, other ,,third", ("calculate", "other", "third")),
    ],
    ids=["empty", "only-separators", "padded", "list"],
)
def test_enabled_tools_split_on_commas_and_drop_empty_items(
    raw: str, names: tuple[str, ...]
) -> None:
    settings = Settings.from_env({"OPENCODE_API_KEY": API_KEY, "ENABLED_TOOLS": raw})

    assert settings.enabled_tools == names


def test_zero_tool_overhead_is_allowed() -> None:
    settings = Settings.from_env({"OPENCODE_API_KEY": API_KEY, "TOOL_OVERHEAD_TOKENS": "0"})

    assert settings.tool_overhead_tokens == 0


def test_an_unknown_enabled_tool_fails_when_the_tools_are_built() -> None:
    settings = Settings.from_env(
        {"OPENCODE_API_KEY": API_KEY, "ENABLED_TOOLS": "calculate,no-such-tool"}
    )

    with pytest.raises(ConfigError, match="no-such-tool"):
        create_tools(settings.enabled_tools, settings)
