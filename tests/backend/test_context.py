from __future__ import annotations

import pytest

from mini_deepagents.chat.context import (
    ContextOverflowError,
    estimate_tokens,
    fit_to_budget,
)


def user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def assistant(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}


# Each message below estimates to 4 + ceil(9 / 3) = 7 tokens.
CONVERSATION = [
    user("message 1"),
    assistant("message 2"),
    user("message 3"),
    assistant("message 4"),
    user("message 5"),
]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", 4),
        ("abc", 5),
        ("abcd", 6),
        ("é", 5),  # 2 UTF-8 bytes
        ("€", 5),  # 3 UTF-8 bytes
        ("€€", 6),  # 6 UTF-8 bytes
        ("x" * 3000, 1004),
    ],
)
def test_estimate_tokens_uses_ceiling_of_utf8_bytes_plus_overhead(
    content: str, expected: int
) -> None:
    assert estimate_tokens(user(content)) == expected


def test_all_messages_fit_returns_all_in_order() -> None:
    result = fit_to_budget(CONVERSATION, budget=1000)

    assert result == CONVERSATION
    assert all(a is b for a, b in zip(result, CONVERSATION, strict=True))


def test_over_budget_drops_oldest_and_keeps_newest() -> None:
    result = fit_to_budget(CONVERSATION, budget=21)

    assert result == CONVERSATION[2:]


def test_does_not_skip_a_large_message_to_fit_an_older_one() -> None:
    small_old = user("a")
    large = user("x" * 300)
    messages = [small_old, large, user("b"), user("c")]

    result = fit_to_budget(messages, budget=20)

    assert result == messages[2:]


@pytest.mark.parametrize("budget", [7, 13, 14, 20, 21, 27, 28, 35, 1000])
def test_result_estimate_never_exceeds_budget(budget: int) -> None:
    result = fit_to_budget(CONVERSATION, budget=budget)

    assert sum(estimate_tokens(m) for m in result) <= budget


@pytest.mark.parametrize("budget", [7, 14, 21, 28, 35])
def test_trimmed_window_starts_with_user_message(budget: int) -> None:
    result = fit_to_budget(CONVERSATION, budget=budget)

    assert result
    assert result[0]["role"] == "user"


def test_newest_message_over_budget_raises() -> None:
    with pytest.raises(ContextOverflowError):
        fit_to_budget([user("short"), user("x" * 300)], budget=50)


def test_empty_input_raises() -> None:
    with pytest.raises(ContextOverflowError):
        fit_to_budget([], budget=1000)


def test_input_list_is_not_mutated() -> None:
    messages = [dict(m) for m in CONVERSATION]
    snapshot = [dict(m) for m in messages]

    fit_to_budget(messages, budget=14)

    assert messages == snapshot
