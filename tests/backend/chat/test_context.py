from __future__ import annotations

import pytest

from mini_deepagents_demo.chat import ByteTokenEstimator, Message, fit_messages
from mini_deepagents_demo.errors import ContextOverflowError

ESTIMATOR = ByteTokenEstimator()


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str) -> Message:
    return Message(role="assistant", content=content)


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
        ("abcd", 6),
        ("é", 5),  # 2 UTF-8 bytes
        ("€€", 6),  # 6 UTF-8 bytes, 3 per char
    ],
)
def test_byte_estimator_uses_ceiling_of_utf8_bytes_plus_overhead(
    content: str, expected: int
) -> None:
    assert ESTIMATOR.estimate(user(content)) == expected


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(1000, CONVERSATION), (21, CONVERSATION[2:]), (7, CONVERSATION[4:])],
)
def test_fit_keeps_the_newest_messages_that_fit(
    budget: int, expected: list[Message]
) -> None:
    assert fit_messages(CONVERSATION, budget, ESTIMATOR) == expected


def test_fit_does_not_skip_a_large_message_to_fit_an_older_one() -> None:
    messages = [user("a"), user("x" * 300), user("b"), user("c")]

    assert fit_messages(messages, 20, ESTIMATOR) == messages[2:]


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(14, CONVERSATION[4:]), (28, CONVERSATION[2:])],
)
def test_fit_drops_leading_assistant_messages(
    budget: int, expected: list[Message]
) -> None:
    assert fit_messages(CONVERSATION, budget, ESTIMATOR) == expected


@pytest.mark.parametrize(
    ("messages", "estimated"),
    [([], 0), ([user("short"), user("x" * 300)], 104)],
)
def test_fit_raises_overflow_with_estimate_and_budget(
    messages: list[Message], estimated: int
) -> None:
    with pytest.raises(ContextOverflowError) as caught:
        fit_messages(messages, 50, ESTIMATOR)

    assert caught.value.estimated_tokens == estimated
    assert caught.value.budget_tokens == 50


def test_fit_does_not_mutate_the_input() -> None:
    messages = list(CONVERSATION)

    fit_messages(messages, 14, ESTIMATOR)

    assert messages == CONVERSATION
