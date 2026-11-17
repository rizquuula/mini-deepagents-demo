"""Fit chat history into a token budget.

The token estimate is deliberately conservative. English text averages about
4 UTF-8 bytes per token, so a divisor of 3 overestimates the count. An
overestimate trims a little more history than necessary, but it keeps each
request at or below the model's real context limit without a tokenizer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from mini_deepagents.chat.domain import Message
from mini_deepagents.chat.ports import TokenEstimator
from mini_deepagents.errors import ContextOverflowError

__all__ = [
    "BYTES_PER_TOKEN",
    "MESSAGE_OVERHEAD_TOKENS",
    "ByteTokenEstimator",
    "ContextOverflowError",
    "estimate_tokens",
    "fit_messages",
    "fit_to_budget",
]

BYTES_PER_TOKEN: Final = 3
MESSAGE_OVERHEAD_TOKENS: Final = 4


class ByteTokenEstimator:
    """Estimate tokens from the UTF-8 byte count. Implements ``TokenEstimator``."""

    def estimate(self, message: Message) -> int:
        """Return ceil(bytes / BYTES_PER_TOKEN) plus the per-message overhead."""
        byte_count = len(message.content.encode("utf-8"))
        return -(-byte_count // BYTES_PER_TOKEN) + MESSAGE_OVERHEAD_TOKENS


def fit_messages(
    messages: Sequence[Message], budget: int, estimator: TokenEstimator
) -> list[Message]:
    """Return the longest run of newest messages that fits ``budget``.

    The input and the output are chronological, oldest first. The function
    never skips a message. The window always starts with a ``"user"``
    message. The input is not mutated.

    Raises:
        ContextOverflowError: The input is empty, or the newest message
            alone exceeds ``budget``.
    """
    if not messages:
        raise ContextOverflowError(
            "no messages to send", estimated_tokens=0, budget_tokens=budget
        )

    kept: list[Message] = []
    used = 0
    for message in reversed(messages):
        cost = estimator.estimate(message)
        if used + cost > budget:
            break
        kept.append(message)
        used += cost

    if not kept:
        newest_cost = estimator.estimate(messages[-1])
        raise ContextOverflowError(
            f"newest message needs {newest_cost} tokens; budget is {budget}",
            estimated_tokens=newest_cost,
            budget_tokens=budget,
        )

    kept.reverse()
    first_user = next(
        (index for index, message in enumerate(kept) if message.role == "user"),
        len(kept),
    )
    return kept[first_user:]


def estimate_tokens(message: Mapping[str, str]) -> int:
    """Return a conservative token estimate for one chat message."""
    byte_count = len(message["content"].encode("utf-8"))
    return -(-byte_count // BYTES_PER_TOKEN) + MESSAGE_OVERHEAD_TOKENS


def fit_to_budget(
    messages: Sequence[Mapping[str, str]], budget: int
) -> list[Mapping[str, str]]:
    """Return the longest run of newest messages that fits ``budget``.

    The input and the output are chronological, oldest first. The window
    always starts with a ``"user"`` message.

    Raises:
        ContextOverflowError: The input is empty, or the newest message
            alone exceeds ``budget``.
    """
    if not messages:
        raise ContextOverflowError("no messages to send")

    kept: list[Mapping[str, str]] = []
    used = 0
    for message in reversed(messages):
        cost = estimate_tokens(message)
        if used + cost > budget:
            break
        kept.append(message)
        used += cost

    if not kept:
        raise ContextOverflowError(
            f"newest message needs {estimate_tokens(messages[-1])} tokens; "
            f"budget is {budget}"
        )

    kept.reverse()
    first_user = next(
        (index for index, message in enumerate(kept) if message["role"] == "user"),
        len(kept),
    )
    return kept[first_user:]
