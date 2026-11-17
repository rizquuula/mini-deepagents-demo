"""Fit chat history into a token budget.

The token estimate is deliberately conservative. English text averages about
4 UTF-8 bytes per token, so a divisor of 3 overestimates the count. An
overestimate trims a little more history than necessary, but it keeps each
request at or below the model's real context limit without a tokenizer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

BYTES_PER_TOKEN: Final = 3
MESSAGE_OVERHEAD_TOKENS: Final = 4


class ContextOverflowError(Exception):
    """The newest message alone does not fit the budget."""


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
