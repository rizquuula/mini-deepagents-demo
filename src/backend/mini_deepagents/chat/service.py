"""The chat service: one chat turn, from the token budget to the saved exchange.

The service is pure domain code. It imports no framework and logs nothing.
The HTTP adapter logs from the data that the service returns.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass

from mini_deepagents.chat.context import fit_messages
from mini_deepagents.chat.domain import (
    ChatEvent,
    Message,
    ReplyDone,
    Session,
    StreamEnd,
    TextDelta,
)
from mini_deepagents.chat.ports import ChatModel, ChatRepository, TokenEstimator
from mini_deepagents.errors import ModelError

__all__ = ["ChatService", "PreparedTurn"]


@dataclass(frozen=True, slots=True)
class PreparedTurn:
    """A chat turn that fits the token budget and is ready to stream.

    ``window`` holds the messages to send, oldest first. It ends with the
    new user message. ``history_messages`` counts the stored messages before
    this turn. ``estimated_tokens`` is the summed estimate of ``window``.
    """

    session_id: int
    content: str
    window: tuple[Message, ...]
    history_messages: int
    estimated_tokens: int
    budget_tokens: int


class ChatService:
    """Run chat turns against a repository and a streaming model."""

    def __init__(
        self,
        repository: ChatRepository,
        model: ChatModel,
        estimator: TokenEstimator,
        *,
        input_budget: int,
        max_output_tokens: int,
        session_key_prefix: str = "mini-deepagents-",
    ) -> None:
        self._repository = repository
        self._model = model
        self._estimator = estimator
        self._input_budget = input_budget
        self._max_output_tokens = max_output_tokens
        self._session_key_prefix = session_key_prefix

    async def create_session(self) -> Session:
        """Create a new empty session and return it."""
        return await asyncio.to_thread(self._repository.create_session)

    async def list_sessions(self) -> list[Session]:
        """Return all sessions, newest first."""
        return await asyncio.to_thread(self._repository.list_sessions)

    async def get_messages(self, session_id: int) -> list[Message]:
        """Return the messages of a session, oldest first.

        Raises:
            SessionNotFoundError: No session has ``session_id``.
        """
        return await asyncio.to_thread(self._repository.get_messages, session_id)

    async def prepare(self, session_id: int, content: str) -> PreparedTurn:
        """Fit the history plus the new user message into the input budget.

        The call writes nothing.

        Raises:
            SessionNotFoundError: No session has ``session_id``.
            ContextOverflowError: The new message alone exceeds the budget.
        """
        history = await asyncio.to_thread(self._repository.get_messages, session_id)
        messages = [*history, Message(role="user", content=content)]
        window = tuple(fit_messages(messages, self._input_budget, self._estimator))
        return PreparedTurn(
            session_id=session_id,
            content=content,
            window=window,
            history_messages=len(history),
            estimated_tokens=sum(self._estimator.estimate(m) for m in window),
            budget_tokens=self._input_budget,
        )

    async def stream_reply(self, turn: PreparedTurn) -> AsyncGenerator[ChatEvent, None]:
        """Stream the model reply, then save the exchange and yield ``ReplyDone``.

        The service saves the exchange only after the model stream completes.
        An early close by the consumer, a cancellation, or a ``ModelError``
        saves nothing. An early close also closes the model stream.

        Raises:
            ModelError: The model call failed.
        """
        session_key = f"{self._session_key_prefix}{turn.session_id}"
        events = self._model.stream(
            turn.window, max_tokens=self._max_output_tokens, session_key=session_key
        )
        parts: list[str] = []
        async with aclosing(events):
            async for event in events:
                if isinstance(event, TextDelta):
                    parts.append(event.text)
                    yield event
                elif isinstance(event, StreamEnd):
                    reply = "".join(parts)
                    await asyncio.to_thread(
                        self._repository.add_exchange,
                        turn.session_id,
                        turn.content,
                        reply,
                    )
                    yield ReplyDone(reply_chars=len(reply), usage=event.usage)
                    return
        raise ModelError("model stream ended without StreamEnd")
