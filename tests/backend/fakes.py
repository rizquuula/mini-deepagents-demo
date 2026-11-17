"""Fakes for the chat ports. Contract tests run them next to the real adapters."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from datetime import datetime, timedelta, timezone

from mini_deepagents.chat import Message, ModelEvent, Session, StreamEnd, TextDelta, Usage
from mini_deepagents.errors import ModelError, SessionNotFoundError

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeChatModel:
    """A ``ChatModel`` that streams fixed deltas and records each call."""

    def __init__(
        self,
        deltas: Sequence[str] = ("Hello", " world"),
        *,
        usage: Usage = Usage(10, 2, 12),
        error: Exception | None = None,
        fail_after: int = 0,
    ) -> None:
        self.deltas = tuple(deltas)
        self.usage = usage
        self.error = error
        self.fail_after = fail_after
        self.calls: list[tuple[list[Message], int, str]] = []
        self.closed_early = False
        self.aclose_calls = 0

    async def stream(
        self, messages: Sequence[Message], *, max_tokens: int, session_key: str
    ) -> AsyncGenerator[ModelEvent, None]:
        self.calls.append((list(messages), max_tokens, session_key))
        finished = False
        try:
            for index, text in enumerate(self.deltas):
                if self.error is not None and index == self.fail_after:
                    raise self.error
                yield TextDelta(text)
            if self.error is not None:
                raise self.error
            if not self.deltas:
                raise ModelError("empty reply")
            finished = True
            yield StreamEnd(self.usage)
        except GeneratorExit:
            if not finished:
                self.closed_early = True
            raise

    async def aclose(self) -> None:
        self.aclose_calls += 1


class InMemoryChatRepository:
    """A ``ChatRepository`` that keeps sessions in a dict."""

    def __init__(self) -> None:
        self._sessions: dict[int, Session] = {}
        self._messages: dict[int, list[Message]] = {}

    def create_session(self) -> Session:
        session_id = len(self._sessions) + 1
        created_at = (_EPOCH + timedelta(seconds=session_id)).isoformat()
        session = Session(id=session_id, created_at=created_at)
        self._sessions[session_id] = session
        self._messages[session_id] = []
        return session

    def list_sessions(self) -> list[Session]:
        return sorted(self._sessions.values(), key=lambda s: s.id, reverse=True)

    def get_messages(self, session_id: int) -> list[Message]:
        return list(self._messages_of(session_id))

    def add_exchange(
        self, session_id: int, user_content: str, assistant_content: str
    ) -> None:
        self._messages_of(session_id).extend(
            [
                Message(role="user", content=user_content),
                Message(role="assistant", content=assistant_content),
            ]
        )

    def _messages_of(self, session_id: int) -> list[Message]:
        try:
            return self._messages[session_id]
        except KeyError:
            raise SessionNotFoundError(f"session {session_id} not found") from None
