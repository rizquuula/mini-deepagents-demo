"""Value types of the chat domain. They hold data only and import no framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

Role: TypeAlias = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """One chat message."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Session:
    """One chat session. ``created_at`` is an ISO 8601 string."""

    id: int
    created_at: str


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts that the model reports. A field is None when unknown."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class TextDelta:
    """One piece of reply text from the model."""

    text: str


@dataclass(frozen=True, slots=True)
class StreamEnd:
    """The last event of a model stream."""

    usage: Usage


@dataclass(frozen=True, slots=True)
class ReplyDone:
    """The last event that the chat service yields."""

    reply_chars: int
    usage: Usage


ModelEvent: TypeAlias = TextDelta | StreamEnd
ChatEvent: TypeAlias = TextDelta | ReplyDone
