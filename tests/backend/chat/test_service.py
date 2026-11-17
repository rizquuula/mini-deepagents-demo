from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from mini_deepagents.chat import (
    ByteTokenEstimator,
    ChatEvent,
    ChatService,
    Message,
    ReplyDone,
    TextDelta,
    Usage,
)
from mini_deepagents.errors import (
    ContextOverflowError,
    ModelError,
    SessionNotFoundError,
)
from tests.backend.fakes import FakeChatModel, InMemoryChatRepository

ESTIMATOR = ByteTokenEstimator()
MAX_OUTPUT_TOKENS = 64


def make_service(
    model: FakeChatModel | None = None,
    *,
    repository: InMemoryChatRepository | None = None,
    input_budget: int = 1000,
) -> tuple[ChatService, InMemoryChatRepository, FakeChatModel]:
    repository = repository or InMemoryChatRepository()
    model = model or FakeChatModel()
    service = ChatService(
        repository,
        model,
        ESTIMATOR,
        input_budget=input_budget,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return service, repository, model


async def collect(events: AsyncIterator[ChatEvent]) -> list[ChatEvent]:
    return [event async for event in events]


def test_prepare_raises_for_unknown_session() -> None:
    service, _, _ = make_service()

    with pytest.raises(SessionNotFoundError):
        asyncio.run(service.prepare(99, "hi"))


def test_prepare_rejects_message_over_budget_and_saves_nothing() -> None:
    service, repository, _ = make_service(input_budget=10)
    session = repository.create_session()
    content = "x" * 300

    with pytest.raises(ContextOverflowError) as info:
        asyncio.run(service.prepare(session.id, content))

    assert info.value.estimated_tokens == ESTIMATOR.estimate(Message("user", content))
    assert info.value.budget_tokens == 10
    assert repository.get_messages(session.id) == []


def test_prepare_caps_window_to_budget_and_stream_sends_that_window() -> None:
    budget = 150
    service, repository, model = make_service(input_budget=budget)
    session = repository.create_session()
    for index in range(10):
        repository.add_exchange(session.id, f"q{index} " + "a" * 78, f"r{index} " + "b" * 78)

    turn = asyncio.run(service.prepare(session.id, "new question"))

    estimates = [ESTIMATOR.estimate(message) for message in turn.window]
    assert sum(estimates) <= budget
    assert turn.estimated_tokens == sum(estimates)
    assert turn.budget_tokens == budget
    assert turn.history_messages == 20
    assert len(turn.window) < 21
    assert turn.window[0].role == "user"
    assert turn.window[-1] == Message("user", "new question")

    asyncio.run(collect(service.stream_reply(turn)))

    assert model.calls == [
        (list(turn.window), MAX_OUTPUT_TOKENS, f"mini-deepagents-{session.id}")
    ]


def test_full_stream_yields_deltas_then_done_and_saves_exchange() -> None:
    usage = Usage(10, 2, 12)
    service, repository, _ = make_service(FakeChatModel(("Hello", " world"), usage=usage))
    session = repository.create_session()

    async def run() -> list[ChatEvent]:
        turn = await service.prepare(session.id, "hi")
        return await collect(service.stream_reply(turn))

    events = asyncio.run(run())

    assert events == [
        TextDelta("Hello"),
        TextDelta(" world"),
        ReplyDone(reply_chars=len("Hello world"), usage=usage),
    ]
    assert repository.get_messages(session.id) == [
        Message("user", "hi"),
        Message("assistant", "Hello world"),
    ]


def test_model_error_propagates_and_saves_nothing() -> None:
    model = FakeChatModel(error=ModelError("x"), fail_after=1)
    service, repository, _ = make_service(model)
    session = repository.create_session()

    async def run() -> list[ChatEvent]:
        turn = await service.prepare(session.id, "hi")
        return await collect(service.stream_reply(turn))

    with pytest.raises(ModelError):
        asyncio.run(run())
    assert repository.get_messages(session.id) == []


def test_early_close_saves_nothing_and_closes_model_stream() -> None:
    service, repository, model = make_service()
    session = repository.create_session()

    async def run() -> ChatEvent:
        turn = await service.prepare(session.id, "hi")
        events = service.stream_reply(turn)
        first = await anext(events)
        await events.aclose()
        return first

    first = asyncio.run(run())

    assert first == TextDelta("Hello")
    assert model.closed_early
    assert repository.get_messages(session.id) == []


def test_session_queries_pass_through_to_repository() -> None:
    service, repository, _ = make_service()

    async def run() -> None:
        first = await service.create_session()
        second = await service.create_session()
        repository.add_exchange(first.id, "q", "a")

        assert await service.list_sessions() == [second, first]
        assert await service.get_messages(first.id) == [
            Message("user", "q"),
            Message("assistant", "a"),
        ]

    asyncio.run(run())
