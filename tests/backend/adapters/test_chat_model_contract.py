"""Contract tests for ``ChatModel``. Each contract test runs against the fake and the real adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field

import httpx
import pytest

from mini_deepagents import ChatModel, Message, ModelError, ModelEvent, StreamEnd, TextDelta, Usage
from mini_deepagents.testing import FakeChatModel

from mini_deepagents_demo.adapters.llm import create_chat_model
from mini_deepagents_demo.adapters.llm.opencode import OpenCodeChatModel
from mini_deepagents_demo.config import Settings
from mini_deepagents_demo.errors import ConfigError

API_KEY = "sk-test-secret-key-123"
BASE_URL = "https://example.test/v1"
MODEL = "deepseek-v4.1-flash"
SESSION_KEY = "mini-deepagents-7"
LOGGER_NAME = "mini_deepagents_demo.adapters.llm.opencode"
MESSAGES = [Message(role="user", content="private user question")]
REPLY = ("Hel", "lo")
REASONING = ("secret thought one", "secret thought two")
PROVIDER_USAGE = {
    "prompt_tokens": 11,
    "completion_tokens": 7,
    "total_tokens": 18,
    "completion_tokens_details": {"reasoning_tokens": 4},
}


def _chunk(delta: dict[str, object], finish_reason: str | None = None) -> dict[str, object]:
    return {"choices": [{"index": 0, "finish_reason": finish_reason, "delta": delta}]}


def provider_chunks(reply: Sequence[str]) -> list[dict[str, object]]:
    """Return the chunks of one provider stream: role, reasoning, reply, usage."""
    usage_chunk = _chunk({}, finish_reason="stop")
    usage_chunk["usage"] = PROVIDER_USAGE
    return [
        _chunk({"role": "assistant", "content": ""}),
        *(_chunk({"reasoning_content": text}) for text in REASONING),
        *(_chunk({"content": text}) for text in reply),
        usage_chunk,
    ]


def sse_body(chunks: Sequence[dict[str, object]]) -> bytes:
    events = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return "".join([*events, "data: [DONE]\n\n"]).encode()


class RecordingStream(httpx.AsyncByteStream):
    """A response body that yields one SSE event per read and records ``aclose``."""

    def __init__(self, body: bytes) -> None:
        self._events = [event + b"\n\n" for event in body.split(b"\n\n") if event]
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self._events:
            yield event

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class MockProvider:
    """A ``MockTransport`` handler that records each request and returns one SSE body."""

    body: bytes
    status_code: int = 200
    requests: list[httpx.Request] = field(default_factory=list)
    stream: RecordingStream | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.stream = RecordingStream(self.body)
        return httpx.Response(
            self.status_code,
            headers={"content-type": "text/event-stream"},
            stream=self.stream,
        )


def opencode_model(provider: MockProvider) -> OpenCodeChatModel:
    return OpenCodeChatModel(
        BASE_URL, API_KEY, MODEL, transport=httpx.MockTransport(provider)
    )


async def collect(model: ChatModel) -> list[ModelEvent]:
    return [
        event
        async for event in model.stream(MESSAGES, max_tokens=64, session_key=SESSION_KEY)
    ]


def run_stream(model: ChatModel) -> list[ModelEvent]:
    async def scenario() -> list[ModelEvent]:
        try:
            return await collect(model)
        finally:
            await model.aclose()

    return asyncio.run(scenario())


@dataclass
class Subject:
    """A model under contract test and a probe that tells if its upstream call stopped."""

    model: ChatModel
    upstream_closed: Callable[[], bool]


def _fake_subject(reply: Sequence[str]) -> Subject:
    fake = FakeChatModel(deltas=reply)
    return Subject(fake, lambda: fake.closed_early)


def _opencode_subject(reply: Sequence[str]) -> Subject:
    provider = MockProvider(sse_body(provider_chunks(reply)))
    return Subject(
        opencode_model(provider),
        lambda: provider.stream is not None and provider.stream.closed,
    )


@pytest.fixture(params=[_fake_subject, _opencode_subject], ids=["fake", "opencode"])
def make_subject(request: pytest.FixtureRequest) -> Callable[[Sequence[str]], Subject]:
    return request.param


# --- Contract: runs against every ChatModel ---------------------------------


def test_stream_yields_text_deltas_then_one_stream_end_last(
    make_subject: Callable[[Sequence[str]], Subject],
) -> None:
    events = run_stream(make_subject(REPLY).model)

    assert events[:-1] == [TextDelta("Hel"), TextDelta("lo")]
    assert isinstance(events[-1], StreamEnd)


def test_stream_with_no_reply_text_raises_model_error(
    make_subject: Callable[[Sequence[str]], Subject],
) -> None:
    with pytest.raises(ModelError, match="empty reply"):
        run_stream(make_subject(()).model)


def test_closing_the_stream_early_stops_the_upstream_call(
    make_subject: Callable[[Sequence[str]], Subject],
) -> None:
    subject = make_subject(REPLY)

    async def scenario() -> TextDelta:
        events = subject.model.stream(MESSAGES, max_tokens=64, session_key=SESSION_KEY)
        first = await anext(events)
        await events.aclose()  # type: ignore[attr-defined]
        await subject.model.aclose()
        return first

    assert asyncio.run(scenario()) == TextDelta("Hel")
    assert subject.upstream_closed()


# --- OpenCode only ----------------------------------------------------------


def test_opencode_sends_one_streaming_chat_completions_request() -> None:
    provider = MockProvider(sse_body(provider_chunks(REPLY)))

    run_stream(opencode_model(provider))

    [request] = provider.requests
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert request.headers["x-opencode-session"] == SESSION_KEY
    assert request.headers["user-agent"] == "mini-deepagents-demo/0.1.0"
    assert json.loads(request.content) == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "private user question"}],
        "max_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def test_opencode_drops_reasoning_and_reports_provider_usage() -> None:
    events = run_stream(opencode_model(MockProvider(sse_body(provider_chunks(REPLY)))))

    assert events == [TextDelta("Hel"), TextDelta("lo"), StreamEnd(Usage(11, 7, 18))]


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (500, b"internal error"),
        (401, b'{"error": "invalid key"}'),
        (200, b"data: {not json\n\n"),
    ],
    ids=["status-500", "status-401", "malformed-line"],
)
def test_opencode_failure_raises_model_error_without_the_key(
    status_code: int, body: bytes
) -> None:
    provider = MockProvider(body, status_code=status_code)

    with pytest.raises(ModelError) as caught:
        run_stream(opencode_model(provider))

    assert API_KEY not in str(caught.value)
    assert BASE_URL not in str(caught.value)


def test_opencode_success_log_carries_token_counts_but_no_content_or_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    run_stream(opencode_model(MockProvider(sse_body(provider_chunks(REPLY)))))

    [record] = [r for r in caplog.records if r.message == "model stream succeeded"]
    assert record.levelno == logging.INFO
    assert (
        record.model,  # type: ignore[attr-defined]
        record.prompt_tokens,  # type: ignore[attr-defined]
        record.completion_tokens,  # type: ignore[attr-defined]
        record.reasoning_tokens,  # type: ignore[attr-defined]
        record.total_tokens,  # type: ignore[attr-defined]
    ) == (MODEL, 11, 7, 4, 18)
    assert isinstance(record.first_delta_ms, int)  # type: ignore[attr-defined]
    assert isinstance(record.duration_ms, int)  # type: ignore[attr-defined]
    logged = " ".join(str(value) for value in vars(record).values())
    for secret in (API_KEY, "Hel", "private user question", REASONING[0]):
        assert secret not in logged


# --- Registry ---------------------------------------------------------------


def test_registry_creates_the_opencode_model() -> None:
    model = create_chat_model("opencode", Settings(api_key="k"))

    assert isinstance(model, OpenCodeChatModel)
    asyncio.run(model.aclose())


def test_registry_rejects_an_unknown_provider_and_lists_known_names() -> None:
    with pytest.raises(ConfigError, match="opencode"):
        create_chat_model("no-such-provider", Settings(api_key="k"))
