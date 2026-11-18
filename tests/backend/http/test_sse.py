from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Sequence

import pytest

from mini_deepagents_demo.adapters.http.sse import chat_event_stream, encode_event
from mini_deepagents_demo.chat import ChatEvent, ReplyDone, TextDelta, Usage
from mini_deepagents_demo.errors import ModelError

SECRET_TEXT = "secret-user-text"


def find_record(caplog: pytest.LogCaptureFixture, message: str) -> logging.LogRecord:
    [record] = [r for r in caplog.records if r.getMessage() == message]
    return record


def parse_frames(chunks: Sequence[bytes]) -> list[tuple[str, dict[str, object]]]:
    frames = []
    for chunk in chunks:
        event_line, data_line = chunk.decode("utf-8").rstrip("\n").split("\n")
        frames.append(
            (event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: ")))
        )
    return frames


async def source(
    events: Sequence[ChatEvent], *, error: Exception | None = None
) -> AsyncGenerator[ChatEvent, None]:
    for event in events:
        yield event
    if error is not None:
        raise error


async def collect(stream: AsyncGenerator[bytes, None]) -> list[bytes]:
    return [chunk async for chunk in stream]


def test_encode_event_writes_compact_json_with_the_version_first() -> None:
    frame = encode_event("delta", {"text": "héllo 👋"})

    assert frame == 'event: delta\ndata: {"v":1,"text":"héllo 👋"}\n\n'.encode()


def test_stream_maps_deltas_and_done_and_logs_the_usage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    events = [TextDelta("Hel"), TextDelta("lo"), ReplyDone(5, Usage(10, 2, 12))]

    chunks = asyncio.run(
        collect(chat_event_stream(source(events), session_id=7, request_id="rid"))
    )

    assert parse_frames(chunks) == [
        ("delta", {"v": 1, "text": "Hel"}),
        ("delta", {"v": 1, "text": "lo"}),
        ("done", {"v": 1, "reply_chars": 5}),
    ]
    record = find_record(caplog, "stream completed")
    assert record.session_id == 7
    assert record.reply_chars == 5
    assert (record.prompt_tokens, record.completion_tokens, record.total_tokens) == (10, 2, 12)
    assert isinstance(record.first_delta_ms, int)
    assert isinstance(record.duration_ms, int)


def test_model_error_after_a_delta_sends_an_error_event_without_content_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    events = source([TextDelta(SECRET_TEXT)], error=ModelError("status=502"))

    chunks = asyncio.run(collect(chat_event_stream(events, session_id=3, request_id="rid")))

    assert parse_frames(chunks) == [
        ("delta", {"v": 1, "text": SECRET_TEXT}),
        (
            "error",
            {
                "v": 1,
                "code": "model_failed",
                "message": "model request failed",
                "request_id": "rid",
            },
        ),
    ]
    record = find_record(caplog, "stream failed")
    assert record.levelno == logging.ERROR
    assert record.session_id == 3
    assert record.error == "status=502"
    assert SECRET_TEXT not in caplog.text


def test_closing_the_byte_stream_early_logs_cancel_and_closes_the_source(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    closed: list[bool] = []

    async def endless() -> AsyncGenerator[ChatEvent, None]:
        try:
            while True:
                yield TextDelta("x")
        finally:
            closed.append(True)

    async def read_one_then_close() -> None:
        stream = chat_event_stream(endless(), session_id=5, request_id="rid")
        await anext(stream)
        await stream.aclose()

    asyncio.run(read_one_then_close())

    assert closed == [True]
    record = find_record(caplog, "stream cancelled")
    assert record.session_id == 5
    assert isinstance(record.duration_ms, int)
