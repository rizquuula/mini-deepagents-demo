"""HTTP client for the OpenAI-compatible chat-completions API on OpenCode Go."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence

import httpx

from mini_deepagents_demo.chat import Message, ModelEvent, StreamEnd, TextDelta, Usage
from mini_deepagents_demo.errors import ModelError

logger = logging.getLogger(__name__)

_CHAT_PATH = "/chat/completions"
_USER_AGENT = "mini-deepagents/0.1.0"
_SESSION_HEADER = "x-opencode-session"
_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"


class OpenCodeChatModel:
    """A ``ChatModel`` that streams replies from OpenCode Go over server-sent events."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + _CHAT_PATH
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(
            timeout=timeout, transport=transport, headers={"User-Agent": _USER_AGENT}
        )

    async def stream(
        self, messages: Sequence[Message], *, max_tokens: int, session_key: str
    ) -> AsyncGenerator[ModelEvent, None]:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"Authorization": f"Bearer {self._api_key}", _SESSION_HEADER: session_key}
        started = time.perf_counter()
        first_delta_ms: int | None = None
        usage: Mapping[str, object] = {}
        try:
            async with self._client.stream(
                "POST", self._url, json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                async for chunk in _sse_chunks(response):
                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, Mapping):
                        usage = chunk_usage
                    for text in _content_deltas(chunk):
                        if first_delta_ms is None:
                            first_delta_ms = _elapsed_ms(started)
                        yield TextDelta(text)
        except httpx.HTTPStatusError as exc:
            raise ModelError(
                f"model request failed: status={exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelError(f"model request failed: {type(exc).__name__}") from exc

        if first_delta_ms is None:
            raise ModelError("empty reply")
        _log_stream_usage(self._model, usage, first_delta_ms, _elapsed_ms(started))
        yield StreamEnd(
            Usage(
                prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                completion_tokens=_int_or_none(usage.get("completion_tokens")),
                total_tokens=_int_or_none(usage.get("total_tokens")),
            )
        )

    async def aclose(self) -> None:
        await self._client.aclose()


async def _sse_chunks(response: httpx.Response) -> AsyncIterator[Mapping[str, object]]:
    """Yield each JSON chunk of an SSE body. Stop at ``[DONE]`` or at the end of the body."""
    async for line in response.aiter_lines():
        if not line.startswith(_SSE_DATA_PREFIX):
            continue
        data = line[len(_SSE_DATA_PREFIX) :].strip()
        if data == _SSE_DONE:
            return
        try:
            chunk = json.loads(data)
        except ValueError as exc:
            raise ModelError("malformed model stream: data line is not JSON") from exc
        if not isinstance(chunk, Mapping):
            raise ModelError("malformed model stream: chunk is not an object")
        yield chunk


def _content_deltas(chunk: Mapping[str, object]) -> list[str]:
    """Return the non-empty reply text of a chunk. Ignore ``reasoning_content``."""
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return []
    texts: list[str] = []
    for choice in choices:
        delta = choice.get("delta") if isinstance(choice, Mapping) else None
        content = delta.get("content") if isinstance(delta, Mapping) else None
        if isinstance(content, str) and content:
            texts.append(content)
    return texts


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _log_stream_usage(
    model: str, usage: Mapping[str, object], first_delta_ms: int, duration_ms: int
) -> None:
    details = usage.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
    logger.info(
        "model stream succeeded",
        extra={
            "model": model,
            "prompt_tokens": _int_or_none(usage.get("prompt_tokens")),
            "completion_tokens": _int_or_none(usage.get("completion_tokens")),
            "reasoning_tokens": _int_or_none(reasoning),
            "total_tokens": _int_or_none(usage.get("total_tokens")),
            "first_delta_ms": first_delta_ms,
            "duration_ms": duration_ms,
        },
    )
