"""HTTP client for the OpenAI-compatible chat-completions API on OpenCode Go."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

_CHAT_PATH = "/chat/completions"
_USER_AGENT = "mini-deepagents/0.1.0"
_SESSION_HEADER = "x-opencode-session"


class LLMError(Exception):
    """The model call failed or returned no usable reply."""


class ChatModel(Protocol):
    def chat(
        self, messages: Sequence[Mapping[str, str]], max_tokens: int, session_id: str
    ) -> str: ...


class OpenCodeClient:
    """Send chat history to one model and return the reply text."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + _CHAT_PATH
        self._api_key = api_key
        self._model = model
        self._client = httpx.Client(
            timeout=timeout, transport=transport, headers={"User-Agent": _USER_AGENT}
        )

    def chat(
        self, messages: Sequence[Mapping[str, str]], max_tokens: int, session_id: str
    ) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": max_tokens,
        }
        body = self._post(payload, session_id)
        reply = _extract_reply(body)
        _log_usage(self._model, body.get("usage"))
        return reply

    def close(self) -> None:
        self._client.close()

    def _post(self, payload: Mapping[str, object], session_id: str) -> dict[str, object]:
        headers = {"Authorization": f"Bearer {self._api_key}", _SESSION_HEADER: session_id}
        try:
            response = self._client.post(self._url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"model request failed: status={exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"model request failed: {type(exc).__name__}") from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError("malformed model response: body is not JSON") from exc
        if not isinstance(body, dict):
            raise LLMError("malformed model response: body is not an object")
        return body


def _extract_reply(body: Mapping[str, object]) -> str:
    try:
        content = body["choices"][0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("malformed model response: no choices[0].message.content") from exc
    if content is None or content == "":
        raise LLMError("empty reply")
    if not isinstance(content, str):
        raise LLMError("malformed model response: content is not a string")
    return content


def _log_usage(model: str, usage: object) -> None:
    counts = usage if isinstance(usage, Mapping) else {}
    logger.info(
        "model call succeeded model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        model,
        counts.get("prompt_tokens"),
        counts.get("completion_tokens"),
        counts.get("total_tokens"),
    )
