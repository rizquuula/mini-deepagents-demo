from __future__ import annotations

import json
import logging

import httpx
import pytest

from mini_deepagents.llm import LLMError, OpenCodeClient

API_KEY = "sk-test-secret-key-123"
MODEL = "deepseek-v4.1-flash"
SESSION_ID = "mini-deepagents-7"
LOGGER_NAME = "mini_deepagents.llm"
MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "private user question"},
]


def ok_body(content: object = "hello there", usage: object | None = None) -> dict[str, object]:
    body: dict[str, object] = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": "hidden reasoning",
                }
            }
        ]
    }
    if usage is not None:
        body["usage"] = usage
    return body


def make_client(
    response: httpx.Response,
    requests: list[httpx.Request],
    base_url: str = "https://example.test/v1",
) -> OpenCodeClient:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    return OpenCodeClient(base_url, API_KEY, MODEL, transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("base_url", ["https://example.test/v1", "https://example.test/v1/"])
def test_chat_posts_to_chat_completions_without_double_slash(base_url: str) -> None:
    requests: list[httpx.Request] = []
    client = make_client(httpx.Response(200, json=ok_body()), requests, base_url)

    client.chat(MESSAGES, max_tokens=64, session_id=SESSION_ID)

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "https://example.test/v1/chat/completions"


def test_chat_sends_auth_session_and_user_agent_headers_and_exact_body() -> None:
    requests: list[httpx.Request] = []
    client = make_client(httpx.Response(200, json=ok_body()), requests)

    client.chat(MESSAGES, max_tokens=64, session_id=SESSION_ID)

    request = requests[0]
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert request.headers["x-opencode-session"] == SESSION_ID
    assert request.headers["User-Agent"] == "mini-deepagents/0.1.0"
    assert json.loads(request.content) == {
        "model": MODEL,
        "messages": MESSAGES,
        "max_tokens": 64,
    }


def test_chat_returns_message_content_and_ignores_reasoning() -> None:
    client = make_client(httpx.Response(200, json=ok_body("the answer")), [])

    assert client.chat(MESSAGES, max_tokens=64, session_id=SESSION_ID) == "the answer"


@pytest.mark.parametrize("status", [401, 500])
def test_chat_raises_llm_error_on_error_status_without_leaking_key(status: int) -> None:
    client = make_client(httpx.Response(status, json={"error": "nope"}), [])

    with pytest.raises(LLMError) as excinfo:
        client.chat(MESSAGES, max_tokens=64, session_id=SESSION_ID)

    assert API_KEY not in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, httpx.HTTPError)


def test_chat_raises_llm_error_on_transport_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = OpenCodeClient(
        "https://example.test/v1", API_KEY, MODEL, transport=httpx.MockTransport(handler)
    )

    with pytest.raises(LLMError) as excinfo:
        client.chat(MESSAGES, max_tokens=64, session_id=SESSION_ID)

    assert API_KEY not in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, httpx.HTTPError)


@pytest.mark.parametrize(
    ("response", "match"),
    [
        pytest.param(httpx.Response(200, json=ok_body("")), "empty reply", id="empty-content"),
        pytest.param(httpx.Response(200, json=ok_body(None)), "empty reply", id="null-content"),
        pytest.param(httpx.Response(200, json={"usage": {}}), None, id="no-choices"),
        pytest.param(httpx.Response(200, json={"choices": []}), None, id="empty-choices"),
        pytest.param(httpx.Response(200, content=b"not json"), None, id="bad-json"),
    ],
)
def test_chat_raises_llm_error_on_unusable_body(response: httpx.Response, match: str | None) -> None:
    client = make_client(response, [])

    with pytest.raises(LLMError, match=match):
        client.chat(MESSAGES, max_tokens=64, session_id=SESSION_ID)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        pytest.param(
            {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            (11, 7, 18),
            id="usage",
        ),
        pytest.param(None, (None, None, None), id="no-usage"),
    ],
)
def test_chat_logs_token_usage_without_key_or_content(
    caplog: pytest.LogCaptureFixture, usage: object, expected: tuple[object, ...]
) -> None:
    client = make_client(httpx.Response(200, json=ok_body("reply text", usage)), [])

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        client.chat(MESSAGES, max_tokens=64, session_id=SESSION_ID)

    [record] = [r for r in caplog.records if r.name == LOGGER_NAME]
    assert record.levelno == logging.INFO
    assert record.getMessage() == "model call succeeded"
    assert record.model == MODEL
    assert (record.prompt_tokens, record.completion_tokens, record.total_tokens) == expected
    assert isinstance(record.duration_ms, int)
    assert record.duration_ms >= 0
    dumped = str(vars(record))
    assert API_KEY not in dumped
    assert "private user question" not in dumped
    assert "reply text" not in dumped
