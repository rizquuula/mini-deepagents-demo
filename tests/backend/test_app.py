from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mini_deepagents.main import create_app
from mini_deepagents.config import Settings
from mini_deepagents.chat.context import estimate_tokens
from mini_deepagents.adapters.llm.opencode import LLMError
from mini_deepagents.observability.logging import RequestIdFilter
from mini_deepagents.adapters.storage.sqlite import ChatStore

REPLY = "fixed reply"
UNKNOWN_SESSION_ID = 9999


class FakeModel:
    """Record each chat call; return a fixed reply or raise ``error``."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[list[Mapping[str, str]], int]] = []
        self.session_ids: list[str] = []
        self.close_calls = 0

    def chat(
        self, messages: Sequence[Mapping[str, str]], max_tokens: int, session_id: str
    ) -> str:
        self.calls.append((list(messages), max_tokens))
        self.session_ids.append(session_id)
        if self.error is not None:
            raise self.error
        return REPLY

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(api_key="sk-test", db_path=tmp_path / "chat.db")


@pytest.fixture
def small_settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="sk-test",
        context_window=300,
        max_output_tokens=100,
        db_path=tmp_path / "chat.db",
    )


@pytest.fixture
def fake() -> FakeModel:
    return FakeModel()


@pytest.fixture
def client(settings: Settings, fake: FakeModel) -> Iterator[TestClient]:
    with TestClient(create_app(settings, model=fake)) as test_client:
        yield test_client


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO)
    caplog.handler.addFilter(RequestIdFilter())
    return caplog


def find_record(caplog: pytest.LogCaptureFixture, message: str) -> logging.LogRecord:
    [record] = [r for r in caplog.records if r.getMessage() == message]
    return record


def create_session(client: TestClient) -> int:
    response = client.post("/api/sessions")
    assert response.status_code == 201
    return response.json()["id"]


def test_created_session_is_listed(client: TestClient) -> None:
    response = client.post("/api/sessions")

    assert response.status_code == 201
    session = response.json()
    assert set(session) == {"id", "created_at"}
    listed = client.get("/api/sessions")
    assert listed.status_code == 200
    assert listed.json() == [session]


def test_send_returns_reply_and_saves_the_exchange(
    client: TestClient, fake: FakeModel, logs: pytest.LogCaptureFixture
) -> None:
    session_id = create_session(client)

    response = client.post(f"/api/sessions/{session_id}/messages", json={"content": "hi"})

    assert response.status_code == 200
    fitted = find_record(logs, "context fitted")
    assert fitted.request_id == response.headers["X-Request-ID"]
    assert fitted.session_id == session_id
    assert (fitted.sent_messages, fitted.dropped_messages) == (1, 0)
    assert response.json() == {"reply": REPLY}
    assert fake.calls == [([{"role": "user", "content": "hi"}], 1024)]
    assert fake.session_ids == [f"mini-deepagents-{session_id}"]
    messages = client.get(f"/api/sessions/{session_id}/messages")
    assert messages.status_code == 200
    assert messages.headers["X-Request-ID"] != response.headers["X-Request-ID"]
    assert messages.json() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": REPLY},
    ]


def test_model_request_stays_inside_the_context_window(
    small_settings: Settings, fake: FakeModel
) -> None:
    store = ChatStore(small_settings.db_path)
    session_id = store.create_session()["id"]
    for turn in range(10):
        store.add_exchange(
            session_id, f"question {turn} " + "q" * 80, f"answer {turn} " + "a" * 80
        )
    history_length = len(store.get_messages(session_id))

    with TestClient(create_app(small_settings, model=fake)) as client:
        response = client.post(
            f"/api/sessions/{session_id}/messages", json={"content": "hello"}
        )

    assert response.status_code == 200
    [(sent, max_tokens)] = fake.calls
    assert max_tokens == 100
    assert sum(estimate_tokens(m) for m in sent) <= 200
    assert len(sent) < history_length + 1
    assert sent[-1] == {"role": "user", "content": "hello"}
    assert sent[0]["role"] == "user"
    assert len(store.get_messages(session_id)) == history_length + 2


def test_oversize_message_returns_413_and_saves_nothing(
    small_settings: Settings, fake: FakeModel, logs: pytest.LogCaptureFixture
) -> None:
    with TestClient(create_app(small_settings, model=fake)) as client:
        session_id = create_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/messages", json={"content": "x" * 600}
        )
        messages = client.get(f"/api/sessions/{session_id}/messages").json()

    assert response.status_code == 413
    assert response.json() == {"detail": "message is too long for the context window"}
    assert fake.calls == []
    assert messages == []
    rejected = find_record(logs, "message rejected")
    assert rejected.levelno == logging.WARNING
    assert rejected.reason == "too_long"


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_unknown_session_returns_404(
    client: TestClient, fake: FakeModel, method: str
) -> None:
    url = f"/api/sessions/{UNKNOWN_SESSION_ID}/messages"

    response = client.request(method, url, json={"content": "hi"})

    assert response.status_code == 404
    assert "detail" in response.json()
    assert fake.calls == []


def test_model_failure_returns_502_and_logs_without_content(
    settings: Settings, logs: pytest.LogCaptureFixture
) -> None:
    failing = FakeModel(error=LLMError("model request failed: status=500"))
    with TestClient(create_app(settings, model=failing)) as client:
        session_id = create_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/messages", json={"content": "private question"}
        )
        messages = client.get(f"/api/sessions/{session_id}/messages").json()

    assert response.status_code == 502
    assert response.json() == {"detail": "model request failed"}
    assert messages == []
    failed = find_record(logs, "model call failed")
    assert failed.levelno == logging.ERROR
    assert failed.session_id == session_id
    assert all("private question" not in str(vars(r)) for r in logs.records)


def test_unexpected_model_error_returns_500_with_request_id(
    settings: Settings, logs: pytest.LogCaptureFixture
) -> None:
    broken = FakeModel(error=RuntimeError("bug"))
    with TestClient(create_app(settings, model=broken)) as client:
        session_id = create_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/messages", json={"content": "hi"}
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    unhandled = find_record(logs, "unhandled error")
    assert unhandled.levelno == logging.ERROR
    assert unhandled.exc_info is not None
    assert unhandled.request_id == response.headers["X-Request-ID"]


def test_empty_content_returns_422(client: TestClient, fake: FakeModel) -> None:
    session_id = create_session(client)

    response = client.post(f"/api/sessions/{session_id}/messages", json={"content": ""})

    assert response.status_code == 422
    assert fake.calls == []


def test_index_serves_the_chat_page(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>mini-deepagents</title>" in response.text


def test_caller_model_is_not_closed_on_shutdown(settings: Settings, fake: FakeModel) -> None:
    with TestClient(create_app(settings, model=fake)):
        pass

    assert fake.close_calls == 0
