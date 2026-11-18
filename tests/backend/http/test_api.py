from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mini_deepagents import ModelError, Session
from mini_deepagents.testing import FakeChatModel, InMemoryChatRepository

from mini_deepagents_demo.config import Settings
from mini_deepagents_demo.main import create_app
from mini_deepagents_demo.observability.logging import RequestIdFilter

UNKNOWN_SESSION_ID = 9999
TOO_LONG = "x" * 30_000


class BrokenRepository(InMemoryChatRepository):
    def list_sessions(self) -> list[Session]:
        raise RuntimeError("disk on fire")


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO)
    caplog.handler.addFilter(RequestIdFilter())
    return caplog


@pytest.fixture
def fake() -> FakeChatModel:
    return FakeChatModel()


def make_client(
    tmp_path: Path, model: FakeChatModel, repository: InMemoryChatRepository | None = None
) -> TestClient:
    app = create_app(
        Settings(api_key="k", db_path=tmp_path / "x.db"),
        model=model,
        repository=repository or InMemoryChatRepository(),
    )
    return TestClient(app)


@pytest.fixture
def client(tmp_path: Path, fake: FakeChatModel) -> Iterator[TestClient]:
    with make_client(tmp_path, fake) as test_client:
        yield test_client


def find_record(caplog: pytest.LogCaptureFixture, message: str) -> logging.LogRecord:
    [record] = [r for r in caplog.records if r.getMessage() == message]
    return record


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for frame in body.strip().split("\n\n"):
        event_line, data_line = frame.split("\n")
        events.append(
            (event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: ")))
        )
    return events


def test_session_routes_create_list_and_read_messages(client: TestClient) -> None:
    created = client.post("/api/v1/sessions")
    listed = client.get("/api/v1/sessions")
    messages = client.get(f"/api/v1/sessions/{created.json()['id']}/messages")

    assert created.status_code == 201
    assert set(created.json()) == {"id", "created_at"}
    assert listed.json() == [created.json()]
    assert messages.status_code == 200
    assert messages.json() == []


def test_send_streams_the_reply_and_saves_the_exchange(
    client: TestClient, logs: pytest.LogCaptureFixture
) -> None:
    session_id = client.post("/api/v1/sessions").json()["id"]

    response = client.post(f"/api/v1/sessions/{session_id}/messages", json={"content": "hi"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [name for name, _ in parse_sse(response.text)] == ["delta", "delta", "done"]
    assert client.get(f"/api/v1/sessions/{session_id}/messages").json() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello world"},
    ]
    request_id = response.headers["X-Request-ID"]
    assert find_record(logs, "context fitted").request_id == request_id


@pytest.mark.parametrize(
    ("method", "session_id", "body", "status", "detail", "log_message", "log_level"),
    [
        ("get", UNKNOWN_SESSION_ID, None, 404, "session not found", "session not found", logging.INFO),
        ("post", UNKNOWN_SESSION_ID, {"content": "hi"}, 404, "session not found", "session not found", logging.INFO),
        (
            "post",
            1,
            {"content": TOO_LONG},
            413,
            "message is too long for the context window",
            "message rejected",
            logging.WARNING,
        ),
        ("post", 1, {"content": ""}, 422, None, None, None),
    ],
    ids=["get-unknown-session", "post-unknown-session", "too-long", "empty-content"],
)
def test_rejected_request_returns_json_error_and_skips_the_model(
    client: TestClient,
    fake: FakeChatModel,
    logs: pytest.LogCaptureFixture,
    method: str,
    session_id: int,
    body: dict[str, str] | None,
    status: int,
    detail: str | None,
    log_message: str | None,
    log_level: int | None,
) -> None:
    client.post("/api/v1/sessions")

    response = client.request(method, f"/api/v1/sessions/{session_id}/messages", json=body)

    assert response.status_code == status
    assert response.headers["content-type"] == "application/json"
    assert "X-Request-ID" in response.headers
    if detail is not None:
        assert response.json() == {"detail": detail}
    if log_message is not None:
        record = find_record(logs, log_message)
        assert record.levelno == log_level
        assert record.session_id == session_id
    assert fake.calls == []


def test_model_failure_sends_an_error_event_and_saves_nothing(tmp_path: Path) -> None:
    model = FakeChatModel(error=ModelError("status=502"), fail_after=1)
    with make_client(tmp_path, model) as client:
        session_id = client.post("/api/v1/sessions").json()["id"]

        response = client.post(
            f"/api/v1/sessions/{session_id}/messages", json={"content": "hi"}
        )
        saved = client.get(f"/api/v1/sessions/{session_id}/messages").json()

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["delta", "error"]
    assert events[-1][1]["code"] == "model_failed"
    assert events[-1][1]["request_id"] == response.headers["X-Request-ID"]
    assert saved == []


def test_unhandled_error_returns_500_with_request_id_and_logs_it(
    tmp_path: Path, fake: FakeChatModel, logs: pytest.LogCaptureFixture
) -> None:
    with make_client(tmp_path, fake, BrokenRepository()) as client:
        response = client.get("/api/v1/sessions")

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert "X-Request-ID" in response.headers
    record = find_record(logs, "unhandled error")
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None


def test_frontend_is_served(client: TestClient) -> None:
    index = client.get("/")
    script = client.get("/static/js/main.js")

    assert index.status_code == 200
    assert "<title>mini-deepagents</title>" in index.text
    assert script.status_code == 200


def test_app_does_not_close_a_model_that_the_caller_passed_in(
    tmp_path: Path, fake: FakeChatModel
) -> None:
    with make_client(tmp_path, fake):
        pass

    assert fake.aclose_calls == 0
