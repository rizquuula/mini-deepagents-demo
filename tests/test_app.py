from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mini_deepagents.app import create_app
from mini_deepagents.config import Settings
from mini_deepagents.context import estimate_tokens
from mini_deepagents.llm import LLMError
from mini_deepagents.store import ChatStore

REPLY = "fixed reply"
UNKNOWN_SESSION_ID = 9999


class FakeModel:
    """Record each chat call; return a fixed reply or raise ``LLMError``."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[list[Mapping[str, str]], int]] = []
        self.close_calls = 0

    def chat(self, messages: Sequence[Mapping[str, str]], max_tokens: int) -> str:
        self.calls.append((list(messages), max_tokens))
        if self.fail:
            raise LLMError("model request failed: status=500")
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
    client: TestClient, fake: FakeModel
) -> None:
    session_id = create_session(client)

    response = client.post(f"/api/sessions/{session_id}/messages", json={"content": "hi"})

    assert response.status_code == 200
    assert response.json() == {"reply": REPLY}
    assert fake.calls == [([{"role": "user", "content": "hi"}], 1024)]
    messages = client.get(f"/api/sessions/{session_id}/messages")
    assert messages.status_code == 200
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
    small_settings: Settings, fake: FakeModel
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


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_unknown_session_returns_404(
    client: TestClient, fake: FakeModel, method: str
) -> None:
    url = f"/api/sessions/{UNKNOWN_SESSION_ID}/messages"

    response = client.request(method, url, json={"content": "hi"})

    assert response.status_code == 404
    assert "detail" in response.json()
    assert fake.calls == []


def test_model_failure_returns_502_and_saves_nothing(settings: Settings) -> None:
    failing = FakeModel(fail=True)
    with TestClient(create_app(settings, model=failing)) as client:
        session_id = create_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/messages", json={"content": "hi"}
        )
        messages = client.get(f"/api/sessions/{session_id}/messages").json()

    assert response.status_code == 502
    assert response.json() == {"detail": "model request failed"}
    assert messages == []


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
