from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from mini_deepagents.store import ChatStore, SessionNotFoundError

UNKNOWN_SESSION_ID = 9999


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "chat.db"


@pytest.fixture
def store(db_path: Path) -> ChatStore:
    return ChatStore(db_path)


def count_message_rows(db_path: Path) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
    return count


def test_new_store_has_no_sessions(store: ChatStore) -> None:
    assert store.list_sessions() == []


def test_list_sessions_returns_created_sessions_newest_first(store: ChatStore) -> None:
    first = store.create_session()
    second = store.create_session()

    assert isinstance(first["id"], int)
    assert isinstance(first["created_at"], str)
    assert [s["id"] for s in store.list_sessions()] == [second["id"], first["id"]]


def test_get_messages_returns_exchanges_in_order(store: ChatStore) -> None:
    session_id = store.create_session()["id"]
    assert store.get_messages(session_id) == []

    store.add_exchange(session_id, "hi", "hello")
    store.add_exchange(session_id, "how are you?", "fine")

    assert store.get_messages(session_id) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you?"},
        {"role": "assistant", "content": "fine"},
    ]


def test_messages_of_one_session_do_not_appear_in_another(store: ChatStore) -> None:
    first_id = store.create_session()["id"]
    second_id = store.create_session()["id"]

    store.add_exchange(first_id, "first question", "first answer")

    assert store.get_messages(second_id) == []


def test_get_messages_on_unknown_session_raises(store: ChatStore) -> None:
    with pytest.raises(SessionNotFoundError):
        store.get_messages(UNKNOWN_SESSION_ID)


def test_add_exchange_on_unknown_session_raises_and_writes_nothing(
    store: ChatStore, db_path: Path
) -> None:
    with pytest.raises(SessionNotFoundError):
        store.add_exchange(UNKNOWN_SESSION_ID, "hi", "hello")

    assert count_message_rows(db_path) == 0


def test_data_survives_reopen(db_path: Path) -> None:
    store = ChatStore(db_path)
    session = store.create_session()
    store.add_exchange(session["id"], "hi", "hello")

    reopened = ChatStore(db_path)

    assert reopened.list_sessions() == [session]
    assert reopened.get_messages(session["id"]) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
