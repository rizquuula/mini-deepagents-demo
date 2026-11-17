"""Contract tests for ``ChatRepository``: the fake and every real adapter must pass them."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest

from mini_deepagents.adapters.storage import create_repository
from mini_deepagents.adapters.storage.sqlite import ChatStore, SqliteChatRepository
from mini_deepagents.chat import Message
from mini_deepagents.chat.ports import ChatRepository
from mini_deepagents.config import Settings
from mini_deepagents.errors import ConfigError, SessionNotFoundError
from tests.backend.fakes import InMemoryChatRepository

UNKNOWN_SESSION_ID = 9999

_FACTORIES: dict[str, Callable[[Path], ChatRepository]] = {
    "in_memory": lambda _path: InMemoryChatRepository(),
    "sqlite": lambda path: SqliteChatRepository(path / "chat.db"),
}


@pytest.fixture(params=sorted(_FACTORIES))
def repository(request: pytest.FixtureRequest, tmp_path: Path) -> ChatRepository:
    return _FACTORIES[request.param](tmp_path)


def _user_version(db_path: Path) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute("PRAGMA user_version").fetchone()[0]


def test_sessions_list_newest_first(repository: ChatRepository) -> None:
    assert repository.list_sessions() == []

    first = repository.create_session()
    second = repository.create_session()

    assert repository.list_sessions() == [second, first]


def test_exchange_is_stored_in_order_per_session(repository: ChatRepository) -> None:
    session = repository.create_session()
    other = repository.create_session()

    repository.add_exchange(session.id, "hi", "hello")
    repository.add_exchange(session.id, "bye", "see you")

    assert repository.get_messages(session.id) == [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
        Message(role="user", content="bye"),
        Message(role="assistant", content="see you"),
    ]
    assert repository.get_messages(other.id) == []


def test_unknown_session_raises_and_writes_nothing(repository: ChatRepository) -> None:
    with pytest.raises(SessionNotFoundError):
        repository.get_messages(UNKNOWN_SESSION_ID)
    with pytest.raises(SessionNotFoundError):
        repository.add_exchange(UNKNOWN_SESSION_ID, "hi", "hello")

    assert repository.list_sessions() == []
    session = repository.create_session()
    assert repository.get_messages(session.id) == []


def test_sqlite_data_survives_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    session = SqliteChatRepository(db_path).create_session()
    SqliteChatRepository(db_path).add_exchange(session.id, "hi", "hello")

    reopened = SqliteChatRepository(db_path)

    assert reopened.list_sessions() == [session]
    assert reopened.get_messages(session.id) == [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]


def test_sqlite_upgrades_a_chat_store_file_to_version_1(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    old_store = ChatStore(db_path)
    old_session = old_store.create_session()
    old_store.add_exchange(int(old_session["id"]), "hi", "hello")
    assert _user_version(db_path) == 0

    repository = SqliteChatRepository(db_path)

    assert _user_version(db_path) == 1
    assert [s.id for s in repository.list_sessions()] == [old_session["id"]]
    assert repository.get_messages(int(old_session["id"])) == [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]
    SqliteChatRepository(db_path)
    assert _user_version(db_path) == 1


def test_registry_creates_the_sqlite_repository(tmp_path: Path) -> None:
    settings = Settings(api_key="k", db_path=tmp_path / "x.db")

    assert isinstance(create_repository("sqlite", settings), SqliteChatRepository)


def test_registry_rejects_an_unknown_backend_and_names_the_known_ones(
    tmp_path: Path,
) -> None:
    settings = Settings(api_key="k", db_path=tmp_path / "x.db")

    with pytest.raises(ConfigError, match="sqlite"):
        create_repository("nope", settings)
