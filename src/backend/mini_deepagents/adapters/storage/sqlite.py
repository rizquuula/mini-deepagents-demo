"""SQLite store for chat sessions and their messages."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from pathlib import Path

from mini_deepagents import errors
from mini_deepagents.adapters.storage.migrations import migrate
from mini_deepagents.chat import Message, Session

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session_id_id ON messages (session_id, id);
"""


class SessionNotFoundError(Exception):
    """Raised when a session id does not exist in the store."""


class ChatStore:
    """Persist chat sessions and messages in one SQLite file.

    Each method opens its own connection, so one instance is safe to share
    across threads.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def create_session(self) -> dict[str, object]:
        with self._connect() as conn:
            cursor = conn.execute("INSERT INTO sessions DEFAULT VALUES")
            row = conn.execute(
                "SELECT id, created_at FROM sessions WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return {"id": row[0], "created_at": row[1]}

    def list_sessions(self) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at FROM sessions ORDER BY id DESC"
            ).fetchall()
        return [{"id": row_id, "created_at": created_at} for row_id, created_at in rows]

    def get_messages(self, session_id: int) -> list[dict[str, str]]:
        with self._connect() as conn:
            self._require_session(conn, session_id)
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def add_exchange(self, session_id: int, user_content: str, assistant_content: str) -> None:
        with self._connect() as conn:
            self._require_session(conn, session_id)
            conn.executemany(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                [
                    (session_id, "user", user_content),
                    (session_id, "assistant", assistant_content),
                ],
            )

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        """Yield a new connection inside one transaction; commit or roll back on exit."""
        return _connect(self._path)

    @staticmethod
    def _require_session(conn: sqlite3.Connection, session_id: int) -> None:
        if not _session_exists(conn, session_id):
            raise SessionNotFoundError(f"session {session_id} does not exist")


class SqliteChatRepository:
    """A ``ChatRepository`` on one SQLite file.

    The constructor migrates the schema once. Each method opens its own
    connection, so one instance is safe to share across threads.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        with _connect(self._path) as conn:
            migrate(conn)

    def create_session(self) -> Session:
        with _connect(self._path) as conn:
            cursor = conn.execute("INSERT INTO sessions DEFAULT VALUES")
            row = conn.execute(
                "SELECT id, created_at FROM sessions WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return Session(id=row[0], created_at=row[1])

    def list_sessions(self) -> list[Session]:
        with _connect(self._path) as conn:
            rows = conn.execute(
                "SELECT id, created_at FROM sessions ORDER BY id DESC"
            ).fetchall()
        return [Session(id=row_id, created_at=created_at) for row_id, created_at in rows]

    def get_messages(self, session_id: int) -> list[Message]:
        with _connect(self._path) as conn:
            self._require_session(conn, session_id)
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [Message(role=role, content=content) for role, content in rows]

    def add_exchange(
        self, session_id: int, user_content: str, assistant_content: str
    ) -> None:
        with _connect(self._path) as conn:
            self._require_session(conn, session_id)
            conn.executemany(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                [
                    (session_id, "user", user_content),
                    (session_id, "assistant", assistant_content),
                ],
            )

    @staticmethod
    def _require_session(conn: sqlite3.Connection, session_id: int) -> None:
        if not _session_exists(conn, session_id):
            raise errors.SessionNotFoundError(f"session {session_id} not found")


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Yield a new connection inside one transaction; commit or roll back on exit."""
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn


def _session_exists(conn: sqlite3.Connection, session_id: int) -> bool:
    row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return row is not None
