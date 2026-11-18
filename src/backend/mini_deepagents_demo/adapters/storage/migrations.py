"""Versioned schema migrations for the SQLite chat store.

``PRAGMA user_version`` holds the version of a database file. ``migrate``
applies each migration with a higher version, in order, one transaction each.

To add migration 2:

1. Append ``(2, "<sql>")`` to ``MIGRATIONS``.
2. Never edit or delete an old pair. A shipped file already ran it.
3. Fix a bad migration with a new migration.
"""

from __future__ import annotations

import sqlite3
from typing import Final

_V1_SCHEMA: Final = """
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

MIGRATIONS: Final[tuple[tuple[int, str], ...]] = (
    # Version 1 is the schema of an unversioned file (user_version 0) from
    # before migrations existed. ``IF NOT EXISTS`` lets it adopt such a file.
    (1, _V1_SCHEMA),
)


def migrate(conn: sqlite3.Connection) -> int:
    """Apply each pending migration to ``conn`` and return the final version.

    Each migration and its ``user_version`` update run in one transaction.
    A failed migration rolls back and leaves the version unchanged.
    """
    version: int = conn.execute("PRAGMA user_version").fetchone()[0]
    for target, sql in MIGRATIONS:
        if target <= version:
            continue
        _apply(conn, target, sql)
        version = target
    return version


def _apply(conn: sqlite3.Connection, version: int, sql: str) -> None:
    # PRAGMA takes no bound parameters. ``version`` is an int from MIGRATIONS.
    script = f"BEGIN;\n{sql}\nPRAGMA user_version = {int(version)};\nCOMMIT;"
    try:
        conn.executescript(script)
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
