"""Key=value log lines with a request ID, written to stderr and a rotating file.

One line looks like this:

    2026-01-02T03:04:05.678Z INFO mini_deepagents.main session created request_id=3f2a9c1b7d0e session_id=7
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_MAX_LOG_BYTES: Final = 5_000_000
_LOG_BACKUP_COUNT: Final = 3
_OWNED_MARK: Final = "_mini_deepagents_handler"
_STANDARD_ATTRS: Final = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None))
) | {"message", "asctime", "taskName", "request_id", "color_message"}  # uvicorn's colored copy
_QUOTE_TRIGGERS: Final = (" ", '"', "=")


class RequestIdFilter(logging.Filter):
    """Copy the current request ID onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class KeyValueFormatter(logging.Formatter):
    """Render ``<time> <LEVEL> <logger> <message> request_id=<id> <extra fields>``."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, UTC).isoformat(
            timespec="milliseconds"
        )
        parts = [
            timestamp.replace("+00:00", "Z"),
            record.levelname,
            record.name,
            record.getMessage(),
            f"request_id={_render(getattr(record, 'request_id', request_id_var.get()))}",
        ]
        parts.extend(
            f"{key}={_render(value)}"
            for key, value in vars(record).items()
            if key not in _STANDARD_ATTRS
        )
        line = " ".join(parts)
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def _render(value: object) -> str:
    if value is None:
        return "null"
    text = str(value)
    if text == "" or any(char in text for char in _QUOTE_TRIGGERS):
        return json.dumps(text)
    return text


def setup_logging(level: str, log_file: Path) -> None:
    """Send every log record to stderr and ``log_file`` as key=value lines.

    A second call replaces the handlers of the first call; it never duplicates them.
    """
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _OWNED_MARK, False)]:
        root.removeHandler(handler)
        handler.close()

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        RotatingFileHandler(
            log_file,
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        ),
    ]
    for handler in handlers:
        handler.setFormatter(KeyValueFormatter())
        handler.addFilter(RequestIdFilter())
        setattr(handler, _OWNED_MARK, True)
        root.addHandler(handler)
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error"):
        server_logger = logging.getLogger(name)
        server_logger.handlers.clear()
        server_logger.propagate = True
    for name in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
