from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mini_deepagents.observability.logging import (
    KeyValueFormatter,
    RequestIdFilter,
    request_id_var,
    setup_logging,
)

FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=UTC).timestamp()


def make_record(
    extra: dict[str, object] | None = None, exc_info: object = None
) -> logging.LogRecord:
    record = logging.getLogger("test.logger").makeRecord(
        "test.logger", logging.INFO, __file__, 1, "session created", None, exc_info, extra=extra
    )
    record.created = FIXED_TIME
    token = request_id_var.set("abc123")
    try:
        RequestIdFilter().filter(record)
    finally:
        request_id_var.reset(token)
    return record


def test_formatter_renders_one_line_with_ordered_and_quoted_fields() -> None:
    record = make_record(
        {
            "session_id": 7,
            "note": "two words",
            "empty": "",
            "error": None,
            "color_message": "uvicorn copy with \x1b[36mcolor codes\x1b[0m",
        }
    )

    line = KeyValueFormatter().format(record)

    assert line == (
        '2026-01-02T03:04:05.678Z INFO test.logger session created request_id=abc123 '
        'session_id=7 note="two words" empty="" error=null'
    )


def test_formatter_appends_the_traceback() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = make_record(exc_info=sys.exc_info())

    lines = KeyValueFormatter().format(record).splitlines()

    assert lines[0].endswith("session created request_id=abc123")
    assert lines[1] == "Traceback (most recent call last):"
    assert lines[-1] == "RuntimeError: boom"


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.usefixtures("restore_root_logger")
def test_setup_logging_writes_one_line_to_the_file_after_two_calls(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "app.log"

    setup_logging("INFO", log_file)
    setup_logging("INFO", log_file)
    logging.getLogger("test.setup").info("app started", extra={"model": "m"})
    for handler in logging.getLogger().handlers:
        handler.flush()

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].endswith("INFO test.setup app started request_id=- model=m")
