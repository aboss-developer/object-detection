"""Structured JSON-lines logging with size-based rotation and secret redaction."""

from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path

from .redact import RedactingFilter
from .timeutil import format_utc, utc_now

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": format_utc(utc_now()),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_text:
            payload["exc"] = record.exc_text
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(log_dir: str, level: str = "INFO", max_bytes: int = 5_000_000, backup_count: int = 5) -> None:
    """Configure root logging: JSON to rotating file, JSON to stderr. Idempotent."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    redactor = RedactingFilter()
    formatter = JsonFormatter()

    file_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "alpha_vision.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    console = logging.StreamHandler()
    for handler in (file_handler, console):
        handler.addFilter(redactor)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Ultralytics and urllib3 are chatty at INFO.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def make_event_log(log_dir: str, max_bytes: int, backup_count: int) -> logging.Logger:
    """A separate rotating JSONL log holding one published event per line (replaces results_list dumps)."""
    logger = logging.getLogger("alpha_vision.events")
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "events.jsonl", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
