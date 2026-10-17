"""Wall-clock UTC timestamps (ISO 8601, millisecond precision, 'Z' suffix)."""

from __future__ import annotations

import re
from datetime import datetime, timezone

_ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_utc(dt: datetime) -> str:
    """Format an aware datetime as e.g. 2026-10-17T10:00:00.123Z."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime; timestamps must be timezone-aware UTC")
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_utc(value: str) -> datetime:
    """Parse a strict UTC ISO 8601 'Z' timestamp. Raises ValueError otherwise."""
    if not isinstance(value, str) or not _ISO_Z.match(value):
        raise ValueError(f"timestamp must be UTC ISO 8601 ending in 'Z', got {value!r}")
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in value else "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(value[:-1], fmt).replace(tzinfo=timezone.utc)
