"""Versioned people-count event schema (v1), validation, and publish-rate policy.

See docs/event-schema.md. Events carry counts and session-local track IDs only:
no images, no bounding boxes, no stream URLs or credentials.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from . import SOURCE_NAME
from .config import CAMERA_ID_RE
from .timeutil import format_utc, parse_utc

SCHEMA_VERSION = 1
EVENT_TYPE_COUNT = "people_count_update"
INPUT_KINDS = {"rtsp", "file"}
MAX_PEOPLE = 1000
MAX_FUTURE_SKEW = timedelta(minutes=2)

_EVENT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_FIELDS = {
    "schema_version",
    "event_id",
    "sequence",
    "camera_id",
    "event_type",
    "timestamp",
    "people_count",
    "tracked_person_ids",
    "input_kind",
    "source",
}


class EventValidationError(ValueError):
    pass


def build_count_event(
    *,
    camera_id: str,
    timestamp: datetime,
    people_count: int,
    tracked_person_ids: list[int],
    sequence: int,
    input_kind: str,
) -> dict[str, Any]:
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
        "sequence": sequence,
        "camera_id": camera_id,
        "event_type": EVENT_TYPE_COUNT,
        "timestamp": format_utc(timestamp),
        "people_count": people_count,
        "tracked_person_ids": sorted(tracked_person_ids),
        "input_kind": input_kind,
        "source": SOURCE_NAME,
    }
    validate_event(event)
    return event


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_event(event: Any, now: datetime | None = None) -> None:
    """Raise EventValidationError if the event does not match schema v1."""
    if not isinstance(event, dict):
        raise EventValidationError("event must be a JSON object")
    missing = _FIELDS - event.keys()
    if missing:
        raise EventValidationError(f"missing fields: {sorted(missing)}")
    extra = event.keys() - _FIELDS
    if extra:
        raise EventValidationError(f"unexpected fields: {sorted(extra)}")
    if event["schema_version"] != SCHEMA_VERSION or not _is_int(event["schema_version"]):
        raise EventValidationError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(event["event_id"], str) or not _EVENT_ID_RE.match(event["event_id"]):
        raise EventValidationError("event_id must be 32 lowercase hex chars")
    if not _is_int(event["sequence"]) or event["sequence"] < 0:
        raise EventValidationError("sequence must be a non-negative integer")
    if not isinstance(event["camera_id"], str) or not CAMERA_ID_RE.match(event["camera_id"]):
        raise EventValidationError("camera_id is invalid")
    if event["event_type"] != EVENT_TYPE_COUNT:
        raise EventValidationError(f"event_type must be {EVENT_TYPE_COUNT!r}")
    try:
        ts = parse_utc(event["timestamp"])
    except ValueError as exc:
        raise EventValidationError(str(exc)) from None
    if now is not None and ts - now > MAX_FUTURE_SKEW:
        raise EventValidationError("timestamp is too far in the future")
    count = event["people_count"]
    if not _is_int(count) or not 0 <= count <= MAX_PEOPLE:
        raise EventValidationError(f"people_count must be an integer in [0, {MAX_PEOPLE}]")
    ids = event["tracked_person_ids"]
    if not isinstance(ids, list) or not all(_is_int(i) and i >= 0 for i in ids):
        raise EventValidationError("tracked_person_ids must be a list of non-negative integers")
    if len(set(ids)) != len(ids):
        raise EventValidationError("tracked_person_ids must be unique")
    if len(ids) > count:
        raise EventValidationError("tracked_person_ids cannot exceed people_count")
    if event["input_kind"] not in INPUT_KINDS:
        raise EventValidationError(f"input_kind must be one of {sorted(INPUT_KINDS)}")
    if event["source"] != SOURCE_NAME:
        raise EventValidationError(f"source must be {SOURCE_NAME!r}")


@dataclass
class PublishPolicy:
    """Decides when a count sample becomes an event.

    * A changed count must be seen on ``stable_samples`` consecutive samples
      (debounces one-frame flicker) before it is considered real.
    * A real change is published, but never more often than ``min_interval``.
    * Otherwise a heartbeat event with the current sample is published every
      ``heartbeat_interval``, so the Worker always has a recent value.
    """

    min_interval: float
    heartbeat_interval: float
    stable_samples: int = 2

    def __post_init__(self) -> None:
        self._published_count: int | None = None
        self._last_publish: float | None = None
        self._candidate: int | None = None
        self._candidate_hits = 0

    def reset(self) -> None:
        """Forget the debounce candidate (e.g. after a reconnect). Keeps the last published value."""
        self._candidate = None
        self._candidate_hits = 0

    def should_publish(self, count: int, now: float) -> bool:
        if self._published_count is None:
            return self._mark(count, now)  # first sample after startup
        if count == self._published_count:
            self._candidate, self._candidate_hits = None, 0
        else:
            if count == self._candidate:
                self._candidate_hits += 1
            else:
                self._candidate, self._candidate_hits = count, 1
            if self._candidate_hits >= self.stable_samples and now - self._last_publish >= self.min_interval:
                return self._mark(count, now)
        if now - self._last_publish >= self.heartbeat_interval:
            return self._mark(count, now)  # heartbeat carries the current sample
        return False

    def _mark(self, count: int, now: float) -> bool:
        self._published_count = count
        self._last_publish = now
        self._candidate, self._candidate_hits = None, 0
        return True
