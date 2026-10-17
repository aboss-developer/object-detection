"""Background event publisher: bounded queue, HTTP timeouts, bounded retries with backoff.

* The queue is a bounded deque; when full the *oldest* event is dropped (a newer
  count supersedes it) and a counter is incremented.
* Retries reuse the same ``event_id`` so the Worker can de-duplicate and never
  sends a second Telegram alert for a retried request.
* 4xx responses other than 408/429 are not retried (the payload or token is wrong;
  retrying would just hammer the Worker).
* 429 honours ``Retry-After`` (capped by ``backoff_max``).
"""

from __future__ import annotations

import collections
import json
import logging
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from .redact import redact_url
from .timeutil import format_utc, utc_now

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


@dataclass
class PublisherStats:
    sent: int = 0
    failed: int = 0
    dropped: int = 0
    retries: int = 0
    last_success_at: str | None = None
    last_error: str | None = None
    queue_len: int = 0


class Publisher:
    """Interface: ``submit(event)``, ``stats()``, ``close(timeout)``."""

    def submit(self, event: dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stats(self) -> PublisherStats:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self, timeout: float = 5.0) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class LogOnlyPublisher(Publisher):
    """Used when WORKER_URL is unset: events are only written to the local event log."""

    def __init__(self) -> None:
        self._stats = PublisherStats()

    def submit(self, event: dict[str, Any]) -> None:
        self._stats.sent += 1
        log.info("event (not published: WORKER_URL unset)", extra={"event_id": event["event_id"],
                                                                   "people_count": event["people_count"]})

    def stats(self) -> PublisherStats:
        return self._stats

    def close(self, timeout: float = 5.0) -> None:
        return None


@dataclass
class HttpPublisher(Publisher):
    url: str
    token: str = field(repr=False)
    queue_size: int = 100
    timeout: float = 10.0
    max_retries: int = 4
    backoff_initial: float = 1.0
    backoff_max: float = 30.0
    session: Any = None
    sleep: Callable[[float], None] | None = None  # injectable for tests

    def __post_init__(self) -> None:
        self._queue: collections.deque = collections.deque(maxlen=self.queue_size)
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._stats = PublisherStats()
        self._session = self.session or requests.Session()
        self._thread = threading.Thread(target=self._run, name="event-publisher", daemon=True)
        self._thread.start()

    # -- public API ---------------------------------------------------------
    def submit(self, event: dict[str, Any]) -> None:
        with self._cond:
            if len(self._queue) == self._queue.maxlen:
                dropped = self._queue.popleft()
                self._stats.dropped += 1
                log.warning("publish queue full; dropped oldest event", extra={"event_id": dropped["event_id"]})
            self._queue.append(event)
            self._cond.notify()

    def stats(self) -> PublisherStats:
        with self._cond:
            self._stats.queue_len = len(self._queue)
            return PublisherStats(**vars(self._stats))

    def close(self, timeout: float = 5.0) -> None:
        """Stop accepting work; give in-flight/queued events up to ``timeout`` seconds, then exit."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.warning("publisher did not finish within shutdown timeout", extra={"queued": len(self._queue)})
        self._session.close()

    # -- worker thread ------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop.is_set():
                    self._cond.wait(timeout=1.0)
                if not self._queue:
                    return  # stopped and drained
                event = self._queue.popleft()
            self._deliver(event)

    def _wait(self, seconds: float) -> bool:
        """Sleep for backoff; returns False if shutdown was requested meanwhile."""
        if self.sleep is not None:
            self.sleep(seconds)
            return not self._stop.is_set()
        return not self._stop.wait(seconds)

    def _backoff(self, attempt: int) -> float:
        base = min(self.backoff_max, self.backoff_initial * (2 ** attempt))
        return base * random.uniform(0.5, 1.0)  # jitter avoids synchronized retry storms

    def _deliver(self, event: dict[str, Any]) -> bool:
        body = json.dumps(event, separators=(",", ":"))
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            delay: float | None = None
            try:
                resp = self._session.post(self.url, data=body, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                error = f"{type(exc).__name__}"
                delay = self._backoff(attempt)
            else:
                if 200 <= resp.status_code < 300:
                    self._record_success(event, resp)
                    return True
                error = f"HTTP {resp.status_code}"
                if resp.status_code not in RETRYABLE_STATUS:
                    self._record_failure(event, error, detail=_short_body(resp))
                    return False
                delay = self._backoff(attempt)
                if resp.status_code == 429:
                    delay = max(delay, min(self.backoff_max, _retry_after(resp)))
            if attempt == self.max_retries:
                break
            with self._cond:
                self._stats.retries += 1
            log.warning("event delivery failed; will retry",
                        extra={"event_id": event["event_id"], "error": error, "attempt": attempt + 1,
                               "retry_in_s": round(delay, 2), "url": redact_url(self.url)})
            if not self._wait(delay):
                break
        self._record_failure(event, error)
        return False

    def _record_success(self, event: dict[str, Any], resp: Any) -> None:
        with self._cond:
            self._stats.sent += 1
            self._stats.last_success_at = format_utc(utc_now())
        log.debug("event delivered", extra={"event_id": event["event_id"], "status": resp.status_code,
                                            "response": _short_body(resp)})

    def _record_failure(self, event: dict[str, Any], error: str, detail: str = "") -> None:
        with self._cond:
            self._stats.failed += 1
            self._stats.last_error = error
        log.error("event delivery failed permanently",
                  extra={"event_id": event["event_id"], "error": error, "detail": detail})


def _retry_after(resp: Any) -> float:
    try:
        return max(0.0, float(resp.headers.get("Retry-After", "0")))
    except (TypeError, ValueError):
        return 0.0


def _short_body(resp: Any, limit: int = 300) -> str:
    try:
        return (resp.text or "")[:limit]
    except Exception:
        return ""
