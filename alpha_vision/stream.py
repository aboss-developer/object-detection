"""Resilient frame source for RTSP (or a local file in development).

A reader thread drains the stream continuously and keeps only the *latest*
frame (a one-slot buffer), so slow inference never builds an unbounded backlog
or processes stale frames. Disconnects, open failures, read timeouts and stalls
trigger reconnects with bounded exponential backoff.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .redact import redact_url
from .rtsp_probe import ProbeResult, probe_rtsp
from .timeutil import utc_now

log = logging.getLogger(__name__)


class Backoff:
    """Exponential backoff with jitter, capped at ``maximum``."""

    def __init__(self, initial: float, maximum: float, jitter: bool = True) -> None:
        self.initial, self.maximum, self.jitter = initial, maximum, jitter
        self.attempt = 0

    def next(self) -> float:
        delay = min(self.maximum, self.initial * (2 ** self.attempt))
        self.attempt += 1
        return delay * random.uniform(0.8, 1.0) if self.jitter else delay

    def reset(self) -> None:
        self.attempt = 0


def open_opencv_capture(source: str, kind: str, open_timeout: float, read_timeout: float) -> Any:
    import cv2

    if kind == "rtsp":
        # TCP avoids UDP packet loss/firewall issues typical on CCTV networks.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(open_timeout * 1000),
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(read_timeout * 1000)]
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    return cv2.VideoCapture(source)


@dataclass(frozen=True)
class Frame:
    seq: int
    image: Any
    captured_at: datetime  # wall-clock UTC when the frame was read
    generation: int  # increments on every successful (re)connect


class FrameSource:
    def __init__(
        self,
        source: str,
        kind: str,
        *,
        capture_factory: Callable[[], Any],
        stall_timeout: float = 15.0,
        backoff: Backoff | None = None,
        auth_failure_backoff: float = 300.0,
        loop_file: bool = False,
        probe: Callable[[str], ProbeResult] | None = probe_rtsp,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source, self._kind = source, kind
        self._factory = capture_factory
        self._stall_timeout = stall_timeout
        self._backoff = backoff or Backoff(1.0, 60.0)
        self._auth_backoff = auth_failure_backoff
        self._loop_file = loop_file
        self._probe = probe if kind == "rtsp" else None
        self._clock = clock

        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._latest: Frame | None = None
        self._seq = 0
        self._generation = 0
        self._state = "idle"
        self._last_frame_mono: float | None = None
        self._last_error: str | None = None
        self._last_diagnosis: str | None = None
        self._reconnects = 0
        self._failure_streak = 0
        self._cap: Any = None
        self._thread: threading.Thread | None = None

    # -- public -------------------------------------------------------------
    @property
    def redacted_source(self) -> str:
        return redact_url(self._source) if self._kind == "rtsp" else self._source

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="frame-reader", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> bool:
        """Signal the reader to stop and wait for it to release the capture. Returns True if it exited."""
        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.error("frame reader did not exit within timeout (blocked in read?)")
                return False
        return True

    @property
    def ended(self) -> bool:
        return self._state in {"ended", "stopped"}

    def latest(self, after_seq: int, timeout: float) -> Frame | None:
        """Return the newest frame with seq > after_seq, waiting up to ``timeout`` seconds."""
        deadline = self._clock() + timeout
        with self._lock:
            while not self._stop.is_set():
                if self._latest is not None and self._latest.seq > after_seq:
                    return self._latest
                if self.ended:
                    return None
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return None
                self._lock.wait(min(remaining, 0.5))
        return None

    def seconds_since_frame(self) -> float | None:
        last = self._last_frame_mono
        return None if last is None else self._clock() - last

    def status(self) -> dict[str, Any]:
        since = self.seconds_since_frame()
        return {
            "state": self._state,
            "input_kind": self._kind,
            "source": self.redacted_source,
            "generation": self._generation,
            "reconnects": self._reconnects,
            "seconds_since_last_frame": None if since is None else round(since, 2),
            "last_error": self._last_error,
            "last_diagnosis": self._last_diagnosis,
        }

    # -- reader thread ------------------------------------------------------
    def _set_state(self, state: str) -> None:
        if state != self._state:
            log.info("stream state change", extra={"from": self._state, "to": state, "source": self.redacted_source})
            self._state = state
            with self._lock:
                self._lock.notify_all()

    def _sleep(self, seconds: float) -> bool:
        return not self._stop.wait(seconds)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._set_state("connecting")
                cap = self._open()
                if cap is None:
                    if not self._sleep(self._failure_delay()):
                        break
                    continue
                outcome = self._read_loop(cap)
                self._release(cap)
                if outcome == "ended":
                    self._set_state("ended")
                    return
                if outcome == "stopped":
                    break
                self._reconnects += 1
                self._set_state("reconnecting")
                if not self._sleep(self._backoff.next()):
                    break
        except Exception:
            log.exception("frame reader crashed")
            self._last_error = "reader crashed"
            self._set_state("ended")
            return
        self._set_state("stopped")

    def _open(self) -> Any:
        try:
            cap = self._factory()
        except Exception as exc:  # factory errors must never leak the URL
            cap, self._last_error = None, f"open raised {type(exc).__name__}"
        if cap is not None and cap.isOpened():
            self._failure_streak = 0
            self._backoff.reset()
            self._generation += 1
            self._cap = cap
            self._last_error = None
            self._last_diagnosis = None
            self._set_state("streaming")
            log.info("stream connected", extra={"source": self.redacted_source, "generation": self._generation,
                                                "width": _prop(cap, 3), "height": _prop(cap, 4),
                                                "stream_fps": _prop(cap, 5)})
            return cap
        if cap is not None:
            self._release(cap)
        self._failure_streak += 1
        self._last_error = self._last_error or "could not open stream"
        log.warning("stream open failed", extra={"source": self.redacted_source, "attempt": self._failure_streak})
        if self._probe is not None and self._failure_streak == 1:
            # One diagnostic probe per failure streak (avoids extra failed logins on the NVR).
            result = self._probe(self._source)
            self._last_diagnosis = result.stage
            log.error("rtsp diagnosis", extra={"stage": result.stage, "status_code": result.status_code,
                                               "hint": result.hint, "detail": result.detail})
        self._set_state("auth_backoff" if self._last_diagnosis == "auth_failed" else "reconnecting")
        return None

    def _failure_delay(self) -> float:
        if self._last_diagnosis == "auth_failed":
            return self._auth_backoff
        return self._backoff.next()

    def _read_loop(self, cap: Any) -> str:
        frame_interval = 0.0
        if self._kind == "file":
            fps = _prop(cap, 5) or 25.0
            frame_interval = 1.0 / fps if fps > 0 else 0.04
        self._last_frame_mono = self._clock()
        while not self._stop.is_set():
            started = self._clock()
            ok, image = cap.read()
            now = self._clock()
            if self._stop.is_set():
                return "stopped"
            if not ok or image is None:
                if self._kind == "file":
                    if self._loop_file:
                        log.info("video file finished; looping")
                        return "reconnect"
                    log.info("video file finished")
                    return "ended"
                self._last_error = "read failed (disconnect or read timeout)"
                log.warning("stream read failed; reconnecting", extra={"source": self.redacted_source})
                return "reconnect"
            if now - self._last_frame_mono > self._stall_timeout:
                self._last_error = f"stream stalled for {now - self._last_frame_mono:.1f}s"
                log.warning("stream stalled; reconnecting", extra={"gap_s": round(now - self._last_frame_mono, 2)})
                return "reconnect"
            self._last_frame_mono = now
            with self._lock:
                self._seq += 1
                self._latest = Frame(self._seq, image, utc_now(), self._generation)
                self._lock.notify_all()
            if frame_interval:
                elapsed = self._clock() - started
                if not self._sleep(max(0.0, frame_interval - elapsed)):
                    return "stopped"
        return "stopped"

    def _release(self, cap: Any) -> None:
        try:
            cap.release()
        except Exception:
            log.warning("capture release raised", exc_info=True)
        if self._cap is cap:
            self._cap = None


def _prop(cap: Any, prop_id: int) -> float | None:
    try:
        value = float(cap.get(prop_id))
        return round(value, 2) if value > 0 else None
    except Exception:
        return None


__all__ = ["Backoff", "Frame", "FrameSource", "open_opencv_capture"]
