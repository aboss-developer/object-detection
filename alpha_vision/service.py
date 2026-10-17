"""Main processing loop: sample frames -> track people -> publish count events."""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from dataclasses import asdict
from typing import Any, Callable

from . import __version__
from .config import Settings
from .detector import FrameCount, PersonTracker
from .events import PublishPolicy, build_count_event
from .health import write_status
from .publisher import HttpPublisher, LogOnlyPublisher, Publisher
from .stream import Backoff, FrameSource, open_opencv_capture
from .timeutil import format_utc, utc_now

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_STUCK = 3  # reader thread wedged: let the supervisor restart the process


class CameraService:
    def __init__(
        self,
        settings: Settings,
        *,
        tracker: Any,
        frames: FrameSource,
        publisher: Publisher,
        event_log: logging.Logger | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.s = settings
        self.tracker = tracker
        self.frames = frames
        self.publisher = publisher
        self.event_log = event_log
        self.clock = clock
        self.stop_event = threading.Event()
        self.policy = PublishPolicy(settings.event_min_interval_seconds, settings.event_publish_interval_seconds,
                                    settings.count_stable_samples)
        self._sequence = 0
        self._started_at = utc_now()
        self._last_count: FrameCount | None = None
        self._last_inference_at: str | None = None
        self._inference_ms: float | None = None
        self._frames_processed = 0
        self._last_status = 0.0
        self._generation = 0

    # -- lifecycle ----------------------------------------------------------
    def request_stop(self, reason: str = "requested") -> None:
        if not self.stop_event.is_set():
            log.info("shutdown requested", extra={"reason": reason})
            self.stop_event.set()

    def install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self.request_stop(signal.Signals(signum).name)

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):  # not main thread / unsupported on this platform
                    pass

    def run(self) -> int:
        interval = 1.0 / self.s.frame_sample_fps
        stuck_after = max(3 * self.s.stall_timeout_seconds, 30.0)
        log.info("service starting", extra={
            "camera_id": self.s.camera_id, "input_kind": self.s.source_kind, "source": self.frames.redacted_source,
            "sample_fps": self.s.frame_sample_fps, "version": __version__})
        if self.s.source_kind == "file":
            log.warning("running on a local VIDEO_FILE, not a live camera; events are marked input_kind=file")
        self.frames.start()
        exit_code = EXIT_OK
        last_seq = 0
        next_sample = self.clock()
        try:
            while not self.stop_event.is_set():
                wait = next_sample - self.clock()
                if wait > 0 and self.stop_event.wait(wait):
                    break
                next_sample = max(next_sample + interval, self.clock())
                frame = self.frames.latest(after_seq=last_seq, timeout=min(1.0, interval * 2))
                self._maybe_write_status()
                if frame is None:
                    if self.frames.ended:
                        log.info("frame source ended")
                        break
                    since = self.frames.seconds_since_frame()
                    if since is not None and since > stuck_after and self.frames.status()["state"] == "streaming":
                        log.error("no frames while 'streaming' (reader likely blocked); exiting for supervisor restart",
                                  extra={"seconds_since_last_frame": round(since, 1)})
                        exit_code = EXIT_STUCK
                        break
                    continue
                last_seq = frame.seq
                self._process(frame)
        finally:
            self.shutdown()
        return exit_code

    def shutdown(self) -> None:
        log.info("shutting down")
        reader_exited = self.frames.stop(timeout=self.s.read_timeout_seconds + 5)
        self.publisher.close(timeout=self.s.http_timeout_seconds + 2)
        close = getattr(self.tracker, "close", None)
        if close:
            close()
        self._write_status(final=True, reader_exited=reader_exited)
        log.info("shutdown complete")

    # -- processing ---------------------------------------------------------
    def _process(self, frame) -> None:
        if frame.generation != self._generation:
            if self._generation:
                log.info("stream reconnected; resetting tracker", extra={"generation": frame.generation})
            self.tracker.reset()
            self.policy.reset()
            self._generation = frame.generation
        started = time.perf_counter()
        result: FrameCount = self.tracker.process(frame.image)
        self._inference_ms = round((time.perf_counter() - started) * 1000, 1)
        self._frames_processed += 1
        self._last_count = result
        self._last_inference_at = format_utc(frame.captured_at)
        if not self.policy.should_publish(result.people_count, self.clock()):
            return
        self._sequence += 1
        event = build_count_event(
            camera_id=self.s.camera_id,
            timestamp=frame.captured_at,
            people_count=result.people_count,
            tracked_person_ids=result.track_ids,
            sequence=self._sequence,
            input_kind=self.s.source_kind,
        )
        if self.event_log is not None:
            self.event_log.info(json.dumps(event, separators=(",", ":")))
        self.publisher.submit(event)

    # -- status -------------------------------------------------------------
    def _maybe_write_status(self) -> None:
        if self.clock() - self._last_status >= self.s.status_interval_seconds:
            self._write_status()

    def _write_status(self, final: bool = False, reader_exited: bool | None = None) -> None:
        self._last_status = self.clock()
        status = {
            "camera_id": self.s.camera_id,
            "version": __version__,
            "updated_at": format_utc(utc_now()),
            "started_at": format_utc(self._started_at),
            "running": not final,
            "stream": self.frames.status(),
            "people_count": None if self._last_count is None else self._last_count.people_count,
            "tracked_person_ids": None if self._last_count is None else self._last_count.track_ids,
            "last_inference_at": self._last_inference_at,
            "inference_ms": self._inference_ms,
            "frames_processed": self._frames_processed,
            "events_generated": self._sequence,
            "publisher": asdict(self.publisher.stats()),
            "publishing_enabled": self.s.publishing_enabled,
        }
        if reader_exited is not None:
            status["reader_exited_cleanly"] = reader_exited
        try:
            write_status(self.s.status_file, status)
        except OSError:
            log.warning("could not write status file", exc_info=True)


def build_service(settings: Settings, event_log: logging.Logger | None = None) -> CameraService:
    tracker = PersonTracker(settings.model_path, settings.tracker_config, settings.detection_confidence,
                            settings.inference_image_size, settings.device)
    frames = FrameSource(
        settings.source,
        settings.source_kind,
        capture_factory=lambda: open_opencv_capture(settings.source, settings.source_kind,
                                                    settings.open_timeout_seconds, settings.read_timeout_seconds),
        stall_timeout=settings.stall_timeout_seconds,
        backoff=Backoff(settings.reconnect_initial_seconds, settings.reconnect_max_seconds),
        auth_failure_backoff=settings.auth_failure_backoff_seconds,
        loop_file=settings.loop_file_source,
    )
    if settings.publishing_enabled:
        publisher: Publisher = HttpPublisher(
            settings.worker_url, settings.worker_ingest_token, queue_size=settings.publish_queue_size,
            timeout=settings.http_timeout_seconds, max_retries=settings.http_max_retries,
            backoff_initial=settings.http_backoff_initial_seconds, backoff_max=settings.http_backoff_max_seconds,
        )
    else:
        log.warning("WORKER_URL not set: events are logged locally only, nothing is sent to Cloudflare")
        publisher = LogOnlyPublisher()
    return CameraService(settings, tracker=tracker, frames=frames, publisher=publisher, event_log=event_log)
