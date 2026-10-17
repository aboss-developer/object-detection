import json
import os
import signal
import threading
import time

import pytest

from alpha_vision.config import load_settings
from alpha_vision.detector import FrameCount
from alpha_vision.events import validate_event
from alpha_vision.publisher import PublisherStats
from alpha_vision.service import CameraService
from alpha_vision.stream import Backoff, FrameSource
from tests.test_stream import Factory, FakeCapture


class FakeTracker:
    def __init__(self, counts):
        self.counts = list(counts)
        self.resets = 0
        self.closed = False

    def reset(self):
        self.resets += 1

    def process(self, frame):
        n = self.counts.pop(0) if self.counts else 0
        return FrameCount(n, list(range(1, n + 1)))

    def close(self):
        self.closed = True


class CollectingPublisher:
    def __init__(self):
        self.events, self.closed = [], False

    def submit(self, event):
        self.events.append(event)

    def stats(self):
        return PublisherStats(sent=len(self.events))

    def close(self, timeout=5.0):
        self.closed = True


def make_service(base_env, caps, counts, kind="file", **env):
    settings = load_settings(dict(base_env, FRAME_SAMPLE_FPS="20", EVENT_MIN_INTERVAL_SECONDS="0.5",
                                  EVENT_PUBLISH_INTERVAL_SECONDS="5", COUNT_STABLE_SAMPLES="1",
                                  STATUS_INTERVAL_SECONDS="1", **env))
    frames = FrameSource(settings.source, kind, capture_factory=Factory(caps), backoff=Backoff(0.01, 0.02, False),
                         probe=None)
    tracker, publisher = FakeTracker(counts), CollectingPublisher()
    return CameraService(settings, tracker=tracker, frames=frames, publisher=publisher), tracker, publisher


def test_end_to_end_with_mock_file_stream(base_env):
    service, tracker, publisher = make_service(base_env, [FakeCapture(frames=60)], [0, 0, 2, 2, 2, 2, 3, 3, 3, 3])
    assert service.run() == 0
    assert publisher.events, "expected at least the initial event"
    for e in publisher.events:
        validate_event(e)
        assert e["input_kind"] == "file"  # never pretends to be a live camera
        assert len(e["tracked_person_ids"]) <= e["people_count"]
    seqs = [e["sequence"] for e in publisher.events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert len(publisher.events) < service._frames_processed  # not one request per frame
    assert tracker.closed and publisher.closed
    status = json.loads(open(service.s.status_file, encoding="utf-8").read())
    assert status["running"] is False and status["stream"]["state"] == "ended"


def test_reconnect_resets_tracker(base_env):
    service, tracker, _ = make_service(base_env, [FakeCapture(frames=5), FakeCapture(frames=5)], [1] * 50,
                                       kind="rtsp")
    t = threading.Thread(target=service.run)
    t.start()
    assert _wait(lambda: service.frames.status()["generation"] >= 2 and tracker.resets >= 2)
    service.request_stop("test")
    t.join(5)
    assert not t.is_alive()


def test_request_stop_shuts_down_and_releases(base_env):
    cap = FakeCapture(frames=10_000)
    service, tracker, publisher = make_service(base_env, [cap], [1] * 10_000)
    t = threading.Thread(target=service.run)
    t.start()
    assert _wait(lambda: service._frames_processed > 2)
    service.request_stop("test")
    t.join(5)
    assert not t.is_alive()
    assert cap.released and tracker.closed and publisher.closed


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal delivery to own process")
def test_sigterm_handler(base_env):
    service, _, _ = make_service(base_env, [FakeCapture(frames=10_000)], [1] * 10_000)
    service.install_signal_handlers()
    threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    assert service.run() == 0


def test_signal_handler_registration_sets_stop(base_env, monkeypatch):
    registered = {}
    monkeypatch.setattr(signal, "signal", lambda sig, fn: registered.__setitem__(sig, fn))
    service, _, _ = make_service(base_env, [], [])
    service.install_signal_handlers()
    assert signal.SIGINT in registered
    registered[signal.SIGINT](signal.SIGINT, None)
    assert service.stop_event.is_set()


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False
