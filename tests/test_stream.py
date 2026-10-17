import threading
import time

import numpy as np

from alpha_vision.rtsp_probe import ProbeResult
from alpha_vision.stream import Backoff, FrameSource

URL = "rtsp://admin:Secret123@192.0.2.10:554/Streaming/Channels/102"


class FakeCapture:
    def __init__(self, frames=3, opened=True, fail_after=True, block=None):
        self.frames, self.opened, self.block = frames, opened, block
        self.released = False
        self.reads = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if self.block is not None:
            self.block.wait(5)
        if self.reads >= self.frames:
            return False, None
        self.reads += 1
        time.sleep(0.005)
        return True, np.full((4, 4, 3), self.reads, dtype=np.uint8)

    def get(self, prop):
        return 25.0 if prop == 5 else 640.0

    def release(self):
        self.released = True


class Factory:
    def __init__(self, caps):
        self.caps = list(caps)
        self.created = []

    def __call__(self):
        cap = self.caps.pop(0) if self.caps else FakeCapture(opened=False)
        self.created.append(cap)
        return cap


def wait_for(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def source(factory, kind="rtsp", probe=None, **kw):
    return FrameSource(URL if kind == "rtsp" else "clip.mp4", kind, capture_factory=factory,
                       backoff=Backoff(0.01, 0.05, jitter=False),
                       probe=probe or (lambda url: ProbeResult(False, "tcp_timeout")), **kw)


def test_backoff_is_exponential_and_bounded():
    b = Backoff(1.0, 10.0, jitter=False)
    assert [b.next() for _ in range(6)] == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0]
    b.reset()
    assert b.next() == 1.0


def test_reconnects_after_disconnect_and_bumps_generation():
    factory = Factory([FakeCapture(frames=2), FakeCapture(opened=False), FakeCapture(frames=50)])
    src = source(factory)
    src.start()
    assert wait_for(lambda: src.status()["generation"] == 2)
    frame = src.latest(after_seq=0, timeout=1)
    assert frame is not None and frame.captured_at.tzinfo is not None
    assert src.stop()
    assert all(c.released for c in factory.created)  # every capture released, including the failed one
    assert src.status()["reconnects"] >= 1
    assert src.status()["state"] == "stopped"


def test_open_failure_runs_one_diagnosis_per_streak():
    probes = []

    def probe(url):
        probes.append(url)
        return ProbeResult(False, "tcp_refused")

    factory = Factory([FakeCapture(opened=False)] * 4 + [FakeCapture(frames=100)])
    src = source(factory, probe=probe)
    src.start()
    assert wait_for(lambda: src.status()["state"] == "streaming")
    src.stop()
    assert len(probes) == 1


def test_auth_failure_uses_long_backoff():
    factory = Factory([FakeCapture(opened=False)] * 3)
    src = source(factory, probe=lambda url: ProbeResult(False, "auth_failed", 401), auth_failure_backoff=60)
    src.start()
    assert wait_for(lambda: src.status()["state"] == "auth_backoff")
    time.sleep(0.2)
    assert len(factory.created) == 1  # not hammering the NVR with bad credentials
    assert src.status()["last_diagnosis"] == "auth_failed"
    started = time.monotonic()
    assert src.stop()
    assert time.monotonic() - started < 2  # shutdown interrupts the long backoff


def test_status_and_logs_never_contain_password(caplog):
    factory = Factory([FakeCapture(opened=False)])
    src = source(factory)
    src.start()
    wait_for(lambda: src.status()["last_error"] is not None)
    src.stop()
    assert "Secret123" not in str(src.status())
    assert "Secret123" not in caplog.text


def test_stall_triggers_reconnect():
    clock_offset = [0.0]
    lock = threading.Lock()

    def clock():
        with lock:
            return time.monotonic() + clock_offset[0]

    class StallingCapture(FakeCapture):
        def read(self):
            if self.reads == 2:
                with lock:
                    clock_offset[0] += 100  # simulate a read that returned after a long hang
            return super().read()

    factory = Factory([StallingCapture(frames=10), FakeCapture(frames=100)])
    src = source(factory, stall_timeout=5.0, clock=clock)
    src.start()
    assert wait_for(lambda: src.status()["generation"] == 2)
    src.stop()
    assert "stalled" in (src.status()["last_error"] or "") or src.status()["reconnects"] >= 1


def test_file_source_ends_without_reconnect():
    factory = Factory([FakeCapture(frames=3)])
    src = source(factory, kind="file")
    src.start()
    assert wait_for(lambda: src.ended)
    assert len(factory.created) == 1
    assert factory.created[0].released
    assert src.status()["input_kind"] == "file"


def test_latest_only_keeps_one_frame():
    factory = Factory([FakeCapture(frames=40)])
    src = source(factory)
    src.start()
    assert wait_for(lambda: src._seq >= 40)
    frame = src.latest(after_seq=0, timeout=1)
    assert frame.seq == 40  # newest frame only; nothing queued behind it
    assert src.latest(after_seq=frame.seq, timeout=0.1) is None
    src.stop()


def test_stop_reports_blocked_reader():
    gate = threading.Event()
    factory = Factory([FakeCapture(frames=10, block=gate)])
    src = source(factory)
    src.start()
    wait_for(lambda: src.status()["state"] == "streaming")
    assert src.stop(timeout=0.2) is False
    gate.set()
    assert wait_for(lambda: not src._thread.is_alive())
    assert factory.created[0].released
