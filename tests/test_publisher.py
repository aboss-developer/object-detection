import json
import threading
import time

import requests

from alpha_vision.publisher import HttpPublisher

TOKEN = "k" * 40


class Resp:
    def __init__(self, status, headers=None, text=""):
        self.status_code, self.headers, self.text = status, headers or {}, text


class FakeSession:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.closed = False
        self.lock = threading.Lock()

    def post(self, url, data, headers, timeout):
        with self.lock:
            self.calls.append({"url": url, "body": json.loads(data), "headers": headers, "timeout": timeout})
            step = self.script.pop(0) if self.script else Resp(200)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step()
        return step

    def close(self):
        self.closed = True


def event(n=1):
    return {"event_id": f"{n:032x}", "people_count": n}


def make(script, **kw):
    sleeps = []
    session = FakeSession(script)
    pub = HttpPublisher("https://w.example/v1/events", TOKEN, session=session, sleep=sleeps.append, **kw)
    return pub, session, sleeps


def wait_for(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_success_sends_auth_header_and_timeout():
    pub, session, _ = make([Resp(200)], timeout=7.5)
    pub.submit(event())
    assert wait_for(lambda: pub.stats().sent == 1)
    pub.close()
    call = session.calls[0]
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert call["timeout"] == 7.5
    assert session.closed


def test_retries_timeouts_with_backoff_and_same_event_id():
    pub, session, sleeps = make([requests.Timeout(), requests.ConnectionError(), Resp(503), Resp(200)],
                                max_retries=4, backoff_initial=1.0, backoff_max=30.0)
    pub.submit(event(5))
    assert wait_for(lambda: pub.stats().sent == 1)
    pub.close()
    assert len(session.calls) == 4
    assert {c["body"]["event_id"] for c in session.calls} == {event(5)["event_id"]}  # dedupe key preserved
    assert len(sleeps) == 3
    assert sleeps[0] <= 1.0 and sleeps[1] <= 2.0 and sleeps[2] <= 4.0  # exponential with jitter
    assert pub.stats().retries == 3


def test_retries_are_bounded():
    pub, session, sleeps = make([Resp(500)] * 10, max_retries=2)
    pub.submit(event())
    assert wait_for(lambda: pub.stats().failed == 1)
    pub.close()
    assert len(session.calls) == 3
    assert pub.stats().sent == 0


def test_client_errors_not_retried():
    for status in (400, 401, 403, 422):
        pub, session, sleeps = make([Resp(status, text='{"error":"x"}')])
        pub.submit(event())
        assert wait_for(lambda p=pub: p.stats().failed == 1)
        pub.close()
        assert len(session.calls) == 1 and sleeps == []


def test_429_honours_retry_after_capped():
    pub, session, sleeps = make([Resp(429, {"Retry-After": "12"}), Resp(200)], backoff_max=10.0)
    pub.submit(event())
    assert wait_for(lambda: pub.stats().sent == 1)
    pub.close()
    assert sleeps == [10.0]


def test_queue_is_bounded_and_drops_oldest():
    gate = threading.Event()

    def blocked():
        gate.wait(5)
        return Resp(200)

    pub, session, _ = make([blocked] + [Resp(200)] * 20, queue_size=3)
    pub.submit(event(1))
    assert wait_for(lambda: len(session.calls) == 1)  # worker busy with event 1
    for n in range(2, 8):
        pub.submit(event(n))
    stats = pub.stats()
    assert stats.queue_len == 3 and stats.dropped == 3
    gate.set()
    assert wait_for(lambda: pub.stats().sent == 4)
    pub.close()
    assert [c["body"]["people_count"] for c in session.calls] == [1, 5, 6, 7]


def test_close_interrupts_backoff():
    session = FakeSession([Resp(503)] * 10)
    pub = HttpPublisher("https://w.example/v1/events", TOKEN, session=session, max_retries=5,
                        backoff_initial=30.0, backoff_max=30.0)
    pub.submit(event())
    assert wait_for(lambda: len(session.calls) == 1)
    started = time.monotonic()
    pub.close(timeout=3)
    assert time.monotonic() - started < 3
    assert not pub._thread.is_alive()
