from datetime import datetime, timedelta, timezone

import pytest

from alpha_vision.events import EventValidationError, PublishPolicy, build_count_event, validate_event
from alpha_vision.timeutil import format_utc, parse_utc, utc_now

NOW = datetime(2026, 10, 17, 10, 0, 0, 123456, tzinfo=timezone.utc)


def make(**overrides):
    event = build_count_event(camera_id="camera-01", timestamp=NOW, people_count=2, tracked_person_ids=[5, 1],
                              sequence=1, input_kind="rtsp")
    event.update(overrides)
    return event


def test_valid_event_shape():
    event = make()
    assert event["timestamp"] == "2026-10-17T10:00:00.123Z"
    assert event["tracked_person_ids"] == [1, 5]
    assert event["schema_version"] == 1
    assert event["event_type"] == "people_count_update"
    assert len(event["event_id"]) == 32
    assert "rtsp://" not in str(event)
    validate_event(event)


@pytest.mark.parametrize("overrides", [
    {"people_count": -1}, {"people_count": 1.5}, {"people_count": True}, {"people_count": "2"},
    {"tracked_person_ids": [1, 1]}, {"tracked_person_ids": [1, 2, 3]}, {"tracked_person_ids": [-1]},
    {"tracked_person_ids": "1,2"}, {"camera_id": "bad id"}, {"event_type": "alert"},
    {"timestamp": "2026-10-17 10:00:00"}, {"timestamp": "2026-10-17T10:00:00+05:30"}, {"timestamp": 1760695200},
    {"schema_version": 2}, {"input_kind": "youtube"}, {"event_id": "not-hex"}, {"sequence": -3},
    {"extra": "field"},
])
def test_invalid_events_rejected(overrides):
    with pytest.raises(EventValidationError):
        validate_event(make(**overrides))


def test_missing_field_rejected():
    event = make()
    del event["camera_id"]
    with pytest.raises(EventValidationError, match="missing"):
        validate_event(event)


def test_future_timestamp_rejected():
    event = make(timestamp=format_utc(NOW + timedelta(minutes=10)))
    with pytest.raises(EventValidationError, match="future"):
        validate_event(event, now=NOW)


def test_utc_formatting_and_parsing_roundtrip():
    ist = timezone(timedelta(hours=5, minutes=30))
    local = datetime(2026, 10, 17, 15, 30, 0, tzinfo=ist)
    assert format_utc(local) == "2026-10-17T10:00:00.000Z"
    assert parse_utc("2026-10-17T10:00:00Z") == datetime(2026, 10, 17, 10, 0, tzinfo=timezone.utc)
    assert parse_utc("2026-10-17T10:00:00.5Z").microsecond == 500000
    with pytest.raises(ValueError):
        format_utc(datetime(2026, 10, 17))  # naive
    assert abs((parse_utc(format_utc(utc_now())) - utc_now()).total_seconds()) < 5


class TestPublishPolicy:
    def policy(self):
        return PublishPolicy(min_interval=2.0, heartbeat_interval=30.0, stable_samples=2)

    def test_first_sample_published(self):
        assert self.policy().should_publish(0, 0.0)

    def test_unchanged_count_is_not_republished_until_heartbeat(self):
        p = self.policy()
        p.should_publish(3, 0.0)
        assert not any(p.should_publish(3, t * 0.5) for t in range(1, 59))
        assert p.should_publish(3, 30.0)

    def test_single_frame_flicker_ignored(self):
        p = self.policy()
        p.should_publish(3, 0.0)
        assert not p.should_publish(4, 5.0)  # seen once
        assert not p.should_publish(3, 5.5)  # back to 3 -> candidate cleared
        assert not p.should_publish(4, 6.0)

    def test_stable_change_published(self):
        p = self.policy()
        p.should_publish(3, 0.0)
        assert not p.should_publish(5, 5.0)
        assert p.should_publish(5, 5.5)

    def test_rate_limited_by_min_interval(self):
        p = self.policy()
        p.should_publish(1, 0.0)
        assert not p.should_publish(2, 0.5)
        assert not p.should_publish(2, 1.0)  # stable but < 2 s since last publish
        assert p.should_publish(2, 2.0)

    def test_publish_rate_is_bounded_under_noise(self):
        p = self.policy()
        published = sum(p.should_publish(i % 7, i * 0.5) for i in range(1200))  # 10 minutes at 2 FPS
        assert published <= 600 / 2.0 + 1
