"""Live NVR integration test. NEVER part of the normal suite.

Runs only when BOTH are true:
  * pytest is invoked with ``-m integration`` (pytest.ini deselects it by default), and
  * ALPHA_VISION_LIVE_RTSP=1 plus RTSP_URL or NVR_HOST/NVR_USERNAME/NVR_PASSWORD are set.

    ALPHA_VISION_LIVE_RTSP=1 .venv/Scripts/python -m pytest -m integration tests/integration -s
"""

import os
import time

import pytest

pytestmark = pytest.mark.integration

if os.environ.get("ALPHA_VISION_LIVE_RTSP") != "1":
    pytest.skip("set ALPHA_VISION_LIVE_RTSP=1 to run live RTSP tests", allow_module_level=True)


@pytest.fixture(scope="module")
def settings():
    from alpha_vision.config import load_dotenv_if_present, load_settings

    load_dotenv_if_present(".env")
    s = load_settings()
    if s.source_kind != "rtsp":
        pytest.skip("RTSP_URL or NVR_HOST must be configured")
    return s


def test_rtsp_handshake(settings):
    from alpha_vision.rtsp_probe import probe_rtsp

    result = probe_rtsp(settings.source, timeout=5)
    assert result.ok, f"{result.stage}: {result.hint}"


def test_frames_decode_and_person_tracking_runs(settings):
    from alpha_vision.detector import PersonTracker
    from alpha_vision.stream import FrameSource, open_opencv_capture

    src = FrameSource(settings.source, "rtsp", capture_factory=lambda: open_opencv_capture(
        settings.source, "rtsp", settings.open_timeout_seconds, settings.read_timeout_seconds))
    tracker = PersonTracker(settings.model_path, settings.tracker_config, settings.detection_confidence,
                            settings.inference_image_size, settings.device)
    src.start()
    try:
        seq, results = 0, []
        deadline = time.monotonic() + 60
        while len(results) < 5 and time.monotonic() < deadline:
            frame = src.latest(seq, timeout=5)
            if frame is None:
                continue
            seq = frame.seq
            results.append(tracker.process(frame.image))
        assert len(results) == 5, f"stream status: {src.status()}"
        for r in results:
            assert r.people_count >= 0 and len(r.track_ids) <= r.people_count
        print("live counts:", [r.people_count for r in results])
    finally:
        src.stop()
        tracker.close()
