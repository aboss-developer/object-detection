"""Slow check with the real yolo11m.pt + ByteTrack on the fixture video (no camera).

    .venv/Scripts/python -m pytest -m model
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.model
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "yolo11m.pt"
VIDEO = ROOT / "tests" / "fixtures" / "sample_video5.mp4"


@pytest.mark.skipif(not (WEIGHTS.is_file() and VIDEO.is_file()), reason="weights or sample video missing")
def test_real_model_tracks_people_on_sample_video():
    import cv2

    from alpha_vision.detector import PersonTracker

    tracker = PersonTracker(str(WEIGHTS), "bytetrack.yaml", confidence=0.35, device="cpu")
    cap = cv2.VideoCapture(str(VIDEO))
    counts, all_ids = [], set()
    try:
        for _ in range(15):
            ok, frame = cap.read()
            assert ok
            result = tracker.process(frame)
            counts.append(result.people_count)
            all_ids.update(result.track_ids)
            assert len(result.track_ids) <= result.people_count
    finally:
        cap.release()
        tracker.close()
    assert max(counts) > 0, "expected people in sample_video5.mp4"
    assert all_ids, "ByteTrack should assign track IDs across consecutive frames"
