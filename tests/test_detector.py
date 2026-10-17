import numpy as np
import pytest

from alpha_vision.detector import FrameCount, PersonTracker, find_person_class_id, parse_person_result
from tests.conftest import FakeBoxes, FakeResult

PERSON = 0


def test_counts_only_confident_people_and_reports_valid_ids():
    boxes = FakeBoxes(cls=[0, 0, 2, 0, 0], conf=[0.9, 0.8, 0.95, 0.2, 0.6], ids=[7, 3, 9, 4, 3])
    result = parse_person_result(FakeResult(boxes), PERSON, 0.35)
    # 3 confident people (ids 7, 3, 3 -> duplicate id reported once); car (cls 2) and low-conf person excluded
    assert result.people_count == 3
    assert result.track_ids == [3, 7]


def test_untracked_detections_still_counted():
    boxes = FakeBoxes(cls=[0, 0], conf=[0.9, 0.9], ids=None)
    assert parse_person_result(FakeResult(boxes), PERSON, 0.35) == FrameCount(2, [])


@pytest.mark.parametrize("result", [FakeResult(None), FakeResult(FakeBoxes([], [])), object()])
def test_empty_results(result):
    assert parse_person_result(result, PERSON, 0.35) == FrameCount(0, [])


def test_invalid_ids_dropped():
    boxes = FakeBoxes(cls=[0, 0, 0], conf=[0.9, 0.9, 0.9], ids=[np.nan, -1, 5])
    assert parse_person_result(FakeResult(boxes), PERSON, 0.35) == FrameCount(3, [5])


def test_inconsistent_arrays_raise():
    boxes = FakeBoxes(cls=[0, 0], conf=[0.9])
    with pytest.raises(ValueError):
        parse_person_result(FakeResult(boxes), PERSON, 0.35)


def test_find_person_class():
    assert find_person_class_id({0: "person", 1: "bicycle"}) == 0
    assert find_person_class_id(["car", "Person"]) == 1
    with pytest.raises(ValueError, match="person"):
        find_person_class_id({0: "forklift"})  # custom model without a person class


class FakeTracker:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class FakeModel:
    names = {0: "person", 2: "car"}

    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.predictor = type("P", (), {"trackers": [FakeTracker()]})()

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [self.results.pop(0)]


def test_tracker_integration_uses_bytetrack_and_persists(blank_frame):
    model = FakeModel([FakeResult(FakeBoxes([0], [0.9], [1])), FakeResult(FakeBoxes([0, 0], [0.9, 0.7], [1, 2]))])
    tracker = PersonTracker("unused.pt", "bytetrack.yaml", confidence=0.35, model=model)
    assert tracker.process(blank_frame) == FrameCount(1, [1])
    assert tracker.process(blank_frame) == FrameCount(2, [1, 2])
    for call in model.calls:
        # persist must always be True: Ultralytics binds it on the first call.
        assert call["persist"] is True
        assert call["tracker"] == "bytetrack.yaml"
        assert call["classes"] == [0]


def test_tracker_reset_clears_bytetrack_state():
    model = FakeModel([])
    tracker = PersonTracker("unused.pt", model=model)
    tracker.reset()
    assert model.predictor.trackers[0].resets == 1


def test_empty_frame_skips_inference():
    model = FakeModel([])
    tracker = PersonTracker("unused.pt", model=model)
    assert tracker.process(np.zeros((0, 0, 3), dtype=np.uint8)) == FrameCount(0, [])
    assert tracker.process(None) == FrameCount(0, [])
    assert model.calls == []


def test_close_releases_model():
    tracker = PersonTracker("unused.pt", model=FakeModel([]))
    tracker.close()
    assert tracker._model is None
