"""YOLO person detection + ByteTrack tracking (wraps the Ultralytics model used by the POC).

Track IDs are local to one tracker session: they reset when the tracker is reset
(e.g. after a stream reconnect) and do not identify people across cameras or
sessions. ``people_count`` is the number of people visible in the current frame,
not a unique-visitor count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ByteTrack associates low-score boxes in its second stage, so the model is run at a
# low threshold and DETECTION_CONFIDENCE is applied when counting (parse_person_result).
TRACKER_INPUT_CONFIDENCE = 0.1


@dataclass(frozen=True)
class FrameCount:
    people_count: int
    track_ids: list[int]  # unique, sorted; may be shorter than people_count (untracked detections)


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def parse_person_result(result: Any, person_class_id: int, min_confidence: float) -> FrameCount:
    """Count person boxes in an Ultralytics ``Results`` object (or a duck-typed stand-in).

    Ultralytics' tracking mode returns tracker output with ``boxes.id`` set; on frames
    where the tracker has no confirmed tracks it returns raw detections with
    ``boxes.id is None``. Both are counted; only valid IDs are reported.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return FrameCount(0, [])
    cls = _as_numpy(boxes.cls)
    conf = _as_numpy(boxes.conf)
    if cls is None or conf is None or cls.shape != conf.shape:
        raise ValueError("detection result has inconsistent cls/conf arrays")
    mask = (cls.astype(int) == person_class_id) & (conf >= min_confidence) & np.isfinite(conf)
    count = int(mask.sum())
    ids = _as_numpy(getattr(boxes, "id", None))
    track_ids: list[int] = []
    if ids is not None and ids.shape == cls.shape:
        selected = ids[mask]
        selected = selected[np.isfinite(selected) & (selected >= 0)]
        track_ids = sorted({int(i) for i in selected})
    return FrameCount(count, track_ids)


def find_person_class_id(names: dict[int, str] | list[str]) -> int:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for idx, name in items:
        if str(name).lower() == "person":
            return int(idx)
    raise ValueError("model has no 'person' class; use a COCO-pretrained model such as yolo11m.pt")


class PersonTracker:
    """Owns the YOLO model; one instance per camera (ByteTrack state lives in the predictor)."""

    def __init__(self, model_path: str, tracker_config: str = "bytetrack.yaml", confidence: float = 0.35,
                 image_size: int = 640, device: str = "", model: Any = None) -> None:
        if model is None:
            from ultralytics import YOLO  # imported lazily so unit tests don't need the weights

            model = YOLO(model_path)
        self._model = model
        self._tracker_config = tracker_config
        self._confidence = confidence
        self._image_size = image_size
        self._device = device or None
        self.person_class_id = find_person_class_id(model.names)

    def reset(self) -> None:
        """Clear ByteTrack state (call after a reconnect: old tracks are meaningless).

        Note: Ultralytics binds ``persist`` once, on the first ``track()`` call, so we
        always pass ``persist=True`` and reset the tracker objects directly instead of
        toggling ``persist`` (which would silently recreate the tracker every frame).
        """
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def process(self, frame: np.ndarray) -> FrameCount:
        if frame is None or getattr(frame, "size", 0) == 0:
            return FrameCount(0, [])
        results = self._model.track(
            frame,
            persist=True,
            tracker=self._tracker_config,
            classes=[self.person_class_id],
            conf=TRACKER_INPUT_CONFIDENCE,
            imgsz=self._image_size,
            device=self._device,
            verbose=False,
        )
        if not results:
            return FrameCount(0, [])
        return parse_person_result(results[0], self.person_class_id, self._confidence)

    def close(self) -> None:
        """Drop model references so GPU/CPU memory can be reclaimed."""
        model, self._model = self._model, None
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch optional in tests
            pass
