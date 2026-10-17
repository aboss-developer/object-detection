import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeBoxes:
    """Duck-typed stand-in for ultralytics.engine.results.Boxes."""

    def __init__(self, cls, conf, ids=None):
        self.cls = np.asarray(cls, dtype=float)
        self.conf = np.asarray(conf, dtype=float)
        self.id = None if ids is None else np.asarray(ids, dtype=float)

    def __len__(self):
        return len(self.cls)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


@pytest.fixture
def blank_frame():
    return np.zeros((360, 640, 3), dtype=np.uint8)


@pytest.fixture
def base_env(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    return {
        "CAMERA_ID": "camera-01",
        "VIDEO_FILE": str(video),
        "MODEL_PATH": str(model),
        "LOG_DIR": str(tmp_path / "logs"),
    }


@pytest.fixture(autouse=True)
def _no_real_env(monkeypatch):
    # Unit tests must never pick up a developer's real RTSP / Worker settings.
    for key in list(os.environ):
        if key.startswith(("RTSP_", "NVR_", "WORKER_", "CAMERA_ID", "VIDEO_FILE")):
            monkeypatch.delenv(key, raising=False)
