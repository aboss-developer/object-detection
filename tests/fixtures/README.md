# Test fixtures

Video files are **not** committed to this repository. Footage of people is personal data, and the licence of
third-party clips is often unclear. `*.mp4` and other video formats are git-ignored.

Only the optional real-model smoke test (`pytest -m model`) and local manual runs need a video. The normal unit-test
suite uses synthetic frames and mocks, and runs without any fixture.

## What the smoke test expects

| Item | Requirement |
|---|---|
| Path | `tests/fixtures/sample_video5.mp4` (the name the test looks for) |
| Content | At least one clearly visible **person** in the first 15 frames, walking or standing, so ByteTrack assigns IDs |
| Format | Any container and codec that OpenCV/FFmpeg can decode (H.264 MP4 is safest) |
| Length | A few seconds is enough. Keep it short (< 30 MB). |

If the file is missing, the smoke test is **skipped**, not passed. Pytest shows it as `SKIPPED`.

## Ways to get a suitable clip

1. **Record your own** (preferred): a short phone clip of colleagues who have agreed to be filmed, or of yourself,
   walking across the frame. Export as MP4.
2. **Export from the NVR**: download a short clip from the Hikvision NVR (web UI → Playback → Clip/Download) of a
   period with people in view. Only do this with the site owner's permission, and keep the file local.
3. **A clearly licensed stock clip**: for example, a free-licence pedestrian clip from a stock-video site whose licence
   allows this use. Check and record the licence before using it.

Then:

```powershell
copy <your-clip>.mp4 tests\fixtures\sample_video5.mp4
.venv\Scripts\python -m pytest -m model
```

For a manual end-to-end run without a camera, set `VIDEO_FILE=tests/fixtures/sample_video5.mp4` in `.env`. Events
are tagged `input_kind: "file"`, and any Telegram alert is labelled TEST INPUT.

Never commit the clip. `git status` should not list it. If it does, check `.gitignore` rather than forcing anything.
