# Alpha Vision: single-camera people-count MVP

```
Hikvision NVR (DS-7608NXI-K1) ──RTSP──▶ alpha_vision edge service ──HTTPS──▶ Cloudflare Worker ──▶ Telegram
                                         YOLO11 person detection          auth + validation
                                         ByteTrack tracking               per-camera Durable Object
                                         live count + events              threshold / cooldown alerts
```

> **Status: MVP, not production-certified.** Everything below the "Verified" line in
> [Validation status](#validation-status) still has to be tested against the physical NVR and network.
> Licensed under **AGPL-3.0**. Commercial and production use is still an open release gate; see [License](#license).

### Project history

This repository started as a single-script proof of concept: `streamingvideo.py` ran YOLO11 (`yolo11m.pt`) on a
public HLS traffic-camera stream, drew boxes in a `cv2.imshow` window and dumped detections to JSON. That script was
removed when Alpha Vision was added, and it is still in the Git history (commit `f1db1fc`). Alpha Vision reuses the
POC's Ultralytics detection and `model.track(..., persist=True)` approach, and fixes its problems:

| POC problem | Fix |
|---|---|
| `cv2.imshow` / `waitKey` required a display | Fully headless; no GUI calls |
| `results_list` grew forever and the whole JSON was rewritten every 100 frames | Fixed-size one-frame buffer; events appended to a size-rotated `logs/events.jsonl` |
| Timestamps from `frame_num / fps` (wrong for live streams), and milliseconds stored in `timestamp_sec` | Wall-clock UTC ISO 8601 (`...Z`) taken when the frame is read |
| `while True` reconnect every 5 s with no backoff and no stall detection | Bounded exponential backoff with jitter, read and open timeouts, stall detection, long back-off after auth failures |
| No shutdown handling; captures not always released | SIGINT/SIGTERM/SIGBREAK handling, ordered shutdown, captures released |
| Hardcoded URLs and cookie file | Environment-based config, credentials redacted everywhere |

## Layout

```
alpha_vision/            edge service (Python 3.10)
  config.py              env config + startup validation + Hikvision URL builder
  stream.py              RTSP/file reader thread, reconnect/backoff/stall handling
  rtsp_probe.py          RTSP DESCRIBE probe (DNS / port / auth / channel diagnosis)
  detector.py            YOLO + ByteTrack wrapper, result parsing, live count
  events.py              event schema v1, validation, publish-rate policy
  publisher.py           bounded queue, HTTP timeouts, retries with backoff
  service.py             main loop, signals, status file
  redact.py, logging_setup.py, timeutil.py, health.py
tests/                   unit tests (mocks, synthetic frames, fake RTSP server)
tests/integration/       live NVR test: opt-in only
tests/fixtures/          README.md only; supply your own local test video (git-ignored)
worker/                  Cloudflare Worker (TypeScript) + Durable Object
docs/event-schema.md     event contract
.env.example             edge config template (no secrets)
LICENSE                  GNU AGPL-3.0
```

Not in Git (by design): `yolo11m.pt` model weights, test videos, `.env`, `worker/.dev.vars`, logs, `.venv`,
`node_modules`. See [Model weights](#model-weights) and [Test video](#test-video).

## 1. Install

Requirements: Python 3.10 (the verified version), Git, and Node.js ≥ 20 for the Worker.

```powershell
git clone https://github.com/aboss-developer/object-detection.git
cd object-detection
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

`requirements-edge.txt` pins the versions verified on 2026-10-17 (`ultralytics 8.3.201`, `opencv-python 4.12.0.88`,
`lap 0.5.12`, `numpy 2.2.6`). Ultralytics installs `torch`; by default pip installs a **CPU build**. For an NVIDIA
GPU, install a CUDA build of torch from pytorch.org first and set `INFERENCE_DEVICE=0`. See [Performance](#performance).

Worker:

```powershell
cd worker
npm ci          # exact versions from package-lock.json
```

### Model weights

`yolo11m.pt` (Ultralytics YOLO11 medium, COCO-pretrained, 40,684,120 bytes) is **not committed**. It's large, and it's
distributed by Ultralytics under AGPL-3.0. Download it into the repository root (the default `MODEL_PATH`):

```powershell
curl.exe -L -o yolo11m.pt https://huggingface.co/Ultralytics/YOLO11/resolve/1d41cfddc10550b04b71ce479a791cc3e772a15d/yolo11m.pt
certutil -hashfile yolo11m.pt SHA256
# expected: d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95
```

This is the official Ultralytics Hugging Face release, pinned to the revision the project was verified with.
Ultralytics also publishes it at `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt`.
The unit tests don't need the weights. `check-config` and `run` fail with a clear error until the file exists.

### Test video

No video is committed: footage of people is personal data, and third-party clip licences are often unclear.
The unit tests don't need one. For the optional real-model smoke test and camera-free local runs, put your own short
clip of people at `tests/fixtures/sample_video5.mp4`. See [tests/fixtures/README.md](tests/fixtures/README.md).

## 2. Configure the edge service

```powershell
copy .env.example .env      # then edit .env locally; it is git-ignored
```

Set exactly **one** source:

* `NVR_HOST`, `NVR_USERNAME`, `NVR_PASSWORD`, `NVR_CHANNEL`, `NVR_STREAM`, `NVR_RTSP_PORT` (recommended). This builds
  `rtsp://<user>:<pass>@<nvr-ip>:554/Streaming/Channels/<channel><01|02>`. Channel 1 sub stream = `102`,
  main = `101`. Special characters in the password are encoded automatically.
* `RTSP_URL`: a full URL.
* `VIDEO_FILE`: a recorded video for development. Events are tagged `input_kind: "file"`, so a mock never
  passes for the live camera.

Recommendations for the NVR:
* Create a **dedicated low-privilege NVR user** with *Remote Live View* on the needed channel only. Don't use `admin`.
* Use the **sub stream** (`NVR_STREAM=sub`), with H.264 at about 640×360 to 704×576. Detection doesn't need the main stream, and it costs bandwidth and CPU.

Validate the config, then test the camera:

```powershell
.venv\Scripts\python -m alpha_vision check-config
.venv\Scripts\python -m alpha_vision check-rtsp          # RTSP handshake + decodes 10 frames
```

`check-rtsp` reports one of `ok`, `dns`, `tcp_refused`, `tcp_timeout`, `auth_failed`, `not_found`,
`forbidden` or `protocol`, with a hint. It makes **at most one** authenticated attempt per run.

## 3. Run the edge service

```powershell
.venv\Scripts\python -m alpha_vision run        # headless; Ctrl+C for graceful shutdown
```

Outputs, all in `LOG_DIR` (default `logs/`), size-rotated and with secrets redacted:

* `alpha_vision.log`: structured JSON log lines (also printed to stderr).
* `events.jsonl`: every generated event (replaces the POC's growing JSON dumps).
* `status.json`: rewritten atomically every `STATUS_INTERVAL_SECONDS`. Contains stream state
  (`connecting` / `streaming` / `reconnecting` / `auth_backoff` / `ended` / `stopped`), seconds since the last frame,
  last error and diagnosis, current count, inference latency, and publisher counters (sent, failed, dropped,
  retries, queue length).

Exit codes: `0` = clean stop, `2` = configuration error, `3` = frame reader stuck, so the supervisor should restart.

**Long-running supervision (Windows):** run it as a service with auto-restart, for example with NSSM:
`nssm install AlphaVision C:\path\to\object-detection\.venv\Scripts\python.exe -m alpha_vision run`, with
*AppDirectory* set to the repository folder and restart-on-exit enabled. On Linux, use a systemd unit with `Restart=always`.
SIGTERM is handled.

## 4. Cloudflare Worker

`worker/` is a standalone Worker project. There was no existing Worker in this repository.

* `POST /v1/events` takes `Authorization: Bearer <INGEST_TOKEN>` and a JSON event. It returns a JSON result.
* `GET /health` returns liveness only, with no configuration details.
* Alert state lives in a **SQLite-backed Durable Object per `camera_id`**. Events for a camera are serialized,
  so concurrent or retried requests can't cause duplicate alerts. In-memory globals aren't used for state.
  No D1, R2, KV or Queues.

Settings (`worker/wrangler.jsonc` → `vars`):

| Var | Default | Meaning |
|---|---|---|
| `PERSON_ALERT_THRESHOLD` | `5` | Alert when `people_count >= threshold`. |
| `ALERT_COOLDOWN_SECONDS` | `600` | Minimum gap between two sent alerts for one camera. |
| `MAX_EVENT_AGE_SECONDS` | `300` | Older events are accepted but never trigger alerts. |
| `ALLOWED_CAMERA_IDS` | `camera-01` | Comma list; empty = any valid ID. Keep it set. |

Secrets: **never** put these in `wrangler.jsonc` or Git.

| Secret | Purpose |
|---|---|
| `INGEST_TOKEN` | Shared bearer token (≥ 32 chars), equal to the edge `WORKER_INGEST_TOKEN`. If it's missing, the Worker rejects all ingest. |
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather. It stays in Cloudflare and never reaches the edge device. |
| `TELEGRAM_CHAT_ID` | Target chat or group ID. |

### Alert rules

1. `count >= threshold` and no active alert → send a Telegram alert, unless still in the cooldown or in a Telegram back-off.
2. Still above the threshold → `suppressed_active`, no repeats.
3. Count drops **below** the threshold → `rearmed`.
4. The next crossing alerts again, but only if `ALERT_COOLDOWN_SECONDS` have passed since the last *sent* alert. Otherwise it's `suppressed_cooldown`, and it alerts on a later update once the cooldown has passed, if the count is still high.
5. Telegram error or timeout → `failed`, and the alert isn't marked sent. The next count update (at least every 30 s) retries it, so the edge doesn't need to retry and there's no retry storm.
6. Telegram 429 → no attempts until `retry_after` has passed.
7. A duplicate `event_id` (edge retry) → no-op. Stale or out-of-order events never trigger alerts.

Why these defaults: a threshold of 5 is a placeholder, so set it for the real space. The 10-minute cooldown keeps a busy
period that hovers around the threshold from paging repeatedly. Edge debouncing (`COUNT_STABLE_SAMPLES=2`) means a
single noisy frame can't trigger an alert.

Message format (plain text):
```
🚨 People threshold reached
Camera: camera-01
Event: people_threshold_alert
People in view: 6 (threshold 5)
Time (UTC): 2026-10-17T10:00:00.123Z
```

### Telegram setup
1. In Telegram, message **@BotFather** → `/newbot` → copy the token. Treat it like a password.
2. Add the bot to the target group, or start a chat with it, and send one message.
3. Get the chat ID: open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser **on your own machine**
   and read `message.chat.id`. Group IDs are negative. Don't paste that URL anywhere, because it contains the token.

### Run the Worker locally (no deployment)
```powershell
cd worker
copy .dev.vars.example .dev.vars   # local only, git-ignored; set INGEST_TOKEN (and optionally a TEST chat)
npm test                           # unit tests
npm run typecheck
npm run dev                        # http://127.0.0.1:8787
```
Then point the edge at it in `.env`: `WORKER_URL=http://127.0.0.1:8787/v1/events`, and `WORKER_INGEST_TOKEN`
set to the same value as `INGEST_TOKEN` in `.dev.vars`.

### Deploy (only after approval)
```powershell
cd worker
npx wrangler login
npx wrangler secret put INGEST_TOKEN
npx wrangler secret put TELEGRAM_BOT_TOKEN
npx wrangler secret put TELEGRAM_CHAT_ID
npx wrangler deploy
```
Then set `WORKER_URL=https://alpha-vision-worker.<your-subdomain>.workers.dev/v1/events` on the edge. The edge only accepts `https://`, except for localhost.

## 5. Tests

| Suite | Command | Needs |
|---|---|---|
| Python unit tests | `.venv\Scripts\python -m pytest` | Nothing: no camera, network, Telegram, credentials, weights or video |
| Lint | `.venv\Scripts\ruff check` | Nothing |
| Worker tests + typecheck | `cd worker; npm ci; npm test; npm run typecheck` | Nothing (Node only) |
| Real-model smoke test (optional) | `.venv\Scripts\python -m pytest -m model` | `yolo11m.pt` in the repo root **and** `tests/fixtures/sample_video5.mp4`. Otherwise it's reported as **SKIPPED**, not passed. |
| Live NVR (optional) | see below | A reachable NVR and credentials in `.env` |

A fresh clone passes the unit, lint and Worker checks without any downloads beyond `pip`/`npm` dependencies.

The live NVR test **cannot run by accident**. It's deselected by `pytest.ini` and also needs an env flag:

```powershell
$env:ALPHA_VISION_LIVE_RTSP = "1"
.venv\Scripts\python -m pytest -m integration tests/integration -s
```

Covered by unit tests: detection parsing and live count, empty frames and no detections, tracker persistence and reset,
event validation (missing, invalid and extra fields, non-UTC timestamps, units), publish-rate policy, publisher timeouts,
retries, backoff, 429 handling, non-retryable 4xx, bounded queue and shutdown, reconnect and backoff, auth-failure
back-off, stall detection, resource release, blocked readers, secret redaction, the RTSP probe against a fake RTSP
server (Digest auth with and without qop, 401, 404, refused, non-RTSP), service end-to-end with mocks, Worker auth,
validation, routing, alert threshold, cooldown and re-arm, concurrency, dedupe, stale events, and Telegram success,
API error, 429, network errors and timeouts.

## Validation status

**Verified on the development machine (Windows 11, Python 3.10, CPU torch, 2026-10-17):**
* Python unit suite: 87 passed, 2 skipped (POSIX-only SIGTERM test; live NVR module). Ruff clean.
* Real-model smoke test (`-m model`): yolo11m + ByteTrack on a local sample video produced counts and track IDs. Passed.
* Fresh clone with a clean venv built only from `requirements-dev.txt` (resolved `torch 2.14.1+cpu`): unit tests,
  Ruff, Worker tests and typecheck pass; the smoke test passes once weights and a video are supplied, and is skipped otherwise.
* Headless end-to-end run on a sample video (`VIDEO_FILE`): clean exit, valid events, status file written.
* Worker: 45 vitest tests passed, `tsc` typecheck clean.
* `wrangler dev` (local workerd with a real Durable Object): checked auth 401, 400, 422 and 403 responses,
  `not_triggered`, a crossing (→ `not_configured`, because Telegram secrets were intentionally blank) and dedupe.
  The edge service published 5 events to it with HTTP 200.
* The ingest token didn't appear in edge or Worker logs. `check-rtsp` diagnoses refused and unreachable ports.

**Still needs validation on the physical NVR, network and accounts:**
* The real RTSP URL, channel or stream mapping, credentials and codec on the DS-7608NXI-K1. Run `check-rtsp`, then the
  `integration` tests.
* Reconnect behaviour when the NVR or camera actually reboots, cables are pulled, or the network drops. OpenCV/FFmpeg read
  timeouts on Windows with a real RTSP server. Hikvision lockout behaviour.
* Detection accuracy for the actual camera angle, lighting and night IR. Choose `DETECTION_CONFIDENCE` and the threshold from real footage.
* Sustained multi-hour and multi-day runs: memory stability and log rotation in practice.
* Real Telegram delivery (bot, chat ID, message formatting) and a deployed Worker over the internet.
* Running as a supervised Windows service (NSSM) and recovery after reboot.

## Performance

On this laptop's CPU-only torch build, `yolo11m` takes about **0.53–0.60 s per frame**, so roughly 1.7 FPS is achievable,
slightly under the 2 FPS target. The service always processes the newest frame, so it degrades gracefully and never
falls behind. Options, in order: install a CUDA build of torch for the RTX 2050 and set `INFERENCE_DEVICE=0`; use the sub
stream; lower `INFERENCE_IMAGE_SIZE`; or try `yolo11n.pt`/`yolo11s.pt`. Each one needs accuracy re-checking, and model changes need approval.

## Troubleshooting

| Symptom (`check-rtsp` stage / log) | Likely cause and fix |
|---|---|
| `dns` | Use the NVR IP address instead of a hostname. |
| `tcp_timeout` | Wrong IP, different VLAN or subnet, firewall, or NVR offline. `ping <nvr-ip>`; check `Test-NetConnection <nvr-ip> -Port 554`. |
| `tcp_refused` | Wrong port, or RTSP disabled. On the NVR: Configuration → Network → Advanced Settings → Port. |
| `auth_failed` (401) | Wrong user or password, or the user lacks permission. **Stop and fix it.** Hikvision locks the IP after several failures (about 30 min). The service waits `AUTH_FAILURE_BACKOFF_SECONDS` between attempts. |
| `not_found` (404) | Wrong channel or stream: channel *N* → `N01` main / `N02` sub. Check that the camera shows online on that NVR channel. |
| `forbidden` (403) | The NVR user lacks *Remote Live View* for the channel. |
| `protocol` | The port isn't RTSP (for example HTTP 80 or SDK 8000). |
| Handshake OK but no frames | Codec issue: set the sub stream to H.264 (not H.265+/Smart codec). Also check bandwidth. |
| `status.json` shows `reconnecting` repeatedly | Unstable network or stream. Check `last_error` and the cable or switch. Sub stream is recommended. |
| Exit code 3 | The reader was stuck inside FFmpeg. Let the supervisor restart it; investigate the network. |
| Worker `401` in edge log | `WORKER_INGEST_TOKEN` ≠ Worker `INGEST_TOKEN`. |
| Worker `500 server_misconfigured` | `INGEST_TOKEN` secret not set on the Worker. |
| `alert.status: not_configured` | `TELEGRAM_BOT_TOKEN` or `TELEGRAM_CHAT_ID` secret missing. |
| `alert.status: failed` with `chat not found` | Wrong chat ID, or the bot isn't in the group or never messaged. |
| `alert.status: rate_limited` | Telegram 429; the Worker waits `retry_after`. |
| `publisher.dropped` > 0 in status | The Worker was unreachable for a long time; the oldest events were dropped (bounded queue). |

## License

This repository is licensed under the **GNU Affero General Public License v3.0**. See [LICENSE](LICENSE).

It depends on **Ultralytics YOLO11** (the `ultralytics` package and the `yolo11m.pt` weights), which Ultralytics also
licenses under AGPL-3.0, with a separate paid Enterprise License. ByteTrack (MIT) is used through Ultralytics'
implementation. The weights aren't redistributed here; users download them from Ultralytics.

**Release gate (still open):** publishing this source under AGPL-3.0 doesn't settle how a production deployment may be
operated or sold. Under AGPL-3.0, users who interact with a modified version over a network must be offered its
source. Any closed-source, proprietary or commercial use needs a separate decision, for example an Ultralytics
Enterprise License or a permissively licensed detector, after legal review. **No compliance claim is made here.**

## Remaining blockers before production

1. Licensing and commercial-use decision (above).
2. Physical NVR validation (see the list above) and threshold or confidence tuning on real footage.
3. A GPU torch build, or a model and resolution choice, to reliably reach the target FPS.
4. Worker deployment and real Telegram verification (requires approval and secrets).
5. A service supervisor (NSSM or systemd) and monitoring of `status.json` and logs. There is currently no alert when the edge
   itself goes silent. A Worker-side "camera offline" alert (no heartbeat for N minutes, using a Durable Object alarm) is a
   recommended next step.
6. Privacy: the system counts people from CCTV. Confirm signage and local data-protection requirements. Only counts and
   session track IDs leave the site; no images do.
