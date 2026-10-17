# Event schema v1: `people_count_update`

Sent by the edge service (`alpha_vision`) to `POST /v1/events` on the Worker.
Validated on both sides: `alpha_vision/events.py` and `worker/src/validate.ts`.
Unknown or missing fields are rejected.

```json
{
  "schema_version": 1,
  "event_id": "9f1c2e7a4b5d4c3e8f0a1b2c3d4e5f60",
  "sequence": 42,
  "camera_id": "camera-01",
  "event_type": "people_count_update",
  "timestamp": "2026-10-17T10:00:00.123Z",
  "people_count": 5,
  "tracked_person_ids": [1, 2, 3, 4, 5],
  "input_kind": "rtsp",
  "source": "alpha-vision-edge"
}
```

| Field | Type | Rules / meaning |
|---|---|---|
| `schema_version` | int | Always `1`. Bump on breaking change. |
| `event_id` | string | 32 lowercase hex chars (UUID4). **Idempotency key**: retries reuse it, and the Worker ignores duplicates. |
| `sequence` | int ≥ 0 | Increases per event within one edge process. Resets on restart, so it's for diagnostics only. |
| `camera_id` | string | `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`. Must be in the Worker's `ALLOWED_CAMERA_IDS` if that is set. |
| `event_type` | string | `people_count_update` only. Alerts are derived by the Worker, not sent by the edge. |
| `timestamp` | string | Wall-clock time the sampled frame was **read**, UTC ISO 8601 with `Z`, millisecond precision. Rejected if > 2 min in the future. Used for alerting only if ≤ `MAX_EVENT_AGE_SECONDS` old and not older than the last accepted event. |
| `people_count` | int 0–1000 | People visible in that frame with confidence ≥ `DETECTION_CONFIDENCE`. **Live count, not unique visitors.** |
| `tracked_person_ids` | int[] | Unique, sorted ByteTrack IDs of counted people. `len ≤ people_count` (a detection may not have a track ID yet). IDs are local to one tracker session, reset on reconnect/restart, and **do not identify people** across cameras or sessions. |
| `input_kind` | `"rtsp"` \| `"file"` | `file` means a recorded video was used (development). Telegram alerts are labelled TEST INPUT. |
| `source` | string | `alpha-vision-edge`. |

Not included, by design: stream URLs or credentials, images, bounding boxes, video.

## When events are sent

* First sample after startup.
* A **changed** count that is stable for `COUNT_STABLE_SAMPLES` consecutive samples, at most once per `EVENT_MIN_INTERVAL_SECONDS`.
* A heartbeat with the current count every `EVENT_PUBLISH_INTERVAL_SECONDS` if nothing changed.

At the default 2 FPS this is at most 1 request every 2 s, and typically 1 every 30 s.

## Worker response

`200` `{"ok":true,"event_id":"…","duplicate":false,"alert":{"status":"…","detail?":"…"},"people_count":5,"threshold":5}`

`alert.status` ∈ `not_triggered`, `rearmed`, `sent`, `suppressed_active`, `suppressed_cooldown`,
`rate_limited`, `failed`, `not_configured`, `skipped_stale`, `duplicate`.

Errors: `{"ok":false,"error":{"code":"…","message":"…"}}`. The edge service doesn't retry 4xx responses except 408 and 429.

| Status | code |
|---|---|
| 400 | `invalid_json` |
| 401 | `unauthorized` |
| 403 | `camera_not_allowed` |
| 404 / 405 | `not_found` / `method_not_allowed` |
| 413 | `payload_too_large` (> 8 KiB) |
| 415 | `unsupported_media_type` |
| 422 | `invalid_event` |
| 500 | `server_misconfigured` (no `INGEST_TOKEN`) / `internal_error` |
