import type { CountEvent } from "./types";

export const CAMERA_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const EVENT_ID_RE = /^[0-9a-f]{32}$/;
const ISO_UTC_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$/;
const FIELDS = [
  "schema_version",
  "event_id",
  "sequence",
  "camera_id",
  "event_type",
  "timestamp",
  "people_count",
  "tracked_person_ids",
  "input_kind",
  "source",
] as const;
export const MAX_PEOPLE = 1000;
export const MAX_FUTURE_SKEW_MS = 2 * 60_000;

export type Validation = { ok: true; event: CountEvent; timestampMs: number } | { ok: false; error: string };

const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);

export function parseUtc(value: unknown): number | null {
  if (typeof value !== "string" || !ISO_UTC_RE.test(value)) return null;
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms : null;
}

/** Validate a schema v1 people_count_update event (mirrors alpha_vision/events.py). */
export function validateEvent(body: unknown, nowMs: number): Validation {
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    return { ok: false, error: "body must be a JSON object" };
  }
  const e = body as Record<string, unknown>;
  const missing = FIELDS.filter((f) => !(f in e));
  if (missing.length) return { ok: false, error: `missing fields: ${missing.join(", ")}` };
  const extra = Object.keys(e).filter((k) => !(FIELDS as readonly string[]).includes(k));
  if (extra.length) return { ok: false, error: `unexpected fields: ${extra.slice(0, 5).join(", ")}` };
  if (e.schema_version !== 1) return { ok: false, error: "schema_version must be 1" };
  if (typeof e.event_id !== "string" || !EVENT_ID_RE.test(e.event_id)) {
    return { ok: false, error: "event_id must be 32 lowercase hex chars" };
  }
  if (!isInt(e.sequence) || e.sequence < 0) return { ok: false, error: "sequence must be a non-negative integer" };
  if (typeof e.camera_id !== "string" || !CAMERA_ID_RE.test(e.camera_id)) {
    return { ok: false, error: "camera_id is invalid" };
  }
  if (e.event_type !== "people_count_update") return { ok: false, error: "event_type must be people_count_update" };
  const ts = parseUtc(e.timestamp);
  if (ts === null) return { ok: false, error: "timestamp must be UTC ISO 8601 ending in Z" };
  if (ts - nowMs > MAX_FUTURE_SKEW_MS) return { ok: false, error: "timestamp is too far in the future" };
  if (!isInt(e.people_count) || e.people_count < 0 || e.people_count > MAX_PEOPLE) {
    return { ok: false, error: `people_count must be an integer in [0, ${MAX_PEOPLE}]` };
  }
  const ids = e.tracked_person_ids;
  if (!Array.isArray(ids) || !ids.every((i) => isInt(i) && i >= 0)) {
    return { ok: false, error: "tracked_person_ids must be a list of non-negative integers" };
  }
  if (new Set(ids).size !== ids.length) return { ok: false, error: "tracked_person_ids must be unique" };
  if (ids.length > e.people_count) return { ok: false, error: "tracked_person_ids cannot exceed people_count" };
  if (e.input_kind !== "rtsp" && e.input_kind !== "file") return { ok: false, error: "input_kind must be rtsp or file" };
  if (e.source !== "alpha-vision-edge") return { ok: false, error: "source must be alpha-vision-edge" };
  return { ok: true, event: e as unknown as CountEvent, timestampMs: ts };
}
