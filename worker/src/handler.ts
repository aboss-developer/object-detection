/**
 * HTTP layer: routing, authentication, payload validation, JSON responses.
 * Kept free of `cloudflare:workers` imports so it can be unit-tested in Node.
 */
import type { AlertConfig, Env } from "./types";
import { validateEvent } from "./validate";

export const MAX_BODY_BYTES = 8 * 1024;
export const MIN_TOKEN_LENGTH = 32;

type ErrorCode =
  | "not_found"
  | "method_not_allowed"
  | "unauthorized"
  | "unsupported_media_type"
  | "payload_too_large"
  | "invalid_json"
  | "invalid_event"
  | "camera_not_allowed"
  | "server_misconfigured"
  | "internal_error";

export function json(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...headers },
  });
}

const fail = (status: number, code: ErrorCode, message: string, headers?: Record<string, string>) =>
  json(status, { ok: false, error: { code, message } }, headers);

function intVar(value: string | undefined, fallback: number, min: number, max: number): number {
  const n = Number(value);
  return Number.isInteger(n) && n >= min && n <= max ? n : fallback;
}

export function readAlertConfig(env: Env): AlertConfig {
  return {
    threshold: intVar(env.PERSON_ALERT_THRESHOLD, 5, 1, 1000),
    cooldownMs: intVar(env.ALERT_COOLDOWN_SECONDS, 600, 0, 86_400) * 1000,
    maxEventAgeMs: intVar(env.MAX_EVENT_AGE_SECONDS, 300, 10, 86_400) * 1000,
    telegramBotToken: env.TELEGRAM_BOT_TOKEN ?? "",
    telegramChatId: env.TELEGRAM_CHAT_ID ?? "",
  };
}

async function sha256(text: string): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
}

/** Constant-time comparison (hash both sides so length differences leak nothing). */
export async function tokensEqual(provided: string, expected: string): Promise<boolean> {
  const [a, b] = await Promise.all([sha256(provided), sha256(expected)]);
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i]! ^ b[i]!;
  return diff === 0;
}

function allowedCameras(env: Env): Set<string> | null {
  const raw = (env.ALLOWED_CAMERA_IDS ?? "").trim();
  if (!raw) return null;
  return new Set(raw.split(",").map((s) => s.trim()).filter(Boolean));
}

export async function handleRequest(request: Request, env: Env, nowMs: number = Date.now()): Promise<Response> {
  const url = new URL(request.url);

  if (url.pathname === "/health") {
    return request.method === "GET" ? json(200, { ok: true, service: "alpha-vision-worker" }) : fail(405, "method_not_allowed", "use GET", { allow: "GET" });
  }
  if (url.pathname !== "/v1/events") return fail(404, "not_found", "unknown route");
  if (request.method !== "POST") return fail(405, "method_not_allowed", "use POST", { allow: "POST" });

  const expected = env.INGEST_TOKEN ?? "";
  if (expected.length < MIN_TOKEN_LENGTH) {
    console.error(JSON.stringify({ msg: "INGEST_TOKEN secret missing or too short; rejecting all ingest" }));
    return fail(500, "server_misconfigured", "ingest is not configured");
  }
  const header = request.headers.get("authorization") ?? "";
  const match = /^Bearer\s+(\S+)$/i.exec(header);
  if (!match || !(await tokensEqual(match[1]!, expected))) {
    console.warn(JSON.stringify({ msg: "ingest auth failed", has_header: header.length > 0 }));
    return fail(401, "unauthorized", "missing or invalid bearer token", { "www-authenticate": "Bearer" });
  }

  if (!(request.headers.get("content-type") ?? "").toLowerCase().startsWith("application/json")) {
    return fail(415, "unsupported_media_type", "content-type must be application/json");
  }
  const declared = Number(request.headers.get("content-length") ?? "0");
  if (declared > MAX_BODY_BYTES) return fail(413, "payload_too_large", `max ${MAX_BODY_BYTES} bytes`);
  const text = await request.text();
  if (new TextEncoder().encode(text).length > MAX_BODY_BYTES) {
    return fail(413, "payload_too_large", `max ${MAX_BODY_BYTES} bytes`);
  }
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return fail(400, "invalid_json", "body is not valid JSON");
  }

  const v = validateEvent(body, nowMs);
  if (!v.ok) return fail(422, "invalid_event", v.error);
  const allowed = allowedCameras(env);
  if (allowed && !allowed.has(v.event.camera_id)) {
    console.warn(JSON.stringify({ msg: "camera not allowed", camera_id: v.event.camera_id }));
    return fail(403, "camera_not_allowed", "camera_id is not in ALLOWED_CAMERA_IDS");
  }

  try {
    const result = await env.CAMERA_STATE.getByName(v.event.camera_id).processEvent(v.event);
    console.log(
      JSON.stringify({
        msg: "event processed",
        camera_id: v.event.camera_id,
        event_id: v.event.event_id,
        people_count: v.event.people_count,
        input_kind: v.event.input_kind,
        alert: result.alert.status,
        detail: result.alert.detail,
      }),
    );
    return json(200, { ok: true, event_id: v.event.event_id, ...result });
  } catch (err) {
    console.error(JSON.stringify({ msg: "event processing failed", error: err instanceof Error ? err.name : "unknown" }));
    return fail(500, "internal_error", "could not process event");
  }
}
