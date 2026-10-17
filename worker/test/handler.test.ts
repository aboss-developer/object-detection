import { afterEach, describe, expect, it, vi } from "vitest";

import { handleRequest, tokensEqual } from "../src/handler";
import { BOT_TOKEN, NOW, TOKEN, type FetchCall, makeEnv, makeEvent, post, telegramFetch } from "./helpers";

afterEach(() => vi.restoreAllMocks());

describe("authentication", () => {
  it.each([
    ["missing header", {}],
    ["wrong token", { authorization: `Bearer ${"y".repeat(48)}` }],
    ["wrong scheme", { authorization: `Basic ${TOKEN}` }],
    ["token prefix only", { authorization: `Bearer ${TOKEN.slice(0, 20)}` }],
  ])("rejects %s with 401 and never alerts", async (_name, headers) => {
    const calls: FetchCall[] = [];
    const { env, stores } = makeEnv({}, telegramFetch([], calls));
    const req = post(makeEvent({ people_count: 9, tracked_person_ids: [] }), { authorization: "", ...headers });
    if (!("authorization" in headers)) req.headers.delete("authorization");
    const res = await handleRequest(req, env, NOW);
    expect(res.status).toBe(401);
    expect(await res.json<any>()).toMatchObject({ ok: false, error: { code: "unauthorized" } });
    expect(calls).toHaveLength(0);
    expect(stores.size).toBe(0);
  });

  it("fails closed when INGEST_TOKEN is not configured", async () => {
    const { env } = makeEnv({ INGEST_TOKEN: undefined });
    const res = await handleRequest(post(makeEvent()), env, NOW);
    expect(res.status).toBe(500);
    expect((await res.json<any>()).error.code).toBe("server_misconfigured");
  });

  it("compares tokens in constant time semantics", async () => {
    expect(await tokensEqual(TOKEN, TOKEN)).toBe(true);
    expect(await tokensEqual(TOKEN + "a", TOKEN)).toBe(false);
  });

  it("never logs the bearer token or bot token", async () => {
    const logs: string[] = [];
    for (const m of ["log", "warn", "error"] as const) vi.spyOn(console, m).mockImplementation((s) => void logs.push(String(s)));
    const { env } = makeEnv({}, telegramFetch([Response.json({ ok: false, description: "Bad Request" }, { status: 400 })]));
    await handleRequest(post(makeEvent({ people_count: 7, tracked_person_ids: [] })), env, NOW);
    await handleRequest(post(makeEvent(), { authorization: "Bearer wrong" }), env, NOW);
    const all = logs.join("\n");
    expect(all).not.toContain(TOKEN);
    expect(all).not.toContain(BOT_TOKEN);
  });
});

describe("routing and payload validation", () => {
  it("serves an unauthenticated health check without config details", async () => {
    const { env } = makeEnv();
    const res = await handleRequest(new Request("https://w.example/health"), env, NOW);
    expect(res.status).toBe(200);
    expect(JSON.stringify(await res.json<any>())).not.toMatch(/TOKEN|THRESHOLD/);
  });

  it("returns 404 / 405 for unknown routes and methods", async () => {
    const { env } = makeEnv();
    expect((await handleRequest(new Request("https://w.example/nope"), env, NOW)).status).toBe(404);
    expect((await handleRequest(new Request("https://w.example/v1/events"), env, NOW)).status).toBe(405);
  });

  it("rejects non-JSON content type, oversize and malformed bodies", async () => {
    const { env } = makeEnv();
    expect((await handleRequest(post(makeEvent(), { "content-type": "text/plain" }), env, NOW)).status).toBe(415);
    expect((await handleRequest(post("x".repeat(9000)), env, NOW)).status).toBe(413);
    expect((await handleRequest(post("{not json"), env, NOW)).status).toBe(400);
  });

  it.each([
    [{ people_count: -1 }],
    [{ people_count: 2.5 }],
    [{ people_count: "3" }],
    [{ camera_id: "../etc" }],
    [{ event_type: "people_threshold_alert" }],
    [{ timestamp: "2026-10-17 10:00:00" }],
    [{ timestamp: "2026-10-17T15:30:00+05:30" }],
    [{ timestamp: "2026-10-17T11:00:00.000Z" }], // 1 h in the future
    [{ tracked_person_ids: [1, 1] }],
    [{ tracked_person_ids: [1, 2, 3] }],
    [{ schema_version: 2 }],
    [{ input_kind: "youtube" }],
    [{ password: "x" }],
  ])("rejects invalid event %j with 422", async (overrides) => {
    const { env } = makeEnv();
    const res = await handleRequest(post(makeEvent(overrides as any)), env, NOW);
    expect(res.status).toBe(422);
    expect((await res.json<any>()).error.code).toBe("invalid_event");
  });

  it("rejects missing fields", async () => {
    const { env } = makeEnv();
    const { camera_id: _drop, ...rest } = makeEvent();
    const res = await handleRequest(post(rest), env, NOW);
    expect(res.status).toBe(422);
  });

  it("enforces ALLOWED_CAMERA_IDS", async () => {
    const { env } = makeEnv();
    const res = await handleRequest(post(makeEvent({ camera_id: "camera-99" })), env, NOW);
    expect(res.status).toBe(403);
  });

  it("accepts a valid event and returns a predictable body", async () => {
    const { env } = makeEnv();
    const event = makeEvent();
    const res = await handleRequest(post(event), env, NOW);
    expect(res.status).toBe(200);
    expect(await res.json<any>()).toEqual({
      ok: true,
      event_id: event.event_id,
      duplicate: false,
      alert: { status: "not_triggered" },
      people_count: 2,
      threshold: 5,
    });
  });
});

describe("end-to-end alerting through the handler", () => {
  it("alerts once on crossing, dedupes retried event, then suppresses while active", async () => {
    const calls: FetchCall[] = [];
    const { env } = makeEnv({}, telegramFetch([], calls));
    const crossing = makeEvent({ people_count: 6, tracked_person_ids: [1, 2, 3, 4, 5, 6] });
    const r1 = await (await handleRequest(post(crossing), env, NOW)).json<any>();
    const retry = await (await handleRequest(post(crossing), env, NOW)).json<any>();
    const r3 = await (await handleRequest(post(makeEvent({ people_count: 7, tracked_person_ids: [] })), env, NOW)).json<any>();
    expect(r1.alert.status).toBe("sent");
    expect(retry).toMatchObject({ duplicate: true, alert: { status: "duplicate" } });
    expect(r3.alert.status).toBe("suppressed_active");
    expect(calls).toHaveLength(1);
    expect(calls[0]!.body.text).toContain("camera-01");
    expect(calls[0]!.body.text).toContain("People in view: 6");
  });

  it("isolates state per camera", async () => {
    const calls: FetchCall[] = [];
    const { env } = makeEnv({ ALLOWED_CAMERA_IDS: "" }, telegramFetch([], calls));
    await handleRequest(post(makeEvent({ people_count: 6, tracked_person_ids: [] })), env, NOW);
    await handleRequest(post(makeEvent({ camera_id: "camera-02", people_count: 6, tracked_person_ids: [] })), env, NOW);
    expect(calls).toHaveLength(2);
  });

  it("returns 200 with alert.status=failed when Telegram errors (edge must not retry-storm)", async () => {
    const { env } = makeEnv({}, telegramFetch([new TypeError("network down")]));
    const res = await handleRequest(post(makeEvent({ people_count: 9, tracked_person_ids: [] })), env, NOW);
    expect(res.status).toBe(200);
    expect((await res.json<any>()).alert).toEqual({ status: "failed", detail: "TypeError" });
  });
});
