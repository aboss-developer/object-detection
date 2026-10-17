import { describe, expect, it } from "vitest";

import { SEEN_EVENT_IDS_LIMIT, decide, processEvent } from "../src/alerts";
import { formatAlert, sendTelegram } from "../src/telegram";
import type { AlertConfig } from "../src/types";
import { BOT_TOKEN, type FetchCall, MemoryStorage, NOW, makeEvent, telegramFetch } from "./helpers";

const cfg: AlertConfig = {
  threshold: 5,
  cooldownMs: 600_000,
  maxEventAgeMs: 300_000,
  telegramBotToken: BOT_TOKEN,
  telegramChatId: "-1001",
};
const at = (offsetMs: number) => new Date(NOW + offsetMs).toISOString();
const ev = (count: number, offsetMs = 0) => makeEvent({ people_count: count, tracked_person_ids: [], timestamp: at(offsetMs) });

describe("decide (threshold + cooldown rules)", () => {
  const idle = { alertActive: false, lastAlertAtMs: null, lastEventTsMs: null, telegramBlockedUntilMs: null };
  it("does not alert below threshold", () => expect(decide(idle, 4, NOW, cfg).status).toBe("not_triggered"));
  it("alerts exactly at threshold", () => expect(decide(idle, 5, NOW, cfg).action).toBe("send"));
  it("suppresses while active", () =>
    expect(decide({ ...idle, alertActive: true, lastAlertAtMs: NOW }, 9, NOW + 1, cfg).status).toBe("suppressed_active"));
  it("re-arms when count drops below threshold", () =>
    expect(decide({ ...idle, alertActive: true, lastAlertAtMs: NOW }, 2, NOW, cfg)).toMatchObject({
      status: "rearmed",
      next: { alertActive: false },
    }));
  it("applies cooldown after re-arm", () => {
    const rearmed = { ...idle, lastAlertAtMs: NOW };
    expect(decide(rearmed, 6, NOW + 599_000, cfg).status).toBe("suppressed_cooldown");
    expect(decide(rearmed, 6, NOW + 600_000, cfg).action).toBe("send");
  });
  it("respects Telegram back-off", () =>
    expect(decide({ ...idle, telegramBlockedUntilMs: NOW + 10 }, 6, NOW, cfg).status).toBe("rate_limited"));
});

describe("processEvent (stateful)", () => {
  it("full episode: cross -> suppress -> drop -> cross within cooldown -> cross after cooldown", async () => {
    const storage = new MemoryStorage();
    const calls: FetchCall[] = [];
    const f = telegramFetch([], calls);
    const run = (count: number, t: number) => processEvent(storage, ev(count, t), cfg, NOW + t, f);
    expect((await run(6, 0)).alert.status).toBe("sent");
    expect((await run(8, 30_000)).alert.status).toBe("suppressed_active");
    expect((await run(1, 60_000)).alert.status).toBe("rearmed");
    expect((await run(6, 120_000)).alert.status).toBe("suppressed_cooldown");
    expect((await run(6, 700_000)).alert.status).toBe("sent");
    expect(calls).toHaveLength(2);
  });

  it("concurrent crossings produce a single alert (claim before await)", async () => {
    const storage = new MemoryStorage();
    const calls: FetchCall[] = [];
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const slowFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(input), body: JSON.parse(String(init?.body)) });
      await gate;
      return Response.json({ ok: true });
    }) as typeof fetch;
    const first = processEvent(storage, ev(6), cfg, NOW, slowFetch);
    await new Promise((r) => setTimeout(r, 0));
    const second = await processEvent(storage, ev(7, 1000), cfg, NOW + 1000, slowFetch);
    release();
    expect((await first).alert.status).toBe("sent");
    expect(second.alert.status).toBe("suppressed_active");
    expect(calls).toHaveLength(1);
  });

  it("Telegram API failure leaves alert un-sent so the next update retries", async () => {
    const storage = new MemoryStorage();
    const f = telegramFetch([Response.json({ ok: false, description: "Bad Request: chat not found" }, { status: 400 })]);
    const r1 = await processEvent(storage, ev(6), cfg, NOW, f);
    expect(r1.alert).toEqual({ status: "failed", detail: "HTTP 400: Bad Request: chat not found" });
    const r2 = await processEvent(storage, ev(6, 5000), cfg, NOW + 5000, f);
    expect(r2.alert.status).toBe("sent"); // failure did not start a cooldown
  });

  it("Telegram 429 blocks further attempts for retry_after seconds", async () => {
    const storage = new MemoryStorage();
    const calls: FetchCall[] = [];
    const f = telegramFetch(
      [Response.json({ ok: false, description: "Too Many Requests", parameters: { retry_after: 20 } }, { status: 429 })],
      calls,
    );
    expect((await processEvent(storage, ev(6), cfg, NOW, f)).alert.status).toBe("rate_limited");
    expect((await processEvent(storage, ev(6, 10_000), cfg, NOW + 10_000, f)).alert.status).toBe("rate_limited");
    expect(calls).toHaveLength(1); // second call did not hit Telegram
    expect((await processEvent(storage, ev(6, 21_000), cfg, NOW + 21_000, f)).alert.status).toBe("sent");
  });

  it("network failure and timeout are reported without leaking the token", async () => {
    const timeout = Object.assign(new Error("timed out"), { name: "TimeoutError" });
    for (const err of [new TypeError("fetch failed"), timeout]) {
      const r = await processEvent(new MemoryStorage(), ev(6), cfg, NOW, telegramFetch([err]));
      expect(r.alert.status).toBe("failed");
      expect(JSON.stringify(r)).not.toContain(BOT_TOKEN);
    }
  });

  it("reports not_configured without Telegram secrets and does not mark alert sent", async () => {
    const storage = new MemoryStorage();
    const r = await processEvent(storage, ev(6), { ...cfg, telegramBotToken: "" }, NOW, telegramFetch([]));
    expect(r.alert.status).toBe("not_configured");
    expect((await storage.get<any>("state")).alertActive).toBe(false);
  });

  it("ignores stale and out-of-order events for alerting", async () => {
    const storage = new MemoryStorage();
    const calls: FetchCall[] = [];
    const f = telegramFetch([], calls);
    expect((await processEvent(storage, ev(9, -400_000), cfg, NOW, f)).alert.status).toBe("skipped_stale");
    await processEvent(storage, ev(1, 0), cfg, NOW, f);
    expect((await processEvent(storage, ev(9, -10_000), cfg, NOW, f)).alert.status).toBe("skipped_stale");
    expect(calls).toHaveLength(0);
  });

  it("keeps the de-dup list bounded", async () => {
    const storage = new MemoryStorage();
    for (let i = 0; i < SEEN_EVENT_IDS_LIMIT + 20; i++) await processEvent(storage, ev(0, i), cfg, NOW + i, telegramFetch([]));
    expect((await storage.get<string[]>("seen"))!.length).toBe(SEEN_EVENT_IDS_LIMIT);
  });
});

describe("telegram message", () => {
  it("includes camera, event type, count and timestamp and marks test input", () => {
    const text = formatAlert(makeEvent({ people_count: 6, tracked_person_ids: [], input_kind: "file" }), 5);
    expect(text).toContain("Camera: camera-01");
    expect(text).toContain("Event: people_threshold_alert");
    expect(text).toContain("People in view: 6 (threshold 5)");
    expect(text).toContain("Time (UTC): 2026-10-17T10:00:00.000Z");
    expect(text).toContain("TEST INPUT");
  });

  it("posts plain text to the bot API", async () => {
    const calls: FetchCall[] = [];
    expect(await sendTelegram(BOT_TOKEN, "-1001", "hi", telegramFetch([], calls))).toEqual({ ok: true });
    expect(calls[0]!.url).toBe(`https://api.telegram.org/bot${BOT_TOKEN}/sendMessage`);
    expect(calls[0]!.body).toEqual({ chat_id: "-1001", text: "hi", disable_web_page_preview: true });
  });
});
