import { processEvent, type StateStorage } from "../src/alerts";
import { readAlertConfig } from "../src/handler";
import type { CountEvent, Env } from "../src/types";

export const TOKEN = "x".repeat(48);
export const NOW = Date.parse("2026-10-17T10:00:00.000Z");
export const BOT_TOKEN = "123456:TEST-bot-token-not-real";

export class MemoryStorage implements StateStorage {
  data = new Map<string, unknown>();
  async get<T>(key: string) {
    return structuredClone(this.data.get(key)) as T | undefined;
  }
  async put<T>(key: string, value: T) {
    this.data.set(key, structuredClone(value));
  }
}

let counter = 0;
export function makeEvent(overrides: Partial<CountEvent> & Record<string, unknown> = {}): CountEvent {
  counter += 1;
  return {
    schema_version: 1,
    event_id: counter.toString(16).padStart(32, "0"),
    sequence: counter,
    camera_id: "camera-01",
    event_type: "people_count_update",
    timestamp: new Date(NOW).toISOString(),
    people_count: 2,
    tracked_person_ids: [1, 2],
    input_kind: "rtsp",
    source: "alpha-vision-edge",
    ...overrides,
  } as CountEvent;
}

export type FetchCall = { url: string; body: any };

export function telegramFetch(responses: Array<Response | Error>, calls: FetchCall[] = []): typeof fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), body: JSON.parse(String(init?.body)) });
    const next = responses.length ? responses.shift()! : Response.json({ ok: true });
    if (next instanceof Error) throw next;
    return next;
  }) as typeof fetch;
}

/** Env whose CAMERA_STATE namespace runs the real alert logic against in-memory storage per camera. */
export function makeEnv(overrides: Partial<Env> = {}, fetchFn: typeof fetch = telegramFetch([])) {
  const stores = new Map<string, MemoryStorage>();
  const env: Env = {
    INGEST_TOKEN: TOKEN,
    TELEGRAM_BOT_TOKEN: BOT_TOKEN,
    TELEGRAM_CHAT_ID: "-1001",
    PERSON_ALERT_THRESHOLD: "5",
    ALERT_COOLDOWN_SECONDS: "600",
    MAX_EVENT_AGE_SECONDS: "300",
    ALLOWED_CAMERA_IDS: "camera-01",
    ...overrides,
    CAMERA_STATE: {
      getByName(name: string) {
        if (!stores.has(name)) stores.set(name, new MemoryStorage());
        const storage = stores.get(name)!;
        return { processEvent: (event: CountEvent) => processEvent(storage, event, readAlertConfig(env), NOW, fetchFn) };
      },
    },
  };
  return { env, stores };
}

export function post(body: unknown, headers: Record<string, string> = {}): Request {
  return new Request("https://worker.example/v1/events", {
    method: "POST",
    headers: { authorization: `Bearer ${TOKEN}`, "content-type": "application/json", ...headers },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}
