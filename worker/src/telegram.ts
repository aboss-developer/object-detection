import type { CountEvent } from "./types";

export type TelegramResult =
  | { ok: true }
  | { ok: false; kind: "rate_limited"; retryAfterSeconds: number; detail: string }
  | { ok: false; kind: "api_error" | "network"; detail: string };

export const TELEGRAM_TIMEOUT_MS = 8000;

export function formatAlert(event: CountEvent, threshold: number): string {
  const lines = [
    "🚨 People threshold reached",
    `Camera: ${event.camera_id}`,
    "Event: people_threshold_alert",
    `People in view: ${event.people_count} (threshold ${threshold})`,
    `Time (UTC): ${event.timestamp}`,
  ];
  if (event.input_kind === "file") lines.push("⚠️ TEST INPUT: recorded video file, not a live camera");
  return lines.join("\n");
}

/** Send a plain-text message. Never includes the bot token in returned/logged details. */
export async function sendTelegram(
  botToken: string,
  chatId: string,
  text: string,
  fetchFn: typeof fetch = fetch,
): Promise<TelegramResult> {
  let resp: Response;
  try {
    resp = await fetchFn(`https://api.telegram.org/bot${botToken}/sendMessage`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ chat_id: chatId, text, disable_web_page_preview: true }),
      signal: AbortSignal.timeout(TELEGRAM_TIMEOUT_MS),
    });
  } catch (err) {
    const name = err instanceof Error ? err.name : "Error";
    return { ok: false, kind: "network", detail: name === "TimeoutError" ? "timeout" : name };
  }
  let body: { ok?: boolean; description?: string; parameters?: { retry_after?: number } } = {};
  try {
    body = await resp.json();
  } catch {
    /* non-JSON error page */
  }
  if (resp.ok && body.ok) return { ok: true };
  const description = String(body.description ?? `HTTP ${resp.status}`).slice(0, 200);
  if (resp.status === 429) {
    const retry = Number(body.parameters?.retry_after ?? resp.headers.get("retry-after") ?? 30);
    return { ok: false, kind: "rate_limited", retryAfterSeconds: Number.isFinite(retry) ? retry : 30, detail: description };
  }
  return { ok: false, kind: "api_error", detail: `HTTP ${resp.status}: ${description}` };
}
