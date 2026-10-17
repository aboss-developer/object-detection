/**
 * Per-camera alert state machine. Pure logic over a tiny storage interface so it can
 * run inside the Durable Object (production) or against an in-memory map (tests).
 *
 * Rules (documented in README "Alert rules"):
 *  - count >= threshold and not active      -> send (unless within cooldown / Telegram back-off)
 *  - count >= threshold and already active   -> suppressed_active (no repeat alerts)
 *  - count <  threshold while active          -> rearm (next crossing can alert again)
 *  - a new alert also requires cooldown since the last *sent* alert
 *  - Telegram failure leaves the alert un-sent, so the next update retries it
 *  - duplicate event_id (edge retry) -> no-op; old/out-of-order events never drive alerts
 */
import { formatAlert, sendTelegram } from "./telegram";
import type { AlertConfig, AlertStatus, CountEvent, ProcessResult } from "./types";
import { parseUtc } from "./validate";

export interface StateStorage {
  get<T>(key: string): Promise<T | undefined>;
  put<T>(key: string, value: T): Promise<void>;
}

export interface CameraState {
  alertActive: boolean;
  lastAlertAtMs: number | null;
  lastEventTsMs: number | null;
  telegramBlockedUntilMs: number | null;
}

export const SEEN_EVENT_IDS_LIMIT = 64;
const INITIAL: CameraState = { alertActive: false, lastAlertAtMs: null, lastEventTsMs: null, telegramBlockedUntilMs: null };

export type Decision = { action: "send" | "none"; status: AlertStatus; next: CameraState };

export function decide(state: CameraState, count: number, nowMs: number, cfg: AlertConfig): Decision {
  if (count < cfg.threshold) {
    return state.alertActive
      ? { action: "none", status: "rearmed", next: { ...state, alertActive: false } }
      : { action: "none", status: "not_triggered", next: state };
  }
  if (state.alertActive) return { action: "none", status: "suppressed_active", next: state };
  if (state.telegramBlockedUntilMs !== null && nowMs < state.telegramBlockedUntilMs) {
    return { action: "none", status: "rate_limited", next: state };
  }
  if (state.lastAlertAtMs !== null && nowMs - state.lastAlertAtMs < cfg.cooldownMs) {
    return { action: "none", status: "suppressed_cooldown", next: state };
  }
  return { action: "send", status: "sent", next: { ...state, alertActive: true, lastAlertAtMs: nowMs } };
}

export async function processEvent(
  storage: StateStorage,
  event: CountEvent,
  cfg: AlertConfig,
  nowMs: number,
  fetchFn: typeof fetch = fetch,
): Promise<ProcessResult> {
  const result = (status: AlertStatus, detail?: string, duplicate = false): ProcessResult => ({
    duplicate,
    alert: detail ? { status, detail } : { status },
    people_count: event.people_count,
    threshold: cfg.threshold,
  });

  const seen = (await storage.get<string[]>("seen")) ?? [];
  if (seen.includes(event.event_id)) return result("duplicate", undefined, true);
  await storage.put("seen", [...seen, event.event_id].slice(-SEEN_EVENT_IDS_LIMIT));

  const state = (await storage.get<CameraState>("state")) ?? INITIAL;
  const tsMs = parseUtc(event.timestamp)!;
  if (nowMs - tsMs > cfg.maxEventAgeMs || (state.lastEventTsMs !== null && tsMs < state.lastEventTsMs)) {
    return result("skipped_stale");
  }

  const decision = decide({ ...state, lastEventTsMs: tsMs }, event.people_count, nowMs, cfg);
  if (decision.action === "none") {
    await storage.put("state", decision.next);
    return result(decision.status);
  }
  if (!cfg.telegramBotToken || !cfg.telegramChatId) {
    await storage.put("state", { ...decision.next, alertActive: false, lastAlertAtMs: state.lastAlertAtMs });
    return result("not_configured", "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set");
  }

  // Claim the alert *before* the network call: requests that interleave while we await
  // Telegram see alertActive=true and cannot send a duplicate.
  await storage.put("state", decision.next);
  const sent = await sendTelegram(cfg.telegramBotToken, cfg.telegramChatId, formatAlert(event, cfg.threshold), fetchFn);
  if (sent.ok) return result("sent");

  const current = (await storage.get<CameraState>("state")) ?? decision.next;
  const reverted: CameraState = { ...current, alertActive: false, lastAlertAtMs: state.lastAlertAtMs };
  if (sent.kind === "rate_limited") {
    reverted.telegramBlockedUntilMs = nowMs + Math.min(sent.retryAfterSeconds, 3600) * 1000;
    await storage.put("state", reverted);
    return result("rate_limited", `retry after ${sent.retryAfterSeconds}s`);
  }
  await storage.put("state", reverted);
  return result("failed", sent.detail);
}
