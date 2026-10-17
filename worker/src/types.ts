export interface CountEvent {
  schema_version: 1;
  event_id: string;
  sequence: number;
  camera_id: string;
  event_type: "people_count_update";
  timestamp: string;
  people_count: number;
  tracked_person_ids: number[];
  input_kind: "rtsp" | "file";
  source: "alpha-vision-edge";
}

export type AlertStatus =
  | "not_triggered" // below threshold
  | "rearmed" // dropped below threshold after an alert; next crossing may alert again
  | "sent"
  | "suppressed_active" // still above threshold, already alerted for this episode
  | "suppressed_cooldown" // crossed again but within the cooldown window
  | "rate_limited" // Telegram asked us to back off (429 retry_after)
  | "failed" // Telegram/network error; will retry on the next update while above threshold
  | "not_configured" // Telegram secrets missing
  | "skipped_stale" // event too old / out of order: not used for alerting
  | "duplicate"; // event_id already processed (edge retry)

export interface ProcessResult {
  duplicate: boolean;
  alert: { status: AlertStatus; detail?: string };
  people_count: number;
  threshold: number;
}

export interface AlertConfig {
  threshold: number;
  cooldownMs: number;
  maxEventAgeMs: number;
  telegramBotToken: string;
  telegramChatId: string;
}

export interface Env {
  CAMERA_STATE: {
    getByName(name: string): { processEvent(event: CountEvent): Promise<ProcessResult> };
  };
  INGEST_TOKEN?: string;
  TELEGRAM_BOT_TOKEN?: string;
  TELEGRAM_CHAT_ID?: string;
  PERSON_ALERT_THRESHOLD?: string;
  ALERT_COOLDOWN_SECONDS?: string;
  MAX_EVENT_AGE_SECONDS?: string;
  ALLOWED_CAMERA_IDS?: string;
}
