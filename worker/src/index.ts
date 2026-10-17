import { DurableObject } from "cloudflare:workers";

import { processEvent, type StateStorage } from "./alerts";
import { handleRequest, readAlertConfig } from "./handler";
import type { CountEvent, Env, ProcessResult } from "./types";

/** One instance per camera_id; serialises that camera's events and persists alert state. */
export class CameraAlertState extends DurableObject<Env> {
  async processEvent(event: CountEvent): Promise<ProcessResult> {
    const storage: StateStorage = {
      get: (key) => this.ctx.storage.get(key),
      put: (key, value) => this.ctx.storage.put(key, value),
    };
    return processEvent(storage, event, readAlertConfig(this.env), Date.now());
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return handleRequest(request, env);
  },
} satisfies ExportedHandler<Env>;
