import { defineConfig } from "vitest/config";

// Unit tests run in Node against the pure modules (handler/alerts/telegram/validate);
// the Durable Object wrapper in src/index.ts is exercised via `wrangler dev`.
export default defineConfig({
  test: { include: ["test/**/*.test.ts"], environment: "node" },
});
