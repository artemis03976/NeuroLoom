import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // `npm run coverage`: line coverage of the core library (PRD 14 asks ≥ 80%).
    coverage: {
      provider: 'v8',
      include: ['src/**'],
      reporter: ['text-summary', 'text'],
      thresholds: { lines: 80 },
    },
  },
});
