// Cold start of the packaged app against PRD section 14: from launching the process to
// an editor that can be used (≤ 3 s on a typical machine).
//
// Each launch starts from a fresh user-data folder, as a first run after installing
// does. "Usable" means the editor has drawn the canvas with its first project and
// rendered one more frame after that; the Python backend starts alongside and is
// timed too, but the editor does not wait for it. The first launch after packaging
// is reported apart: the operating system has not read the new files yet (and on
// Windows the virus scanner checks them), which is the closest to a true cold start.
//
// Times include Playwright's own launch overhead, so they are a little pessimistic.
// Run `npm run dist -- --dir` first. Prints a report and exits 1 when a launch misses
// the budget.
// Usage: node scripts/startup.mjs [--runs N] [path/to/executable]
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { _electron } from 'playwright-core';
import { packagedExecutable } from './packaged.mjs';

const BUDGET_MS = 3000;
const args = process.argv.slice(2);
const runsAt = args.indexOf('--runs');
const runs = runsAt >= 0 ? Number(args.splice(runsAt, 2)[1]) : 5;
const executablePath = args[0] ?? packagedExecutable();
if (!executablePath || !existsSync(executablePath)) {
  console.error('No packaged app found; run `npm run dist -- --dir` first or pass the executable path.');
  process.exit(1);
}
console.log(`app: ${executablePath}`);

/** One launch: when the window appeared, when the editor became usable, when Python answered. */
async function launch() {
  const userData = mkdtempSync(join(tmpdir(), 'neuroloom-startup-'));
  const start = performance.now();
  // Playwright adds --no-sandbox on Linux unless told otherwise; start as users do.
  const app = await _electron.launch({ executablePath, args: ['--lang=en-US'], env: { ...process.env, NEUROLOOM_USER_DATA: userData }, chromiumSandbox: true });
  try {
    const win = await app.firstWindow();
    const window = performance.now() - start;
    // Asked from the window's first moment, alongside the editor's own start.
    const python = (async () => {
      while (performance.now() - start < 30_000) {
        const ping = await win.evaluate(() => window.neuroloom.rpc('ping')).catch(() => null);
        if (ping?.pong) return performance.now() - start;
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
      return undefined;
    })();
    await win.waitForSelector('.react-flow__node', { timeout: 30_000 });
    await win.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const usable = performance.now() - start;
    return { window, usable, python: await python };
  } finally {
    await app.close();
    rmSync(userData, { recursive: true, force: true });
  }
}

const results = [];
for (let i = 0; i < runs; i++) results.push(await launch());

const ms = (v) => (v === undefined ? '—' : `${Math.round(v)} ms`);
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
results.forEach((r, i) => console.log(`launch ${i + 1}${i === 0 ? ' (first)' : '        '}  window ${ms(r.window).padStart(8)}  usable ${ms(r.usable).padStart(8)}  python ${ms(r.python).padStart(8)}`));
const rest = results.slice(1);
const worst = Math.max(...results.map((r) => r.usable));
const ok = worst <= BUDGET_MS;
console.log('');
console.log(`${ok ? 'PASS' : 'FAIL'}  start to usable editor: first ${ms(results[0].usable)}${rest.length ? `, then median ${ms(median(rest.map((r) => r.usable)))}` : ''}, worst ${ms(worst)}  (budget ≤ ${BUDGET_MS} ms)`);
const pythonTimes = results.map((r) => r.python).filter((v) => v !== undefined);
console.log(`info  Python backend answering: median ${ms(pythonTimes.length ? median(pythonTimes) : undefined)} (the editor does not wait for it)`);
if (!ok) process.exitCode = 1;
