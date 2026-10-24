// Performance of the built desktop app against PRD section 13, in a real window:
//   - canvas: pan at 1000 nodes / 1333 edges (≥ 50 fps), and a 200-node view with
//     edge animation on (≥ 60 fps);
//   - an edit at 500 nodes (shape update ≤ 100 ms) and at 1000 nodes;
//   - renderer memory with the 1000-node project open (≤ 500 MB).
// Run `npm run build:desktop` first; prints a report and exits 1 when a budget is
// missed (frame rates depend on the machine: run it on a typical one).
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron } from 'playwright-core';

const desktopRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const userData = mkdtempSync(join(tmpdir(), 'neuroloom-perf-'));

const rows = [];
const report = (label, value, unit, budget, better = 'lower') => {
  const ok = budget === undefined || (better === 'lower' ? value <= budget : value >= budget);
  rows.push(`${ok ? 'PASS' : 'FAIL'}  ${label.padEnd(44)} ${String(value).padStart(7)} ${unit}${budget === undefined ? '' : `  (budget ${better === 'lower' ? '≤' : '≥'} ${budget})`}`);
  if (!ok) process.exitCode = 1;
};

// Chromium stops drawing windows it thinks are covered, and paces frames by the
// display; measure an unthrottled window with frames unpaced, so the rates are what
// the app can draw (also with the screen locked or off).
const app = await _electron.launch({
  args: [
    '.',
    '--lang=en-US',
    '--disable-features=CalculateNativeWinOcclusion',
    '--disable-renderer-backgrounding',
    '--disable-background-timer-throttling',
    '--disable-gpu-vsync',
    '--disable-frame-rate-limit',
    '--js-flags=--expose-gc',
  ],
  cwd: desktopRoot,
  env: { ...process.env, NEUROLOOM_USER_DATA: userData },
});
const step = (text) => console.log(`… ${text}`);
try {
  const win = await app.firstWindow();
  await app.evaluate(({ BrowserWindow }) => {
    const w = BrowserWindow.getAllWindows()[0];
    w.setAlwaysOnTop(true);
    w.webContents.setBackgroundThrottling(false);
    w.setSize(1440, 960);
    w.center();
    w.show();
    w.focus();
  });
  // ?bench puts the store and the React Flow instance on window (apps/studio/src/lib/bench.ts).
  await win.waitForSelector('.react-flow__node', { timeout: 30_000 });
  await win.goto(`${win.url().split(/[?#]/)[0]}?bench`);
  await win.waitForFunction(() => !!window.__neuroloom && !!window.__neuroloomFlow, null, { timeout: 30_000 });

  /** Loads a residual chain, waits until it is laid out, measured and settled. */
  const load = (blocks) =>
    win.evaluate(async (blocks) => {
      const nl = window.__neuroloom;
      nl.useStudio.getState().loadProject(nl.residualChain(blocks), { name: 'Deep', relayout: false });
      const nodes = 3 * blocks + 2;
      for (let i = 0; i < 100 && document.querySelectorAll('.react-flow__node').length < Math.min(nodes, 1); i++) await new Promise((r) => setTimeout(r, 100));
      await new Promise((r) => setTimeout(r, 2500));
    }, blocks);

  /** Median time from an edit to the frame that shows it. */
  const edit = () =>
    win.evaluate(async () => {
      const nl = window.__neuroloom;
      const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const times = [];
      for (let i = 0; i < 9; i++) {
        const start = performance.now();
        nl.useStudio.getState().setParam(`fc${20 + i}`, 'bias', { literal: i % 2 === 0 });
        await frame();
        times.push(performance.now() - start);
      }
      return Math.round(times.sort((a, b) => a - b)[4]);
    });

  /**
   * Frames per second while dragging the canvas at `zoom`, as a user pans: the graph
   * centred, then dragged back and forth by a quarter of the view each way.
   */
  const pan = async (zoom) => {
    await win.evaluate((zoom) => {
      const flow = window.__neuroloomFlow;
      const box = document.querySelector('.react-flow').getBoundingClientRect();
      const nodes = flow.getNodes();
      const xs = nodes.map((n) => n.position.x);
      const ys = nodes.map((n) => n.position.y);
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
      const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
      flow.setViewport({ x: box.width / 2 - cx * zoom, y: box.height / 2 - cy * zoom, zoom });
    }, zoom);
    await win.waitForTimeout(1000);
    const box = await win.locator('.react-flow__pane').boundingBox();
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    await win.evaluate(() => {
      window.__frames = 0;
      window.__counting = true;
      const tick = () => {
        window.__frames++;
        if (window.__counting) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    const started = Date.now();
    await win.mouse.move(cx, cy);
    await win.mouse.down();
    for (let i = 1; Date.now() - started < 3000; i++) {
      const a = (i / 200) * 2 * Math.PI;
      await win.mouse.move(cx + Math.sin(a) * box.width * 0.25, cy + Math.sin(2 * a) * box.height * 0.25);
    }
    await win.mouse.up();
    const frames = await win.evaluate(() => {
      window.__counting = false;
      return window.__frames;
    });
    return Math.round(frames / ((Date.now() - started) / 1000));
  };

  /** Frames per second with nothing moving but the edge animation (particles, the most expensive kind). */
  const animate = () =>
    win.evaluate(async () => {
      window.__neuroloomFlow.fitView();
      await new Promise((r) => setTimeout(r, 500));
      window.__neuroloom.useFlow.getState().set({ mode: 'particles' });
      await new Promise((r) => setTimeout(r, 1000));
      let frames = 0;
      const start = performance.now();
      await new Promise((resolve) => {
        const step = (now) => {
          frames++;
          if (now - start < 3000) requestAnimationFrame(step);
          else resolve();
        };
        requestAnimationFrame(step);
      });
      return {
        fps: Math.round(frames / ((performance.now() - start) / 1000)),
        particles: document.querySelectorAll('.react-flow__edges circle').length,
        dashed: document.querySelectorAll('.react-flow__edges .nl-edge-dashes').length,
      };
    });

  step('200 nodes');
  await load(66); // 200 nodes on one level
  const small = await animate();
  report(`200 nodes, ${small.particles} particles moving`, small.fps, 'fps', 60, 'higher');

  step('500 nodes');
  await load(166); // 500 nodes
  report('edit at 500 nodes, to the frame showing it', await edit(), 'ms', 100);

  // Memory of the renderer process: private bytes after a garbage collection.
  const memory = async () => {
    await win.evaluate(() => window.gc?.());
    const metrics = await app.evaluate(({ app }) => app.getAppMetrics());
    const tab = metrics.filter((m) => m.type === 'Tab');
    return Math.round(tab.reduce((n, m) => n + (m.memory.privateBytes ?? m.memory.workingSetSize), 0) / 1024);
  };

  step('1000 nodes');
  await win.evaluate(() => window.__neuroloom.useStudio.getState().loadProject(window.__neuroloom.residualChain(1), { name: 'Small' }));
  await load(333); // 1001 nodes, 1333 edges
  report('renderer memory, 1000-node project opened', await memory(), 'MB', 500);
  report('edit at 1000 nodes, to the frame showing it', await edit(), 'ms');
  report('pan at 1000 nodes, zoom 0.8 (culled)', await pan(0.8), 'fps', 50, 'higher');
  report('pan at 1000 nodes, zoom 0.25 (simplified)', await pan(0.25), 'fps', 50, 'higher');
  report('pan at 1000 nodes, zoom 0.1 (all in view)', await pan(0.1), 'fps', 50, 'higher');

  report('renderer memory, after editing and panning it', await memory(), 'MB', 500);
} finally {
  await app.close();
  rmSync(userData, { recursive: true, force: true });
}
console.log(rows.join('\n'));
