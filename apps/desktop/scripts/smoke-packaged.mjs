// Smoke test of the packaged app (the unpacked build that `npm run dist` leaves in
// release/): it starts, serves the editor from its own resources, runs the bundled
// Python package with an interpreter found on this machine, and opens .loom files
// passed on the command line, also when handed over by a second launch.
// Usage: node scripts/smoke-packaged.mjs [path/to/executable]
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron } from 'playwright-core';

const desktopRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const release = join(desktopRoot, 'release');

function findExecutable() {
  const candidates = {
    win32: [join(release, 'win-unpacked', 'NeuroLoom.exe')],
    darwin: [process.arch === 'arm64' ? 'mac-arm64' : 'mac', 'mac-arm64', 'mac'].map((d) => join(release, d, 'NeuroLoom.app', 'Contents', 'MacOS', 'NeuroLoom')),
    linux: [join(release, 'linux-unpacked', 'neuroloom')],
  }[process.platform] ?? [];
  return candidates.find((p) => existsSync(p));
}

const executablePath = process.argv[2] ?? findExecutable();
if (!executablePath || !existsSync(executablePath)) {
  console.error('No packaged app found; run `npm run dist -- --dir` first or pass the executable path.');
  process.exit(1);
}
console.log(`app: ${executablePath}`);

const check = (label, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? `  (${detail})` : ''}`);
  if (!ok) process.exitCode = 1;
};

const userData = mkdtempSync(join(tmpdir(), 'neuroloom-packaged-'));
const env = { ...process.env, NEUROLOOM_USER_DATA: userData };
const project = (name, nodes) => {
  const path = join(userData, `${name}.loom`);
  const io = {
    x: { id: 'x', kind: 'input', type: 'neuroloom.Input', name: 'x', params: { shape: { literal: 'B, 8' } } },
    out: { id: 'out', kind: 'output', type: 'neuroloom.Output', name: 'output', params: {} },
  };
  const layers = Object.fromEntries(
    nodes.map((id) => [id, { id, kind: 'layer', type: 'torch.nn.Linear', name: id, params: { in_features: { auto: true }, out_features: { literal: 8 } } }]),
  );
  const chain = ['x', ...nodes, 'out'];
  writeFileSync(
    path,
    JSON.stringify({
      schemaVersion: '0.3',
      name,
      entry: 'main',
      modules: {
        main: {
          id: 'main',
          className: name,
          hyperparams: [],
          nodes: { ...io, ...layers },
          edges: chain.slice(1).map((id, i) => ({ id: `e${i}`, source: { node: chain[i], port: 'out' }, target: { node: id, port: 'in' } })),
          layout: Object.fromEntries(chain.map((id, i) => [id, { x: i * 260, y: 0 }])),
          interface: { inputs: ['x'], outputs: ['out'] },
        },
      },
    }),
  );
  return path;
};
const first = project('first', ['fc']);
const second = project('second', ['fc1', 'fc2']);

const app = await _electron.launch({ executablePath, args: [first, '--lang=en-US'], env });
try {
  const win = await app.firstWindow();
  const windowTitle = () => app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.getTitle());
  const waitForTitle = async (title) => {
    const deadline = Date.now() + 20_000;
    while ((await windowTitle()) !== title && Date.now() < deadline) await win.waitForTimeout(200);
    return windowTitle();
  };

  const packaged = await app.evaluate(({ app }) => ({ isPackaged: app.isPackaged, resources: process.resourcesPath, name: app.getName() }));
  check('runs as a packaged app named NeuroLoom', packaged.isPackaged && packaged.name === 'NeuroLoom', JSON.stringify(packaged));

  await win.waitForSelector('.react-flow__node', { timeout: 30_000 });
  check('editor renders from the packaged resources', win.url().startsWith('app://studio/'), win.url());

  const title = await waitForTitle('first — NeuroLoom');
  const nodes = await win.locator('.react-flow__node').count();
  check('opens the .loom file given on the command line', title === 'first — NeuroLoom' && nodes === 3, `${title}, ${nodes} nodes`);

  const ping = await win.evaluate(() => window.neuroloom.rpc('ping')).catch((e) => ({ error: e?.message ?? String(e) }));
  const python = await win.evaluate(() => window.neuroloom.env.current()).catch(() => '?');
  check('bundled Python package answers (needs Python 3.10+ on this machine)', ping?.pong === true, `${python}: ${JSON.stringify(ping)}`);
  const info = await win.evaluate(() => window.neuroloom.rpc('env_info')).catch(() => null);
  console.log(`info  torch in that interpreter: ${info?.torch ?? 'not installed'}`);

  // A second launch with a file hands it to the running app and exits.
  const exitCode = await new Promise((resolve) => {
    const child = spawn(executablePath, [second], { env, stdio: 'ignore' });
    const timer = setTimeout(() => {
      child.kill();
      resolve('timeout');
    }, 30_000);
    child.on('exit', (code) => {
      clearTimeout(timer);
      resolve(code);
    });
  });
  const handed = await waitForTitle('second — NeuroLoom');
  const windows = await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().length);
  check('a second launch hands its file to the running app', exitCode === 0 && handed === 'second — NeuroLoom' && windows === 1, `exit ${exitCode}, ${handed}, ${windows} window(s)`);

  const screenshot = join(desktopRoot, 'dist', 'smoke-packaged.png');
  await win.screenshot({ path: screenshot });
  console.log(`screenshot: ${screenshot}`);
} finally {
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().forEach((w) => w.destroy())).catch(() => {});
  await app.close();
  rmSync(userData, { recursive: true, force: true });
}
