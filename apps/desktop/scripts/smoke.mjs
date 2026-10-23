// End-to-end smoke test of the built desktop app: launches Electron, waits for
// the editor, talks to the Python backend over IPC and saves a screenshot.
// Run `npm run build -w @neuroloom/desktop` first.
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron } from 'playwright-core';

const desktopRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const userData = mkdtempSync(join(tmpdir(), 'neuroloom-smoke-'));
const screenshot = process.argv[2] ?? join(desktopRoot, 'dist', 'smoke.png');

const check = (label, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? `  (${detail})` : ''}`);
  if (!ok) process.exitCode = 1;
};

const app = await _electron.launch({
  args: ['.'],
  cwd: desktopRoot,
  env: { ...process.env, NEUROLOOM_USER_DATA: userData },
});
try {
  const win = await app.firstWindow();
  await win.waitForSelector('.react-flow__node', { timeout: 30_000 });
  check('editor renders nodes', (await win.locator('.react-flow__node').count()) >= 5);

  const isolated = await win.evaluate(() => typeof window.require === 'undefined' && typeof window.process === 'undefined');
  check('renderer has no Node.js access', isolated);

  const ping = await win.evaluate(() => window.neuroloom.rpc('ping'));
  check('Python backend answers over IPC', ping?.pong === true);

  const env = await win.evaluate(() => window.neuroloom.rpc('env_info'));
  check('backend has torch', typeof env?.torch === 'string', `torch ${env?.torch}`);

  const failure = await win.evaluate(() => window.neuroloom.rpc('no_such_method').then(() => null, (e) => e));
  check('RPC errors arrive as objects', failure?.code === -32601, failure?.message);

  const file = join(userData, 'smoke.loom');
  const saved = await win.evaluate(
    (path) => window.neuroloom.file.save({ path, text: '{"hello":"世界"}\n', suggestedName: 'smoke' }),
    file,
  );
  const reopened = await win.evaluate((path) => window.neuroloom.file.openPath(path), file);
  check('files save and reopen over IPC', saved === file && reopened.text === '{"hello":"世界"}\n');
  check('saved file is listed as recent', (await win.evaluate(() => window.neuroloom.file.recent())).includes(file));

  const refused = await win.evaluate(() => window.neuroloom.file.openPath('C:/Windows/win.ini').then(() => null, (e) => e.message));
  check('only project files can be opened by path', typeof refused === 'string' && refused.includes('.loom'), refused);

  await win.evaluate(() => window.neuroloom.recovery.write({ text: 'x', savedAt: 1 }));
  const recovered = await win.evaluate(() => window.neuroloom.recovery.read());
  await win.evaluate(() => window.neuroloom.recovery.clear());
  const cleared = await win.evaluate(() => window.neuroloom.recovery.read());
  check('recovery file round-trips and clears', recovered?.text === 'x' && cleared === null);

  // Commands end to end: open from the menu, edit, Ctrl+S back to the same file, undo.
  const model = join(userData, 'smoke-model.loom');
  const project = {
    schemaVersion: '0.2',
    name: 'Tiny',
    entry: 'main',
    modules: {
      main: {
        id: 'main',
        className: 'Tiny',
        nodes: {
          x: { id: 'x', kind: 'input', type: 'neuroloom.Input', name: 'x', params: { shape: { literal: 'B, 8' } } },
          fc: { id: 'fc', kind: 'layer', type: 'torch.nn.Linear', name: 'fc', params: { in_features: { auto: true }, out_features: { literal: 4 } } },
          out: { id: 'out', kind: 'output', type: 'neuroloom.Output', name: 'output', params: {} },
        },
        edges: [
          { id: 'e1', source: { node: 'x', port: 'out' }, target: { node: 'fc', port: 'in' } },
          { id: 'e2', source: { node: 'fc', port: 'out' }, target: { node: 'out', port: 'in' } },
        ],
        layout: { x: { x: 0, y: 0 }, fc: { x: 260, y: 0 }, out: { x: 520, y: 0 } },
        interface: { inputs: ['x'], outputs: ['out'] },
      },
    },
  };
  writeFileSync(model, JSON.stringify(project));
  const windowTitle = () => app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].getTitle());
  await app.evaluate(({ BrowserWindow }, path) => BrowserWindow.getAllWindows()[0].webContents.send('nl:command', `openRecent:${path}`), model);
  await win.waitForFunction(() => document.querySelectorAll('.react-flow__node').length === 3);
  check('menu command opens a project', (await windowTitle()) === 'smoke-model — NeuroLoom', await windowTitle());

  await win.locator('.react-flow__node', { hasText: 'fc' }).click();
  await win.keyboard.press('Delete');
  await win.waitForFunction(() => document.querySelectorAll('.react-flow__node').length === 2);
  check('editing marks the window unsaved', (await windowTitle()).startsWith('● '), await windowTitle());

  await win.keyboard.press('Control+S');
  await win.waitForFunction(() => !document.title.startsWith('x'), null, { timeout: 2000 }).catch(() => {});
  await win.waitForTimeout(500);
  const written = JSON.parse(readFileSync(model, 'utf8'));
  check('Ctrl+S writes back to the same file', Object.keys(written.modules.main.nodes).length === 2 && !(await windowTitle()).startsWith('● '));
  check('saved file uses sorted keys', readFileSync(model, 'utf8').startsWith('{\n  "entry": "main",'));

  await win.keyboard.press('Control+Z');
  await win.waitForFunction(() => document.querySelectorAll('.react-flow__node').length === 3);
  check('Ctrl+Z restores the node and the unsaved mark', (await windowTitle()).startsWith('● '));

  // Static import reads the code without running it and reports per class.
  const imported = await win.evaluate(() =>
    window.neuroloom.rpc('import_code', {
      source: 'from torch import nn\nclass M(nn.Module):\n    def __init__(self, d=4):\n        super().__init__()\n        self.fc = nn.Linear(d, 2)\n    def forward(self, x):\n        return self.fc(x)\n',
      inputShapes: { x: 'B,4' },
      mode: 'static',
    }),
  );
  check('static import over IPC', imported?.report?.[0]?.level === 'L1', JSON.stringify(imported?.report));

  // Modules: double-click drills in, Alt+← comes back (the dirty document is discarded).
  win.on('dialog', (d) => void d.accept());
  const nested = join(userData, 'smoke-nested.loom');
  const io = (shape) => ({
    x: { id: 'x', kind: 'input', type: 'neuroloom.Input', name: 'x', params: { shape: { literal: shape } } },
    out: { id: 'out', kind: 'output', type: 'neuroloom.Output', name: 'output', params: {} },
  });
  // A module node's ports are the ids of its definition's input and output nodes ("x", "out").
  const wire = (inner) => [
    { id: 'e1', source: { node: 'x', port: 'out' }, target: { node: inner, port: inner === 'blk' ? 'x' : 'in' } },
    { id: 'e2', source: { node: inner, port: 'out' }, target: { node: 'out', port: 'in' } },
  ];
  const place = (inner) => ({ x: { x: 0, y: 0 }, [inner]: { x: 260, y: 0 }, out: { x: 520, y: 0 } });
  writeFileSync(
    nested,
    JSON.stringify({
      schemaVersion: '0.3',
      name: 'Nested',
      entry: 'main',
      modules: {
        Block: {
          id: 'Block',
          className: 'Block',
          hyperparams: [{ name: 'width' }],
          nodes: { ...io(''), fc: { id: 'fc', kind: 'layer', type: 'torch.nn.Linear', name: 'fc', params: { in_features: { auto: true }, out_features: { expr: 'width' } } } },
          edges: wire('fc'),
          layout: place('fc'),
          interface: { inputs: ['x'], outputs: ['out'] },
        },
        main: {
          id: 'main',
          className: 'Nested',
          hyperparams: [],
          nodes: { ...io('B, 8'), blk: { id: 'blk', kind: 'module', type: 'Block', name: 'blk', params: { width: { literal: 4 } } } },
          edges: wire('blk'),
          layout: place('blk'),
          interface: { inputs: ['x'], outputs: ['out'] },
        },
      },
    }),
  );
  await app.evaluate(({ BrowserWindow }, path) => BrowserWindow.getAllWindows()[0].webContents.send('nl:command', `openRecent:${path}`), nested);
  const hasNode = (name) => [...document.querySelectorAll('.react-flow__node .nl-node-name')].some((n) => n.textContent === name);
  await win.waitForFunction(hasNode, 'blk');
  check('opens a 0.3 project with a module', (await windowTitle()) === 'smoke-nested — NeuroLoom', await windowTitle());
  await win.locator('.react-flow__node', { hasText: 'blk' }).dblclick();
  await win.waitForFunction(hasNode, 'fc');
  const crumb = await win.locator('nav').first().textContent();
  check('double-click opens a module, with a breadcrumb', crumb?.includes('Nested') && crumb.includes('blk'), crumb ?? '');
  await win.keyboard.press('Alt+ArrowLeft');
  await win.waitForFunction(hasNode, 'blk');
  check('Alt+← goes back up', true);

  // Native menus follow the interface language.
  const menuLabels = () => app.evaluate(({ Menu }) => Menu.getApplicationMenu()?.items.map((i) => i.label) ?? []);
  const send = (command) => app.evaluate(({ BrowserWindow }, c) => BrowserWindow.getAllWindows()[0].webContents.send('nl:command', c), command);
  await send('locale:en');
  await win.waitForTimeout(300);
  const english = await menuLabels();
  await send('locale:zh');
  await win.waitForTimeout(300);
  const chinese = await menuLabels();
  check('menus switch language with the editor', english.includes('Module') && chinese.includes('模块'), `${english.join('/')} | ${chinese.join('/')}`);

  const candidates = await win.evaluate(() => window.neuroloom.env.list());
  check('environment scan finds a ready interpreter', candidates.some((c) => c.status === 'ready'), `${candidates.length} found`);

  await win.locator('header button', { hasText: 'torch' }).click();
  await win.waitForSelector('#env-title');
  await win.waitForSelector('[role=dialog] li', { timeout: 60_000 });
  await win.waitForTimeout(300);
  await win.screenshot({ path: screenshot });
  console.log(`screenshot: ${screenshot}`);
} finally {
  // destroy() skips the unsaved-changes prompt, which would otherwise block a dirty window.
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().forEach((w) => w.destroy())).catch(() => {});
  await app.close();
  rmSync(userData, { recursive: true, force: true });
}
