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

// The app starts in English whatever the machine's language, so every run
// (a Chinese desktop, an English CI runner) takes the same path.
const app = await _electron.launch({
  args: ['.', '--lang=en-US'],
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

  // Image export (PRD F7.3) through the real dialog and IPC handler; the native
  // save dialog is stubbed in the main process to answer with a path.
  const saveTo = (path) =>
    app.evaluate(({ dialog }, target) => {
      dialog.showSaveDialog = async () => ({ canceled: false, filePath: target });
    }, path);
  const exportFromDialog = async (format, extra = '') => {
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.send('nl:command', 'exportImage'));
    await win.waitForSelector('#image-export-title');
    await win.evaluate(([f, x]) => {
      document.querySelector(`input[name=image-format][value=${f}]`).click();
      if (x) document.querySelector(x).click();
    }, [format, extra]);
    await win.locator('[role=dialog] button[type=submit]').click();
    await win.waitForSelector('#image-export-title', { state: 'detached', timeout: 15_000 });
  };
  const svgFile = join(userData, 'smoke-image.svg');
  await saveTo(svgFile);
  await exportFromDialog('svg');
  const svgText = readFileSync(svgFile, 'utf8');
  const svgDoc = await win.evaluate((text) => {
    const doc = new DOMParser().parseFromString(text, 'image/svg+xml');
    const root = doc.documentElement;
    return { root: root.nodeName, broken: !!doc.querySelector('parsererror'), width: Number(root.getAttribute('width')), height: Number(root.getAttribute('height')), texts: [...doc.querySelectorAll('text')].map((t) => t.textContent) };
  }, svgText);
  check(
    'exports the canvas as an SVG file',
    svgDoc.root === 'svg' && !svgDoc.broken && ['x', 'blk', 'output', '[B, 8]'].every((t) => svgDoc.texts.includes(t)) && !svgText.includes('var('),
    `${svgDoc.width}×${svgDoc.height}, ${svgText.length} bytes`,
  );

  const pngFile = join(userData, 'smoke-image.png');
  await saveTo(pngFile);
  await exportFromDialog('png', 'input[name=image-scale][value="2"]');
  const png = readFileSync(pngFile);
  const pngSize = [png.readUInt32BE(16), png.readUInt32BE(20)];
  check(
    'exports a 2× PNG',
    png.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) && pngSize[0] === svgDoc.width * 2 && pngSize[1] === svgDoc.height * 2,
    `${pngSize.join('×')}, ${png.length} bytes`,
  );

  const pdfFile = join(userData, 'smoke-image.pdf');
  await saveTo(pdfFile);
  const pdfPath = await win.evaluate((svg) => window.neuroloom.file.exportImage({ format: 'pdf', data: svg, suggestedName: 'smoke' }), svgText);
  const pdf = readFileSync(pdfFile).toString('latin1');
  const box = /\/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]/.exec(pdf);
  const near = (a, b) => Math.abs(Number(a) - b) <= 1;
  check(
    'prints a one-page vector PDF the size of the drawing',
    pdfPath === pdfFile && pdf.startsWith('%PDF-') && pdf.trimEnd().endsWith('%%EOF') && !!box && near(box[1], svgDoc.width * 0.75) && near(box[2], svgDoc.height * 0.75) &&
      (pdf.match(/\/Type\s*\/Page\b/g) ?? []).length === 1 && !pdf.includes('/Subtype /Image'),
    box ? `${box[1]}×${box[2]} pt, ${pdf.length} bytes` : 'no MediaBox',
  );

  // Native menus follow the interface language. The editor re-renders in the new
  // language before telling the main process, which can take a while on a slow
  // runner, so wait for the menu instead of a fixed time.
  const menuLabels = () => app.evaluate(({ Menu }) => Menu.getApplicationMenu()?.items.map((i) => i.label) ?? []);
  const send = (command) => app.evaluate(({ BrowserWindow }, c) => BrowserWindow.getAllWindows()[0].webContents.send('nl:command', c), command);
  const menuIn = async (command, label) => {
    await send(command);
    const deadline = Date.now() + 10_000;
    let labels = await menuLabels();
    while (!labels.includes(label) && Date.now() < deadline) {
      await win.waitForTimeout(200);
      labels = await menuLabels();
    }
    return labels;
  };
  const chinese = await menuIn('locale:zh', '模块');
  const english = await menuIn('locale:en', 'Module');
  check('menus switch language with the editor', chinese.includes('模块') && english.includes('Module'), `${chinese.join('/')} | ${english.join('/')}`);

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
