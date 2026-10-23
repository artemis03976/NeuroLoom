// Electron main process: window, menus, files, and the Python backend (PRD 15.2–15.4).
import { createEnvironment, discoverPythons, PythonBackend, type RpcFailure } from '@neuroloom/backend-bridge';
import { CHANNELS, type Recovery, type RpcEnvelope } from '@neuroloom/host-api';
import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  nativeTheme,
  net,
  protocol,
  shell,
  type IpcMainInvokeEvent,
} from 'electron';
import { existsSync, readFileSync, rmSync } from 'node:fs';
import { basename, dirname, extname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildMenu } from './menu';
import { SettingsStore, writeAtomic } from './settings';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../../..');
const studioDist = join(repoRoot, 'apps/studio/dist');
const pythonDir = join(repoRoot, 'python');
const devUrl = process.env.NEUROLOOM_DEV_URL?.replace(/\/$/, '');
const APP_ORIGIN = 'app://studio';

// Tests point userData at a throwaway directory so they never touch real settings.
if (process.env.NEUROLOOM_USER_DATA) app.setPath('userData', process.env.NEUROLOOM_USER_DATA);

protocol.registerSchemesAsPrivileged([{ scheme: 'app', privileges: { standard: true, secure: true, supportFetchAPI: true } }]);

const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'", // React Flow positions nodes with inline styles
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "connect-src 'self'",
].join('; ');

let settings: SettingsStore;
let docDirty = false;
let allowClose = false;

// ---------------------------------------------------------------- backend

const managedEnvDir = () => join(app.getPath('userData'), 'python-env');
const projectVenv = () => (app.isPackaged ? undefined : join(pythonDir, '.venv'));

let backendPromise: Promise<PythonBackend> | undefined;
function backend(): Promise<PythonBackend> {
  backendPromise ??= (async () => {
    let python = settings.data.pythonPath;
    if (!python || !existsSync(python)) {
      const found = await discoverPythons({ managedEnvDir: managedEnvDir(), projectVenv: projectVenv() });
      python = found.find((c) => c.status === 'ready')?.path ?? found[0]?.path ?? 'python';
    }
    return new PythonBackend(python, { cwd: pythonDir, onLog: (line) => console.error(`[python] ${line}`) });
  })();
  return backendPromise;
}

async function selectPython(path: string): Promise<void> {
  if (!existsSync(path)) throw new Error(`找不到解释器：${path}`);
  settings.data.pythonPath = path;
  settings.save();
  (await backend()).setInterpreter(path);
}

// ---------------------------------------------------------------- IPC

function trusted(e: IpcMainInvokeEvent): boolean {
  const url = e.senderFrame?.url ?? '';
  return url.startsWith(`${APP_ORIGIN}/`) || (!!devUrl && url.startsWith(`${devUrl}/`));
}

function handle(channel: string, fn: (e: IpcMainInvokeEvent, ...args: never[]) => unknown): void {
  ipcMain.handle(channel, (e, ...args) => {
    if (!trusted(e)) throw new Error('Rejected IPC from an untrusted page');
    return fn(e, ...(args as never[]));
  });
}

const windowOf = (e: IpcMainInvokeEvent) => BrowserWindow.fromWebContents(e.sender) ?? undefined;
const recoveryFile = () => join(app.getPath('userData'), 'recovery.json');

function readProjectFile(path: string) {
  if (!['.loom', '.json'].includes(extname(path).toLowerCase())) throw new Error('只能打开 .loom 或 .json 文件');
  const text = readFileSync(path, 'utf8');
  settings.addRecent(path);
  buildMenu(settings.recent());
  return { path, name: basename(path), text };
}

function registerIpc(): void {
  handle(CHANNELS.rpc, async (_e, method: string, params: Record<string, unknown>): Promise<RpcEnvelope> => {
    try {
      return { ok: true, result: await (await backend()).call(method, params) };
    } catch (error) {
      const failure = error as RpcFailure;
      return { ok: false, error: { code: failure.code ?? -32603, message: failure.message ?? String(error), data: failure.data } };
    }
  });

  handle(CHANNELS.envList, async () =>
    discoverPythons({ configured: settings.data.pythonPath, managedEnvDir: managedEnvDir(), projectVenv: projectVenv() }),
  );
  handle(CHANNELS.envCurrent, async () => (await backend()).interpreter);
  handle(CHANNELS.envSelect, (_e, path: string) => selectPython(path));
  handle(CHANNELS.envCreate, async (e) => {
    const candidates = await discoverPythons({ projectVenv: projectVenv() });
    const base = candidates.find((c) => c.status !== 'old-python' && c.status !== 'broken');
    if (!base) throw new Error('没有找到 Python 3.10 或更新的版本，请先安装 Python');
    const python = await createEnvironment({
      base: base.path,
      target: managedEnvDir(),
      onLine: (line) => e.sender.send(CHANNELS.envProgress, line),
    });
    await selectPython(python);
    return python;
  });

  handle(CHANNELS.fileOpen, async (e) => {
    const result = await dialog.showOpenDialog(windowOf(e)!, {
      properties: ['openFile'],
      filters: [
        { name: 'NeuroLoom 项目', extensions: ['loom'] },
        { name: 'JSON', extensions: ['json'] },
      ],
    });
    return result.canceled || !result.filePaths[0] ? null : readProjectFile(result.filePaths[0]);
  });
  handle(CHANNELS.fileOpenPath, (_e, path: string) => readProjectFile(path));
  handle(CHANNELS.fileSave, async (e, args: { path?: string; text: string; suggestedName: string }) => {
    let path = args.path;
    if (!path) {
      const result = await dialog.showSaveDialog(windowOf(e)!, {
        defaultPath: join(app.getPath('documents'), `${args.suggestedName}.loom`),
        filters: [{ name: 'NeuroLoom 项目', extensions: ['loom'] }],
      });
      if (result.canceled || !result.filePath) return null;
      path = extname(result.filePath) ? result.filePath : `${result.filePath}.loom`;
    }
    writeAtomic(path, args.text);
    settings.addRecent(path);
    buildMenu(settings.recent());
    return path;
  });
  handle(CHANNELS.fileExportPython, async (e, args: { text: string; suggestedName: string }) => {
    const result = await dialog.showSaveDialog(windowOf(e)!, {
      defaultPath: join(app.getPath('documents'), `${args.suggestedName}.py`),
      filters: [{ name: 'Python', extensions: ['py'] }],
    });
    if (result.canceled || !result.filePath) return null;
    writeAtomic(result.filePath, args.text);
    return result.filePath;
  });
  handle(CHANNELS.fileRecent, () => settings.recent());

  handle(CHANNELS.recoveryWrite, (_e, recovery: Recovery) => writeAtomic(recoveryFile(), JSON.stringify(recovery)));
  handle(CHANNELS.recoveryRead, () => {
    try {
      return JSON.parse(readFileSync(recoveryFile(), 'utf8')) as Recovery;
    } catch {
      return null;
    }
  });
  handle(CHANNELS.recoveryClear, () => rmSync(recoveryFile(), { force: true }));

  ipcMain.on(CHANNELS.docState, (e, state: { dirty: boolean; title: string }) => {
    const win = BrowserWindow.fromWebContents(e.sender);
    if (!win) return;
    docDirty = state.dirty;
    win.setTitle(`${state.dirty ? '● ' : ''}${state.title} — NeuroLoom`);
    win.setDocumentEdited(state.dirty);
  });
  ipcMain.on(CHANNELS.docCloseAfterSave, (e) => {
    allowClose = true;
    BrowserWindow.fromWebContents(e.sender)?.close();
  });
}

// ---------------------------------------------------------------- window

function serveStudio(): void {
  protocol.handle('app', async (req) => {
    const { host, pathname } = new URL(req.url);
    if (host !== 'studio') return new Response('not found', { status: 404 });
    const target = resolve(studioDist, `.${decodeURIComponent(pathname)}`);
    const rel = relative(studioDist, target);
    if (!rel || rel.startsWith('..') || isAbsolute(rel)) return new Response('bad request', { status: 400 });
    const res = await net.fetch(pathToFileURL(target).toString());
    if (!target.endsWith('.html')) return res;
    return new Response(res.body, { status: res.status, headers: { 'content-type': 'text/html', 'content-security-policy': CSP } });
  });
}

function createWindow(): BrowserWindow {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 640,
    show: false,
    title: 'NeuroLoom',
    icon: join(repoRoot, 'brand/assets/icons/icon-256.png'),
    backgroundColor: nativeTheme.shouldUseDarkColors ? '#0E1222' : '#F6F7FB',
    webPreferences: {
      preload: join(here, 'preload.cjs'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      spellcheck: false,
    },
  });
  win.once('ready-to-show', () => win.show());

  // Links open in the system browser; the app window never navigates away.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://')) void shell.openExternal(url);
    return { action: 'deny' };
  });
  win.webContents.on('will-navigate', (e, url) => {
    if (!url.startsWith(`${APP_ORIGIN}/`) && !(devUrl && url.startsWith(`${devUrl}/`))) e.preventDefault();
  });

  win.on('close', (e) => {
    if (!docDirty || allowClose) return;
    e.preventDefault();
    const choice = dialog.showMessageBoxSync(win, {
      type: 'warning',
      buttons: ['保存', '不保存', '取消'],
      defaultId: 0,
      cancelId: 2,
      message: '当前项目有未保存的更改',
      detail: '关闭前要保存吗？',
    });
    if (choice === 0) win.webContents.send(CHANNELS.command, 'saveAndClose');
    else if (choice === 1) {
      allowClose = true;
      win.close();
    }
  });

  void win.loadURL(devUrl ? `${devUrl}/` : `${APP_ORIGIN}/index.html`);
  return win;
}

app.whenReady().then(() => {
  settings = new SettingsStore(app.getPath('userData'));
  serveStudio();
  registerIpc();
  buildMenu(settings.recent());
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  void backendPromise?.then((b) => b.dispose());
  if (process.platform !== 'darwin') app.quit();
});
