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
// An installed app carries the studio UI, the `neuroloom` Python package and the window
// icon as extra resources (see electron-builder.config.mjs); in development they come
// straight from the repository.
const repoRoot = resolve(here, '../../..');
const studioDist = app.isPackaged ? join(process.resourcesPath, 'studio') : join(repoRoot, 'apps/studio/dist');
const pythonDir = app.isPackaged ? join(process.resourcesPath, 'python') : join(repoRoot, 'python');
const windowIcon = app.isPackaged ? join(process.resourcesPath, 'icon.png') : join(repoRoot, 'brand/assets/icons/icon-256.png');
const devUrl = process.env.NEUROLOOM_DEV_URL?.replace(/\/$/, '');
const APP_ORIGIN = 'app://studio';

// The installed Python package sits inside the app (read-only, and signed on macOS):
// the user's interpreter must not write __pycache__ next to it.
if (app.isPackaged) process.env.PYTHONDONTWRITEBYTECODE = '1';

// Tests point userData at a throwaway directory so they never touch real settings.
// (Before the single-instance lock, which is held per userData directory.)
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

/** The editor's language; menus and native dialogs follow it. */
const locale = () => settings?.data.locale ?? (app.getLocale().toLowerCase().startsWith('zh') ? 'zh' : 'en');
const t = (zh: string, en: string) => (locale() === 'en' ? en : zh);
const rebuildMenu = () => buildMenu(settings.recent(), locale());

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
  if (!existsSync(path)) throw new Error(t(`找不到解释器：${path}`, `Interpreter not found: ${path}`));
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
  if (!['.loom', '.json'].includes(extname(path).toLowerCase())) throw new Error(t('只能打开 .loom 或 .json 文件', 'Only .loom or .json files can be opened'));
  const text = readFileSync(path, 'utf8');
  settings.addRecent(path);
  rebuildMenu();
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
    if (!base) throw new Error(t('没有找到 Python 3.10 或更新的版本，请先安装 Python', 'No Python 3.10 or newer was found; install Python first'));
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
        { name: t('NeuroLoom 项目', 'NeuroLoom project'), extensions: ['loom'] },
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
        filters: [{ name: t('NeuroLoom 项目', 'NeuroLoom project'), extensions: ['loom'] }],
      });
      if (result.canceled || !result.filePath) return null;
      path = extname(result.filePath) ? result.filePath : `${result.filePath}.loom`;
    }
    writeAtomic(path, args.text);
    settings.addRecent(path);
    rebuildMenu();
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
  ipcMain.on(CHANNELS.locale, (_e, value: unknown) => {
    if ((value !== 'zh' && value !== 'en') || value === settings.data.locale) return;
    settings.data.locale = value;
    settings.save();
    rebuildMenu();
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
    icon: windowIcon,
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
      buttons: [t('保存', 'Save'), t('不保存', "Don't save"), t('取消', 'Cancel')],
      defaultId: 0,
      cancelId: 2,
      message: t('当前项目有未保存的更改', 'The project has unsaved changes'),
      detail: t('关闭前要保存吗？', 'Save before closing?'),
    });
    if (choice === 0) win.webContents.send(CHANNELS.command, 'saveAndClose');
    else if (choice === 1) {
      allowClose = true;
      win.close();
    }
  });

  win.webContents.on('did-finish-load', () => deliverProject(win));
  void win.loadURL(devUrl ? `${devUrl}/` : `${APP_ORIGIN}/index.html`);
  return win;
}

// ---------------------------------------------------------------- projects opened by the OS

// Double-clicking a .loom file (file association) starts the app with the path as an
// argument on Windows and Linux, sends `open-file` on macOS, and reaches an already
// running app as `second-instance`. The editor opens it like a recent file.
let pendingProject: string | undefined;
let started = false;

/** The first .loom file among command-line arguments (a path, or a file:// URL from a Linux desktop entry). */
function projectFromArgs(argv: string[], cwd = process.cwd()): string | undefined {
  for (const arg of argv.slice(1)) {
    if (arg.startsWith('-')) continue;
    let path: string;
    try {
      path = arg.startsWith('file://') ? fileURLToPath(arg) : resolve(cwd, arg);
    } catch {
      continue;
    }
    if (extname(path).toLowerCase() === '.loom' && existsSync(path)) return path;
  }
  return undefined;
}

function deliverProject(win: BrowserWindow): void {
  if (!pendingProject) return;
  // The preload holds commands until the editor subscribes, so this may arrive early.
  win.webContents.send(CHANNELS.command, `openRecent:${pendingProject}`);
  pendingProject = undefined;
}

function openProject(path: string): void {
  pendingProject = path;
  if (!started) return; // the first window picks it up
  const win = BrowserWindow.getAllWindows()[0];
  if (!win) {
    createWindow();
    return;
  }
  if (win.isMinimized()) win.restore();
  win.focus();
  if (!win.webContents.isLoading()) deliverProject(win);
}

app.on('open-file', (e, path) => {
  e.preventDefault();
  openProject(path);
});

// One app instance: a second launch hands its file to the first and exits.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', (_e, argv, workingDirectory) => {
    const path = projectFromArgs(argv, workingDirectory);
    if (path) return openProject(path);
    const win = BrowserWindow.getAllWindows()[0];
    if (win?.isMinimized()) win.restore();
    win?.focus();
  });
  pendingProject ??= projectFromArgs(process.argv);

  app.whenReady().then(() => {
    settings = new SettingsStore(app.getPath('userData'));
    serveStudio();
    registerIpc();
    rebuildMenu();
    createWindow();
    started = true;
    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });
}

app.on('window-all-closed', () => {
  void backendPromise?.then((b) => b.dispose());
  if (process.platform !== 'darwin') app.quit();
});
