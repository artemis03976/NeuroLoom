// The editor's view of its host: the Electron app (native dialogs, files on disk,
// environment management) or a plain browser tab (downloads and file pickers).
import type { DesktopApi, MenuCommand, OpenedFile, Recovery } from '@neuroloom/host-api';
import { tr, type Locale } from './i18n';

export interface Host {
  kind: 'desktop' | 'web';
  openFile(): Promise<OpenedFile | null>;
  openPath(path: string): Promise<OpenedFile>;
  /** Returns where the file went (a path on desktop, a label on the web), or null if cancelled. */
  saveFile(args: { path?: string; text: string; suggestedName: string }): Promise<string | null>;
  exportPython(args: { text: string; suggestedName: string }): Promise<string | null>;
  recent(): Promise<string[]>;
  recovery: { write(r: Recovery): Promise<void>; read(): Promise<Recovery | null>; clear(): Promise<void> };
  setDocState(state: { dirty: boolean; title: string }): void;
  closeAfterSave(): void;
  onCommand(listener: (command: MenuCommand) => void): () => void;
  /** Tells native menus and dialogs the interface language. */
  setLocale(locale: Locale): void;
  /** Python environment management; desktop only. */
  env?: DesktopApi['env'];
}

function desktopHost(api: DesktopApi): Host {
  return {
    kind: 'desktop',
    openFile: () => api.file.open(),
    openPath: (path) => api.file.openPath(path),
    saveFile: (args) => api.file.save(args),
    exportPython: (args) => api.file.exportPython(args),
    recent: () => api.file.recent(),
    recovery: api.recovery,
    setDocState: (state) => api.doc.setState(state),
    closeAfterSave: () => api.doc.closeAfterSave(),
    onCommand: (listener) => api.onCommand(listener),
    setLocale: (locale) => api.setLocale(locale),
    env: api.env,
  };
}

function download(text: string, name: string, type: string): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

const RECOVERY_KEY = 'neuroloom.recovery';

function webHost(): Host {
  return {
    kind: 'web',
    openFile: () =>
      new Promise((resolve) => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.loom,.json,application/json';
        input.addEventListener('cancel', () => resolve(null));
        input.addEventListener('change', async () => {
          const file = input.files?.[0];
          resolve(file ? { path: '', name: file.name, text: await file.text() } : null);
        });
        input.click();
      }),
    openPath: () => Promise.reject(new Error(tr('浏览器中无法按路径打开文件', 'A browser cannot open files by path'))),
    async saveFile({ text, suggestedName }) {
      const name = `${suggestedName}.loom`;
      download(text, name, 'application/json');
      return name;
    },
    async exportPython({ text, suggestedName }) {
      const name = `${suggestedName}.py`;
      download(text, name, 'text/x-python');
      return name;
    },
    recent: async () => [],
    recovery: {
      async write(r) {
        try {
          localStorage.setItem(RECOVERY_KEY, JSON.stringify(r));
        } catch {
          // Storage may be full or disabled; recovery is best effort.
        }
      },
      async read() {
        try {
          const raw = localStorage.getItem(RECOVERY_KEY);
          return raw ? (JSON.parse(raw) as Recovery) : null;
        } catch {
          return null;
        }
      },
      async clear() {
        try {
          localStorage.removeItem(RECOVERY_KEY);
        } catch {
          // Nothing to clear when storage is unavailable.
        }
      },
    },
    setDocState: ({ dirty, title }) => {
      document.title = `${dirty ? '● ' : ''}${title} — NeuroLoom`;
    },
    closeAfterSave: () => {},
    onCommand: () => () => {},
    setLocale: (locale) => {
      document.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en';
    },
  };
}

export const host: Host = window.neuroloom ? desktopHost(window.neuroloom) : webHost();

/** Readable message for any rejection, without Electron's "Error invoking remote method" prefix. */
export function errorMessage(e: unknown): string {
  const raw = e instanceof Error ? e.message : String((e as { message?: unknown })?.message ?? e);
  return raw.replace(/^Error invoking remote method '[^']+': (?:Error: )?/, '');
}
