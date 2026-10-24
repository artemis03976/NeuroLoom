// What the Electron preload exposes to the editor as `window.neuroloom`.
// Everything crosses the context bridge, so only plain data and functions.

export const CHANNELS = {
  rpc: 'nl:rpc',
  envList: 'nl:env:list',
  envCurrent: 'nl:env:current',
  envSelect: 'nl:env:select',
  envCreate: 'nl:env:create',
  envProgress: 'nl:env:progress',
  fileOpen: 'nl:file:open',
  fileOpenPath: 'nl:file:open-path',
  fileSave: 'nl:file:save',
  fileExportPython: 'nl:file:export-python',
  fileExportPackage: 'nl:file:export-package',
  fileExportImage: 'nl:file:export-image',
  fileRecent: 'nl:file:recent',
  recoveryWrite: 'nl:recovery:write',
  recoveryRead: 'nl:recovery:read',
  recoveryClear: 'nl:recovery:clear',
  docState: 'nl:doc:state',
  docCloseAfterSave: 'nl:doc:close-after-save',
  locale: 'nl:locale',
  command: 'nl:command',
} as const;

export interface RpcFailure {
  code: number;
  message: string;
  data?: unknown;
}

export type RpcEnvelope = { ok: true; result: unknown } | { ok: false; error: RpcFailure };

export interface PythonCandidateInfo {
  path: string;
  source: 'env' | 'configured' | 'managed' | 'project' | 'launcher' | 'path' | 'conda';
  version?: string;
  torch?: string | null;
  cuda?: boolean;
  status: 'ready' | 'no-torch' | 'old-torch' | 'old-python' | 'broken';
}

export interface OpenedFile {
  path: string;
  name: string;
  text: string;
}

/** File formats of an exported diagram (PRD F7.3). */
export type ImageFormat = 'svg' | 'png' | 'pdf';

/**
 * A diagram to save. `data` is SVG markup for 'svg' and PNG bytes for 'png'. For
 * 'pdf' it is either SVG markup, which the desktop app prints to a vector PDF the
 * size of the drawing, or finished PDF bytes.
 */
export interface ImageExport {
  format: ImageFormat;
  data: string | Uint8Array;
  suggestedName: string;
}

/** Unsaved work kept by autosave, restored after a crash. */
export interface Recovery {
  text: string;
  path?: string;
  savedAt: number;
}

/** Commands sent from native menus (and handled the same way as keyboard shortcuts). */
export type MenuCommand =
  | 'new'
  | 'open'
  | 'save'
  | 'saveAs'
  | 'saveAndClose'
  | 'import'
  | 'exportPy'
  | 'exportOptions'
  | 'exportImage'
  | 'undo'
  | 'redo'
  | 'autoLayout'
  | 'environment'
  | 'packModule'
  | 'unpackModule'
  | 'goUp'
  | 'newModule'
  | 'toggleSnap'
  | 'toggleBottomPanel'
  | 'replay'
  | 'locale:zh'
  | 'locale:en'
  | `openRecent:${string}`;

export interface DesktopApi {
  platform: string;
  /** Calls the Python backend; rejects with an RpcFailure. */
  rpc(method: string, params?: Record<string, unknown>): Promise<unknown>;
  env: {
    list(): Promise<PythonCandidateInfo[]>;
    current(): Promise<string>;
    select(path: string): Promise<void>;
    /** Creates a managed venv with CPU PyTorch and switches to it; returns its interpreter. */
    create(): Promise<string>;
    onProgress(listener: (line: string) => void): () => void;
  };
  file: {
    open(): Promise<OpenedFile | null>;
    openPath(path: string): Promise<OpenedFile>;
    /** Saves to `path`, or asks where when it is absent. Returns the path, or null if cancelled. */
    save(args: { path?: string; text: string; suggestedName: string }): Promise<string | null>;
    exportPython(args: { text: string; suggestedName: string }): Promise<string | null>;
    /** Writes a Python package: asks for a folder and creates `suggestedName` in it. Returns the package folder, or null. */
    exportPackage(args: { files: { name: string; text: string }[]; suggestedName: string }): Promise<string | null>;
    /** Asks where to save an exported diagram and writes it. Returns the path, or null if cancelled. */
    exportImage(args: ImageExport): Promise<string | null>;
    recent(): Promise<string[]>;
  };
  recovery: {
    write(recovery: Recovery): Promise<void>;
    read(): Promise<Recovery | null>;
    clear(): Promise<void>;
  };
  doc: {
    setState(state: { dirty: boolean; title: string }): void;
    closeAfterSave(): void;
  };
  /** The interface language, for native menus and dialogs. */
  setLocale(locale: 'zh' | 'en'): void;
  onCommand(listener: (command: MenuCommand) => void): () => void;
}

declare global {
  interface Window {
    neuroloom?: DesktopApi;
  }
}
