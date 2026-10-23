// Sandboxed preload, bundled to CommonJS. Exposes exactly the DesktopApi contract.
import { CHANNELS, type DesktopApi, type MenuCommand, type RpcEnvelope } from '@neuroloom/host-api';
import { contextBridge, ipcRenderer, type IpcRendererEvent } from 'electron';

const invoke = <T>(channel: string, ...args: unknown[]): Promise<T> => ipcRenderer.invoke(channel, ...args) as Promise<T>;

function subscribe<T>(channel: string, listener: (value: T) => void): () => void {
  const wrapped = (_e: IpcRendererEvent, value: T) => listener(value);
  ipcRenderer.on(channel, wrapped);
  return () => ipcRenderer.removeListener(channel, wrapped);
}

const api: DesktopApi = {
  platform: process.platform,
  async rpc(method, params = {}) {
    const envelope = await invoke<RpcEnvelope>(CHANNELS.rpc, method, params);
    if (!envelope.ok) throw envelope.error;
    return envelope.result;
  },
  env: {
    list: () => invoke(CHANNELS.envList),
    current: () => invoke(CHANNELS.envCurrent),
    select: (path) => invoke(CHANNELS.envSelect, path),
    create: () => invoke(CHANNELS.envCreate),
    onProgress: (listener) => subscribe<string>(CHANNELS.envProgress, listener),
  },
  file: {
    open: () => invoke(CHANNELS.fileOpen),
    openPath: (path) => invoke(CHANNELS.fileOpenPath, path),
    save: (args) => invoke(CHANNELS.fileSave, args),
    exportPython: (args) => invoke(CHANNELS.fileExportPython, args),
    recent: () => invoke(CHANNELS.fileRecent),
  },
  recovery: {
    write: (recovery) => invoke(CHANNELS.recoveryWrite, recovery),
    read: () => invoke(CHANNELS.recoveryRead),
    clear: () => invoke(CHANNELS.recoveryClear),
  },
  doc: {
    setState: (state) => ipcRenderer.send(CHANNELS.docState, state),
    closeAfterSave: () => ipcRenderer.send(CHANNELS.docCloseAfterSave),
  },
  onCommand: (listener) => subscribe<MenuCommand>(CHANNELS.command, listener),
};

contextBridge.exposeInMainWorld('neuroloom', api);
