// Transient UI state that commands need to reach: dialogs, menus, toasts, recovery prompt.
import type { Recovery } from '@neuroloom/host-api';
import { create } from 'zustand';
import type { PortLink } from './store';

export interface QuickAdd {
  /** Where the panel opens and the node goes, in window coordinates. */
  screen: { x: number; y: number };
  link?: PortLink;
}

export interface Toast {
  id: number;
  tone: 'success' | 'error' | 'info';
  message: string;
  details?: string[];
}

/** A one-field dialog, e.g. the class name for "pack as module". */
export interface Prompt {
  title: string;
  label: string;
  initial: string;
  /** An error message, or undefined when the value is acceptable. */
  validate(value: string): string | undefined;
  submit(value: string): void;
}

export interface ContextMenu {
  screen: { x: number; y: number };
  /** The node right-clicked, or none for the pane. */
  nodeId?: string;
}

interface UiState {
  dialog: 'new' | 'import' | 'environment' | 'exportOptions' | 'exportImage' | null;
  prompt: Prompt | null;
  contextMenu: ContextMenu | null;
  toasts: Toast[];
  recovery: Recovery | null;
  quickAdd: QuickAdd | null;
  /** Last pointer position over the canvas, so Tab opens quick-add where the user is looking. */
  pointer: { x: number; y: number } | null;
  /** Bumped when the Python environment changes, so status displays refetch. */
  backendVersion: number;
  snapToGrid: boolean;
  openDialog(dialog: UiState['dialog']): void;
  setPrompt(prompt: Prompt | null): void;
  setContextMenu(menu: ContextMenu | null): void;
  setRecovery(recovery: Recovery | null): void;
  backendChanged(): void;
  setQuickAdd(quickAdd: QuickAdd | null): void;
  toggleSnap(): void;
}

const SNAP_KEY = 'neuroloom.snapToGrid';
const loadSnap = () => {
  try {
    return localStorage.getItem(SNAP_KEY) === '1';
  } catch {
    return false;
  }
};

export const useUi = create<UiState>()((set, get) => ({
  dialog: null,
  prompt: null,
  contextMenu: null,
  toasts: [],
  recovery: null,
  quickAdd: null,
  pointer: null,
  backendVersion: 0,
  snapToGrid: loadSnap(),
  openDialog: (dialog) => set({ dialog }),
  setPrompt: (prompt) => set({ prompt }),
  setContextMenu: (contextMenu) => set({ contextMenu }),
  setRecovery: (recovery) => set({ recovery }),
  backendChanged: () => set({ backendVersion: get().backendVersion + 1 }),
  setQuickAdd: (quickAdd) => set({ quickAdd }),
  toggleSnap() {
    const snapToGrid = !get().snapToGrid;
    try {
      localStorage.setItem(SNAP_KEY, snapToGrid ? '1' : '0');
    } catch {
      // Not persisted; the toggle still applies to this session.
    }
    set({ snapToGrid });
  },
}));

let nextToast = 1;
export function notify(message: string, tone: Toast['tone'] = 'info', details?: string[]): void {
  const toast = { id: nextToast++, tone, message, details };
  useUi.setState((s) => ({ toasts: [...s.toasts.slice(-3), toast] }));
  setTimeout(() => useUi.setState((s) => ({ toasts: s.toasts.filter((t) => t.id !== toast.id) })), tone === 'error' ? 8000 : 5000);
}
