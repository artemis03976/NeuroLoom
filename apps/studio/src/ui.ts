// Transient UI state that commands need to reach: dialogs, toasts, recovery prompt.
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

interface UiState {
  dialog: 'new' | 'import' | 'environment' | null;
  toasts: Toast[];
  recovery: Recovery | null;
  quickAdd: QuickAdd | null;
  /** Last pointer position over the canvas, so Tab opens quick-add where the user is looking. */
  pointer: { x: number; y: number } | null;
  /** Bumped when the Python environment changes, so status displays refetch. */
  backendVersion: number;
  openDialog(dialog: UiState['dialog']): void;
  setRecovery(recovery: Recovery | null): void;
  backendChanged(): void;
  setQuickAdd(quickAdd: QuickAdd | null): void;
}

export const useUi = create<UiState>()((set, get) => ({
  dialog: null,
  toasts: [],
  recovery: null,
  quickAdd: null,
  pointer: null,
  backendVersion: 0,
  openDialog: (dialog) => set({ dialog }),
  setRecovery: (recovery) => set({ recovery }),
  backendChanged: () => set({ backendVersion: get().backendVersion + 1 }),
  setQuickAdd: (quickAdd) => set({ quickAdd }),
}));

let nextToast = 1;
export function notify(message: string, tone: Toast['tone'] = 'info', details?: string[]): void {
  const toast = { id: nextToast++, tone, message, details };
  useUi.setState((s) => ({ toasts: [...s.toasts.slice(-3), toast] }));
  setTimeout(() => useUi.setState((s) => ({ toasts: s.toasts.filter((t) => t.id !== toast.id) })), tone === 'error' ? 8000 : 5000);
}
