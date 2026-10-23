import { copyNodes, entryModule, parseClip, serializeProject } from '@neuroloom/core';
import type { MenuCommand } from '@neuroloom/host-api';
import { useEffect } from 'react';
import { runCommand } from './commands';
import { host } from './host';
import { useDirty, useStudio } from './store';
import { useUi } from './ui';

const AUTOSAVE_MS = 30_000;
/** Recovery older than this is not offered (it is most likely stale). */
const RECOVERY_MAX_AGE_MS = 7 * 24 * 3600 * 1000;

const isEditable = (target: EventTarget | null) =>
  target instanceof HTMLElement &&
  (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));

function shortcut(e: KeyboardEvent): MenuCommand | undefined {
  if (!(e.ctrlKey || e.metaKey) || e.altKey) return undefined;
  switch (e.key.toLowerCase()) {
    case 'z':
      return e.shiftKey ? 'redo' : 'undo';
    case 'y':
      return 'redo';
    case 's':
      return e.shiftKey ? 'saveAs' : 'save';
    case 'o':
      return 'open';
    case 'n':
      return 'new';
    case 'e':
      return 'exportPy';
    case 'i':
      return 'import';
    case 'l':
      return 'autoLayout';
    default:
      return undefined;
  }
}

/** Keyboard shortcuts and native-menu commands go through the same dispatcher. */
export function useCommands(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Tab on the canvas opens quick-add at the pointer (PRD F1.6).
      if (e.key === 'Tab' && !e.ctrlKey && !e.metaKey && !e.altKey && !isEditable(e.target) && !useUi.getState().dialog) {
        const box = document.querySelector('.react-flow')?.getBoundingClientRect();
        const pointer = useUi.getState().pointer;
        if (box) {
          e.preventDefault();
          const inside = pointer && pointer.x > box.left && pointer.x < box.right && pointer.y > box.top && pointer.y < box.bottom;
          const at = inside ? pointer : { x: box.left + box.width / 2, y: box.top + box.height / 2 };
          useUi.getState().setQuickAdd({ screen: at });
        }
        return;
      }
      const command = shortcut(e);
      if (!command) return;
      // Inside a text field, undo/redo belong to the field.
      if ((command === 'undo' || command === 'redo') && isEditable(e.target)) return;
      e.preventDefault();
      runCommand(command);
    };
    window.addEventListener('keydown', onKey);
    const unsubscribe = host.onCommand(runCommand);
    return () => {
      window.removeEventListener('keydown', onKey);
      unsubscribe();
    };
  }, []);
}

const PASTE_OFFSET = 40;

/**
 * Copy, cut, paste (PRD F1.2) through the document's clipboard events, so the
 * desktop app's native Edit menu and the keyboard behave the same. Ctrl+D duplicates.
 */
export function useClipboard(): void {
  useEffect(() => {
    let pasteCount = 0;
    const selection = () => useStudio.getState().flowNodes.filter((n) => n.selected).map((n) => n.id);
    const clipOfSelection = () => {
      const ids = selection();
      return ids.length > 0 ? copyNodes(entryModule(useStudio.getState().project), ids) : null;
    };
    const onCopy = (e: ClipboardEvent) => {
      if (isEditable(document.activeElement)) return;
      const clip = clipOfSelection();
      if (!clip || !e.clipboardData) return;
      e.preventDefault();
      e.clipboardData.setData('text/plain', JSON.stringify(clip));
      pasteCount = 0;
      if (e.type === 'cut') useStudio.getState().deleteNodes(clip.nodes.map((n) => n.id));
    };
    const onPaste = (e: ClipboardEvent) => {
      if (isEditable(document.activeElement)) return;
      const clip = parseClip(e.clipboardData?.getData('text/plain') ?? '');
      if (!clip) return;
      e.preventDefault();
      pasteCount++;
      useStudio.getState().paste(clip, { x: PASTE_OFFSET * pasteCount, y: PASTE_OFFSET * pasteCount });
    };
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== 'd' || isEditable(e.target)) return;
      const clip = clipOfSelection();
      if (!clip) return;
      e.preventDefault();
      useStudio.getState().paste(clip, { x: PASTE_OFFSET, y: PASTE_OFFSET });
    };
    document.addEventListener('copy', onCopy);
    document.addEventListener('cut', onCopy);
    document.addEventListener('paste', onPaste);
    window.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('copy', onCopy);
      document.removeEventListener('cut', onCopy);
      document.removeEventListener('paste', onPaste);
      window.removeEventListener('keydown', onKey);
    };
  }, []);
}

/** Window title, unsaved-changes guard, autosave for crash recovery, and the recovery offer. */
export function useDocumentLifecycle(): void {
  const dirty = useDirty();
  const name = useStudio((s) => s.doc.name);

  useEffect(() => host.setDocState({ dirty, title: name }), [dirty, name]);

  useEffect(() => {
    if (host.kind !== 'web' || !dirty) return;
    const guard = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener('beforeunload', guard);
    return () => window.removeEventListener('beforeunload', guard);
  }, [dirty]);

  useEffect(() => {
    const timer = setInterval(() => {
      const { project, doc } = useStudio.getState();
      if (doc.saved === project) return;
      void host.recovery.write({ text: serializeProject(project), path: doc.path ?? undefined, savedAt: Date.now() });
    }, AUTOSAVE_MS);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    void host.recovery.read().then((r) => {
      if (r && Date.now() - r.savedAt < RECOVERY_MAX_AGE_MS) useUi.getState().setRecovery(r);
    });
  }, []);
}
