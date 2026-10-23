// Document and editor commands, shared by keyboard shortcuts, the native menu
// (desktop) and the top bar.
import { entryModule, parseProjectText, serializeProject } from '@neuroloom/core';
import type { MenuCommand } from '@neuroloom/host-api';
import { errorMessage, host } from './host';
import { useStudio } from './store';
import { notify, useUi } from './ui';

const isDirty = () => {
  const { doc, project } = useStudio.getState();
  return doc.saved !== project;
};

const confirmDiscard = () => !isDirty() || window.confirm('当前项目有未保存的更改，确定要放弃吗？');

const baseName = (path: string) => path.split(/[\\/]/).pop() ?? path;
const stem = (name: string) => baseName(name).replace(/\.(loom|json)$/i, '');

/** Opens the template picker (blank project or a template). */
export function newDocument(): void {
  if (!confirmDiscard()) return;
  useUi.getState().openDialog('new');
}

export async function openDocument(path?: string): Promise<void> {
  if (!confirmDiscard()) return;
  try {
    const file = path ? await host.openPath(path) : await host.openFile();
    if (!file) return;
    const project = parseProjectText(file.text);
    useStudio.getState().loadProject(project, { path: file.path || null, name: stem(file.name) });
    void host.recovery.clear();
  } catch (e) {
    notify(`无法打开文件：${errorMessage(e)}`, 'error');
  }
}

/** Returns true when the document ended up saved. */
export async function saveDocument(saveAs = false): Promise<boolean> {
  const { project, doc } = useStudio.getState();
  try {
    const where = await host.saveFile({
      path: saveAs ? undefined : (doc.path ?? undefined),
      text: serializeProject(project),
      suggestedName: doc.name,
    });
    if (!where) return false;
    useStudio.getState().markSaved(project, host.kind === 'desktop' ? where : null, stem(where));
    void host.recovery.clear();
    notify(host.kind === 'desktop' ? `已保存到 ${where}` : `已下载 ${baseName(where)}`, 'success');
    return true;
  } catch (e) {
    notify(`保存失败：${errorMessage(e)}`, 'error');
    return false;
  }
}

export async function exportPython(): Promise<void> {
  const { codegen, project } = useStudio.getState();
  const name = entryModule(project).className.replace(/[^A-Za-z0-9_]/g, '').toLowerCase() || 'model';
  try {
    const where = await host.exportPython({ text: codegen.code, suggestedName: name });
    if (where) notify(host.kind === 'desktop' ? `已导出 ${where}` : `已下载 ${baseName(where)}`, 'success');
  } catch (e) {
    notify(`导出失败：${errorMessage(e)}`, 'error');
  }
}

export function restoreRecovery(): void {
  const recovery = useUi.getState().recovery;
  if (!recovery) return;
  try {
    const project = parseProjectText(recovery.text);
    useStudio.getState().loadProject(project, {
      path: recovery.path ?? null,
      name: recovery.path ? stem(recovery.path) : project.name,
      saved: false,
    });
    notify('已恢复上次未保存的工作', 'success');
  } catch (e) {
    notify(`无法恢复：${errorMessage(e)}`, 'error');
  }
  useUi.getState().setRecovery(null);
}

export function discardRecovery(): void {
  useUi.getState().setRecovery(null);
  void host.recovery.clear();
}

export function runCommand(command: MenuCommand): void {
  const studio = useStudio.getState();
  switch (command) {
    case 'new':
      return newDocument();
    case 'open':
      return void openDocument();
    case 'save':
      return void saveDocument(false);
    case 'saveAs':
      return void saveDocument(true);
    case 'saveAndClose':
      return void saveDocument(false).then((saved) => saved && host.closeAfterSave());
    case 'import':
      return useUi.getState().openDialog('import');
    case 'environment':
      return useUi.getState().openDialog('environment');
    case 'exportPy':
      return void exportPython();
    case 'undo':
      return studio.undo();
    case 'redo':
      return studio.redo();
    case 'autoLayout':
      return studio.requestLayout();
    default:
      if (command.startsWith('openRecent:')) void openDocument(command.slice('openRecent:'.length));
  }
}
