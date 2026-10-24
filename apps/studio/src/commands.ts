// Document and editor commands, shared by keyboard shortcuts, the native menu
// (desktop), the top bar and the canvas context menu.
import { EditError, generateFiles, isValidClassName, parseProjectText, serializeProject, snakeCase, uniqueClassName, type ModuleId, type NodeId } from '@neuroloom/core';
import type { MenuCommand } from '@neuroloom/host-api';
import { useReplay } from './canvas/replay';
import { errorMessage, host } from './host';
import { tr, useLocale } from './i18n';
import { probeShapes } from './lib/probe';
import { useStudio } from './store';
import { notify, useUi } from './ui';

const isDirty = () => {
  const { doc, project } = useStudio.getState();
  return doc.saved !== project;
};

const confirmDiscard = () => !isDirty() || window.confirm(tr('当前项目有未保存的更改，确定要放弃吗？', 'The project has unsaved changes. Discard them?'));

const baseName = (path: string) => path.split(/[\\/]/).pop() ?? path;
const stem = (name: string) => baseName(name).replace(/\.(loom|json)$/i, '');

/** Runs an edit that may be refused (EditError), reporting the reason instead of failing. */
export function attempt(edit: () => void): boolean {
  try {
    edit();
    return true;
  } catch (e) {
    if (e instanceof EditError) {
      notify(e.message, 'error');
      return false;
    }
    throw e;
  }
}

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
    notify(tr(`无法打开文件：${errorMessage(e)}`, `Cannot open the file: ${errorMessage(e)}`), 'error');
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
    notify(host.kind === 'desktop' ? tr(`已保存到 ${where}`, `Saved to ${where}`) : tr(`已下载 ${baseName(where)}`, `Downloaded ${baseName(where)}`), 'success');
    return true;
  } catch (e) {
    notify(tr(`保存失败：${errorMessage(e)}`, `Saving failed: ${errorMessage(e)}`), 'error');
    return false;
  }
}

export async function exportPython(): Promise<void> {
  const { codegen, project, analysis } = useStudio.getState();
  const entry = project.modules[project.entry]!;
  const name = entry.className.replace(/[^A-Za-z0-9_]/g, '').toLowerCase() || 'model';
  try {
    const where = project.exportOptions?.splitFiles
      ? await host.exportPackage({ files: generateFiles(project, analysis), suggestedName: snakeCase(entry.className) || 'model' })
      : await host.exportPython({ text: codegen.code, suggestedName: name });
    if (where) notify(host.kind === 'desktop' ? tr(`已导出 ${where}`, `Exported ${where}`) : tr(`已下载 ${baseName(where)}`, `Downloaded ${baseName(where)}`), 'success');
  } catch (e) {
    notify(tr(`导出失败：${errorMessage(e)}`, `Export failed: ${errorMessage(e)}`), 'error');
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
    notify(tr('已恢复上次未保存的工作', 'Restored your unsaved work'), 'success');
  } catch (e) {
    notify(tr(`无法恢复：${errorMessage(e)}`, `Cannot restore: ${errorMessage(e)}`), 'error');
  }
  useUi.getState().setRecovery(null);
}

export function discardRecovery(): void {
  useUi.getState().setRecovery(null);
  void host.recovery.clear();
}

/**
 * Runs the model once (in the backend's subprocess) to declare the output
 * shapes of code blocks and opaque modules, which inference cannot see.
 */
export async function fillShapesByRunning(): Promise<void> {
  try {
    const observed = await probeShapes(useStudio.getState().project);
    useStudio.getState().applyShapes(observed);
    notify(
      observed.length > 0
        ? tr(`已补全 ${observed.length} 个输出的形状`, `Filled in ${observed.length} output shape${observed.length === 1 ? '' : 's'}`)
        : tr('没有需要补全的形状', 'No shapes to fill in'),
      'success',
    );
  } catch (e) {
    notify(tr(`运行失败：${errorMessage(e)}`, `Running failed: ${errorMessage(e)}`), 'error');
  }
}

/** Whether any output in the project is hidden from inference (code blocks, opaque modules). */
export const hasHiddenShapes = (): boolean => {
  const { project } = useStudio.getState();
  return Object.values(project.modules).some((m) =>
    Object.values(m.nodes).some((n) => n.kind === 'code' || (n.kind === 'module' && !!project.modules[n.type]?.opaque)),
  );
};

// ---------------------------------------------------------------- modules

/** A class name check for the prompt: PascalCase and not taken. */
function classNameError(name: string, except?: ModuleId): string | undefined {
  if (!isValidClassName(name)) return tr('类名须是以大写字母开头的 Python 标识符', 'Use a Python identifier starting with a capital letter');
  const taken = Object.values(useStudio.getState().project.modules).some((m) => m.className === name && m.id !== except);
  return taken ? tr(`类名 ${name} 已被使用`, `${name} is already used`) : undefined;
}

/** Ctrl+G: packs the selected nodes into a new module (PRD F4.1). */
export function packSelection(): void {
  const selected = useStudio.getState().flowNodes.filter((n) => n.selected);
  if (selected.length === 0) return notify(tr('先选中要打包的节点', 'Select the nodes to pack first'));
  useUi.getState().setPrompt({
    title: tr('打包为模块', 'Pack as module'),
    label: tr('新模块的类名', 'Class name of the new module'),
    initial: uniqueClassName(useStudio.getState().project, 'Block'),
    validate: (v) => classNameError(v),
    submit: (v) => attempt(() => useStudio.getState().packSelection(v)) && notify(tr(`已创建模块 ${v}`, `Created module ${v}`), 'success'),
  });
}

/** Ctrl+Shift+G: inlines the selected module instance. */
export function unpackSelection(nodeId?: NodeId): void {
  const studio = useStudio.getState();
  const id = nodeId ?? studio.flowNodes.find((n) => n.selected)?.id;
  const node = id ? studio.project.modules[studio.view.moduleId]?.nodes[id] : undefined;
  if (node?.kind !== 'module') return notify(tr('选中一个模块实例再解包', 'Select a module instance to unpack'));
  attempt(() => studio.unpack(node.id));
}

export function newModule(): void {
  useUi.getState().setPrompt({
    title: tr('新建模块', 'New module'),
    label: tr('类名', 'Class name'),
    initial: uniqueClassName(useStudio.getState().project, 'Block'),
    validate: (v) => classNameError(v),
    submit: (v) => useStudio.getState().createModule(v),
  });
}

export function renameModulePrompt(moduleId: ModuleId): void {
  const mod = useStudio.getState().project.modules[moduleId];
  if (!mod) return;
  useUi.getState().setPrompt({
    title: tr('重命名模块', 'Rename module'),
    label: tr('类名', 'Class name'),
    initial: mod.className,
    validate: (v) => classNameError(v, moduleId),
    submit: (v) => useStudio.getState().setClassName(v, moduleId),
  });
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
    case 'exportOptions':
      return useUi.getState().openDialog('exportOptions');
    case 'exportImage':
      return useUi.getState().openDialog('exportImage');
    case 'undo':
      return studio.undo();
    case 'redo':
      return studio.redo();
    case 'autoLayout':
      return studio.requestLayout();
    case 'packModule':
      return packSelection();
    case 'unpackModule':
      return unpackSelection();
    case 'goUp':
      return studio.goUp();
    case 'newModule':
      return newModule();
    case 'replay':
      return useReplay.getState().open ? useReplay.getState().close() : useReplay.getState().start();
    case 'toggleSnap':
      return useUi.getState().toggleSnap();
    case 'locale:zh':
    case 'locale:en':
      return useLocale.getState().setLocale(command === 'locale:zh' ? 'zh' : 'en');
    default:
      if (command.startsWith('openRecent:')) void openDocument(command.slice('openRecent:'.length));
  }
}
