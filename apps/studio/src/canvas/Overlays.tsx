// Things drawn over the canvas: the breadcrumb (PRD F4.2), the selection bar
// (align, distribute, pack), the context menu, and the source of opaque modules.
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { packSelection, renameModulePrompt, unpackSelection, attempt } from '../commands';
import { tr } from '../i18n';
import type { AlignMode } from '../lib/align';
import { useCurrentModule, useSelectionCount, useStudio } from '../store';
import { useUi } from '../ui';

/** `Model / encoder (Block) / attn (Attention)`; each step returns to that level. */
export function Breadcrumb() {
  const project = useStudio((s) => s.project);
  const view = useStudio((s) => s.view);
  const goToDepth = useStudio((s) => s.goToDepth);
  const openModule = useStudio((s) => s.openModule);
  const entry = project.modules[project.entry]!;

  const steps: { label: string; detail?: string; go: () => void }[] = [{ label: entry.className, go: () => goToDepth(0) }];
  let moduleId = project.entry;
  view.path.forEach((nodeId, i) => {
    const node = project.modules[moduleId]?.nodes[nodeId];
    const def = node ? project.modules[node.type] : undefined;
    if (!node || !def) return;
    steps.push({ label: node.name, detail: def.sequential ? 'nn.Sequential' : def.className, go: () => goToDepth(i + 1) });
    moduleId = def.id;
  });
  // A module opened from the outline that the entry does not use.
  if (view.path.length === 0 && view.moduleId !== project.entry) {
    const mod = project.modules[view.moduleId]!;
    steps.length = 0;
    steps.push({ label: mod.className, detail: tr('未被使用', 'unused'), go: () => openModule(mod.id) });
  }

  return (
    <nav aria-label={tr('模块层级', 'Module path')} className="pointer-events-auto flex max-w-full items-center gap-1 overflow-x-auto rounded-lg border border-line bg-surface/95 px-2 py-1 text-[12.5px] shadow-sm backdrop-blur">
      {steps.map((s, i) => (
        <span key={i} className="flex items-center gap-1 whitespace-nowrap">
          {i > 0 && <span className="text-muted">/</span>}
          <button
            type="button"
            onClick={s.go}
            disabled={i === steps.length - 1}
            className="rounded px-1.5 py-0.5 font-mono enabled:hover:bg-surface-2 disabled:font-semibold"
          >
            {s.label}
            {s.detail && <span className="ml-1 font-sans text-[11px] text-muted">{s.detail}</span>}
          </button>
        </span>
      ))}
      {(view.path.length > 0 || view.moduleId !== project.entry) && (
        <button
          type="button"
          className="ml-1 rounded px-1.5 py-0.5 text-[11.5px] text-muted hover:bg-surface-2"
          title="Alt+←"
          onClick={() => useStudio.getState().goUp()}
        >
          {tr('↑ 上一级', '↑ Up')}
        </button>
      )}
    </nav>
  );
}

const ALIGN: { mode: AlignMode; icon: string; zh: string; en: string }[] = [
  { mode: 'left', icon: '⇤', zh: '左对齐', en: 'Align left' },
  { mode: 'centerX', icon: '↔', zh: '水平居中', en: 'Center horizontally' },
  { mode: 'right', icon: '⇥', zh: '右对齐', en: 'Align right' },
  { mode: 'top', icon: '⤒', zh: '顶部对齐', en: 'Align top' },
  { mode: 'centerY', icon: '↕', zh: '垂直居中', en: 'Center vertically' },
  { mode: 'bottom', icon: '⤓', zh: '底部对齐', en: 'Align bottom' },
  { mode: 'distributeX', icon: '⫼', zh: '水平等距分布', en: 'Distribute horizontally' },
  { mode: 'distributeY', icon: '☰', zh: '垂直等距分布', en: 'Distribute vertically' },
];

/** Shown while two or more nodes are selected. */
export function SelectionBar() {
  const count = useSelectionCount();
  const align = useStudio((s) => s.alignSelection);
  if (count < 2) return null;
  const button = 'grid size-7 place-items-center rounded-md text-[14px] hover:bg-surface-2';
  return (
    <div role="toolbar" aria-label={tr('对齐与分布', 'Align and distribute')} className="pointer-events-auto flex items-center gap-0.5 rounded-lg border border-line bg-surface px-1 py-1 shadow-md">
      <span className="px-2 text-[11.5px] text-muted">{tr(`已选 ${count} 个`, `${count} selected`)}</span>
      {ALIGN.map((a, i) => (
        <span key={a.mode} className="flex items-center">
          {(i === 3 || i === 6) && <span className="mx-0.5 h-4 w-px bg-line" />}
          <button type="button" className={button} title={tr(a.zh, a.en)} aria-label={tr(a.zh, a.en)} onClick={() => align(a.mode)}>
            {a.icon}
          </button>
        </span>
      ))}
      <span className="mx-0.5 h-4 w-px bg-line" />
      <button type="button" className="rounded-md px-2 py-1 text-[12px] hover:bg-surface-2" title="Ctrl+G" onClick={packSelection}>
        {tr('打包为模块', 'Pack as module')}
      </button>
    </div>
  );
}

function MenuItem({ children, onClick, keys, disabled }: { children: ReactNode; onClick: () => void; keys?: string; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="menuitem"
      disabled={disabled}
      onClick={() => {
        useUi.getState().setContextMenu(null);
        onClick();
      }}
      className="flex w-full items-center justify-between gap-6 px-3 py-1.5 text-left text-[13px] enabled:hover:bg-surface-2 disabled:opacity-40"
    >
      <span>{children}</span>
      {keys && <span className="font-mono text-[11px] text-muted">{keys}</span>}
    </button>
  );
}

export function ContextMenuView() {
  const menu = useUi((s) => s.contextMenu);
  const mod = useCurrentModule();
  const project = useStudio((s) => s.project);
  const root = useRef<HTMLDivElement>(null);
  const count = useSelectionCount();

  useEffect(() => {
    if (!menu) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === 'Escape' : !root.current?.contains(e.target as Node)) useUi.getState().setContextMenu(null);
    };
    window.addEventListener('mousedown', close);
    window.addEventListener('keydown', close);
    return () => {
      window.removeEventListener('mousedown', close);
      window.removeEventListener('keydown', close);
    };
  }, [menu]);
  if (!menu) return null;

  const node = menu.nodeId ? mod.nodes[menu.nodeId] : undefined;
  const def = node?.kind === 'module' ? project.modules[node.type] : undefined;
  const studio = useStudio.getState();
  const mod2 = navigator.platform.toLowerCase().includes('mac') ? '⌘' : 'Ctrl+';
  return (
    <div
      ref={root}
      role="menu"
      className="fixed z-50 w-60 rounded-lg border border-line bg-surface py-1 shadow-xl"
      style={{ left: Math.min(menu.screen.x, window.innerWidth - 250), top: Math.min(menu.screen.y, window.innerHeight - 260) }}
    >
      {def && (
        <>
          <MenuItem onClick={() => studio.enterModule(node!.id)}>{def.opaque ? tr('查看源代码', 'View source') : tr('进入模块', 'Open module')}</MenuItem>
          {!def.sequential && <MenuItem onClick={() => renameModulePrompt(def.id)}>{tr(`重命名类 ${def.className}…`, `Rename class ${def.className}…`)}</MenuItem>}
          <MenuItem onClick={() => attempt(() => studio.makeUnique(node!.id))}>{tr('转为独立副本', 'Make independent copy')}</MenuItem>
          <MenuItem onClick={() => unpackSelection(node!.id)} keys={`${mod2}Shift+G`} disabled={!!def.opaque || !!node!.repeat}>
            {tr('解包到当前层', 'Unpack here')}
          </MenuItem>
          <div className="my-1 border-t border-line" />
        </>
      )}
      <MenuItem onClick={packSelection} keys={`${mod2}G`} disabled={count === 0 || !!mod.sequential}>
        {tr('打包为模块…', 'Pack as module…')}
      </MenuItem>
      {count > 0 && (
        <MenuItem onClick={() => studio.deleteNodes(studio.flowNodes.filter((n) => n.selected).map((n) => n.id))} keys="Delete">
          {tr('删除', 'Delete')}
        </MenuItem>
      )}
      {!node && (
        <MenuItem onClick={() => useStudio.getState().requestLayout()} keys={`${mod2}L`}>
          {tr('自动布局', 'Auto layout')}
        </MenuItem>
      )}
    </div>
  );
}

/** An opaque module has no graph to show; its Python source is shown (and editable) instead. */
export function SourceView() {
  const mod = useCurrentModule();
  const setText = useStudio((s) => s.setText);
  const [draft, setDraft] = useState<string | null>(null);
  if (!mod.opaque) return null;
  const external = !mod.opaque.source;
  return (
    <div className="absolute inset-0 z-10 flex flex-col bg-bg/95 p-6 pt-16">
      <div className="mx-auto flex min-h-0 w-full max-w-4xl flex-1 flex-col rounded-xl border border-line bg-surface shadow-lg">
        <header className="border-b border-line px-5 py-3">
          <h2 className="font-mono text-[14px] font-semibold">{mod.className}</h2>
          <p className="mt-1 text-[12px] text-muted">
            {external
              ? tr('来自其他库的模块，由文件头代码导入。NeuroLoom 只记录它的构造参数和调用方式；形状需要运行后才知道。', 'A module from another library, imported by the file preamble. NeuroLoom records only its arguments and calls; its shapes are known only when the model runs.')
              : tr('导入时无法结构化的类（级别 L3），按原样保留为 Python 源代码，导出时一字不差地写回。', 'A class the importer could not structure (level L3), kept as Python source and written back unchanged on export.')}
          </p>
        </header>
        {!external && (
          <textarea
            className="min-h-0 flex-1 resize-none bg-code px-5 py-4 font-mono text-[12.5px] leading-relaxed outline-none"
            value={draft ?? mod.opaque.source}
            spellCheck={false}
            onChange={(e) => {
              setDraft(e.target.value);
              setText('source', e.target.value);
            }}
            onBlur={() => setDraft(null)}
          />
        )}
      </div>
    </div>
  );
}
