import { CODE_TYPE, allSpecs, dependsOn, instancesOf, type Category, type LoomProject, type ModuleId, type NodeId, type NodeSpec } from '@neuroloom/core';
import { useReactFlow } from '@xyflow/react';
import { useMemo, useState, type CSSProperties, type DragEvent } from 'react';
import { NODE_DRAG_TYPE } from '../canvas/FlowCanvas';
import { categoryColor, categoryLabel, glyphOf } from '../canvas/glyph';
import { attempt, newModule, renameModulePrompt } from '../commands';
import { pick, tr } from '../i18n';
import { blockType, moduleType, useStudio } from '../store';
import { BLOCKS } from '../templates';

const ORDER: Category[] = ['io', 'linear', 'conv', 'pool', 'norm', 'activation', 'dropout', 'recurrent', 'attention', 'tensor', 'math'];

/** Matches display names, the PyTorch type and its short name ("conv", "nn.Conv2d", "卷积" all find Conv2d). */
export function matches(spec: NodeSpec, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const haystack = [spec.displayName.en, spec.displayName.zh, spec.type, spec.type.replace(/^torch\./, ''), spec.codegen.fn ?? '']
    .join(' ')
    .toLowerCase();
  return q.split(/\s+/).every((word) => haystack.includes(word));
}

/** Modules that can be placed in `target` without it containing itself. */
export function placeableModules(project: LoomProject, target: ModuleId) {
  return Object.values(project.modules).filter((m) => !m.sequential && m.id !== project.entry && !dependsOn(project, m.id, target));
}

function useAddAtCenter() {
  const addNodeAt = useStudio((s) => s.addNodeAt);
  const { screenToFlowPosition, getNodes } = useReactFlow();
  return (type: string) => {
    const rect = document.querySelector('.react-flow')?.getBoundingClientRect();
    const p = rect ? screenToFlowPosition({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }) : { x: 0, y: 0 };
    // Step down until the spot is free, so a new node never hides an existing one.
    const pos = { x: Math.round(p.x - 84), y: Math.round(p.y - 30) };
    const NEW_HEIGHT = 90;
    const blocker = (y: number) =>
      getNodes().find((n) => {
        const h = n.measured?.height ?? 80;
        return Math.abs(n.position.x - pos.x) < 190 && y < n.position.y + h + 16 && n.position.y < y + NEW_HEIGHT;
      });
    for (let b = blocker(pos.y); b; b = blocker(pos.y)) pos.y = b.position.y + (b.measured?.height ?? 80) + 24;
    attempt(() => addNodeAt(type, pos));
  };
}

const dragType = (type: string) => (e: DragEvent) => {
  e.dataTransfer.setData(NODE_DRAG_TYPE, type);
  e.dataTransfer.effectAllowed = 'copy';
};

const itemClass = 'group flex w-full cursor-grab items-center gap-2.5 rounded-lg px-2 py-1 text-left hover:bg-surface-2 active:cursor-grabbing';

function Glyph({ text, color }: { text: string; color: string }) {
  return (
    <span className="grid size-6 flex-none place-items-center rounded-md font-mono text-[10px] font-bold text-white" style={{ background: color } as CSSProperties}>
      {text}
    </span>
  );
}

function NodeLibrary() {
  const addAtCenter = useAddAtCenter();
  const [query, setQuery] = useState('');
  const specs = useMemo(() => allSpecs().filter((s) => matches(s, query)), [query]);
  const blocks = useMemo(() => {
    const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return BLOCKS.filter((b) => {
      const haystack = [b.title, b.titleZh, b.project.modules[b.project.entry]!.className].join(' ').toLowerCase();
      return words.every((w) => haystack.includes(w));
    });
  }, [query]);
  const showCode = !query.trim() || ['code', 'python', '代码'].some((w) => w.includes(query.trim().toLowerCase()) || query.toLowerCase().includes(w));

  return (
    <>
      <div className="px-3 pt-2 pb-2">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={tr(`搜索 ${allSpecs().length} 个节点…`, `Search ${allSpecs().length} nodes…`)}
          className="w-full rounded-md border border-line bg-bg px-2 py-1.5 text-[12.5px] outline-none focus:border-primary"
        />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-4">
        {specs.length === 0 && blocks.length === 0 && !showCode && <p className="px-2 py-4 text-[12px] text-muted">{tr('没有匹配的节点', 'No matching nodes')}</p>}
        {ORDER.map((category) => {
          const group = specs.filter((s) => s.category === category);
          if (group.length === 0) return null;
          return (
            <section key={category} className="mb-2">
              <div className="sticky top-0 bg-surface px-2 py-1 text-[11px] text-muted">
                {categoryLabel(category)} <span className="opacity-60">{group.length}</span>
              </div>
              {group.map((spec) => (
                <button
                  key={spec.type}
                  type="button"
                  draggable
                  onDragStart={dragType(spec.type)}
                  onClick={() => addAtCenter(spec.type)}
                  title={[pick(spec.summary), spec.shapeDoc && `${tr('输出形状：', 'Output shape: ')}${pick(spec.shapeDoc)}`, spec.type].filter(Boolean).join('\n')}
                  className={itemClass}
                >
                  <Glyph text={glyphOf(spec)} color={categoryColor(spec)} />
                  <span className="min-w-0">
                    <span className="block truncate font-mono text-[12.5px] leading-tight">{spec.displayName.en}</span>
                    <span className="block truncate text-[11px] leading-tight text-muted">{spec.displayName.zh}</span>
                  </span>
                </button>
              ))}
            </section>
          );
        })}
        {blocks.length > 0 && (
          <section className="mb-2">
            <div className="sticky top-0 bg-surface px-2 py-1 text-[11px] text-muted">
              {tr('复合模块', 'Blocks')} <span className="opacity-60">{blocks.length}</span>
            </div>
            {blocks.map((b) => (
              <button
                key={b.id}
                type="button"
                draggable
                onDragStart={dragType(blockType(b.id))}
                onClick={() => addAtCenter(blockType(b.id))}
                title={tr(b.descriptionZh, b.description)}
                className={itemClass}
              >
                <Glyph text="M" color="var(--nl-cat-module)" />
                <span className="min-w-0">
                  <span className="block truncate font-mono text-[12.5px] leading-tight">{b.project.modules[b.project.entry]!.className}</span>
                  <span className="block truncate text-[11px] leading-tight text-muted">{tr(b.titleZh, b.title)}</span>
                </span>
              </button>
            ))}
          </section>
        )}
        {showCode && (
          <section className="mb-2">
            <div className="sticky top-0 bg-surface px-2 py-1 text-[11px] text-muted">{categoryLabel('code')}</div>
            <button
              type="button"
              draggable
              onDragStart={dragType(CODE_TYPE)}
              onClick={() => addAtCenter(CODE_TYPE)}
              title={tr('任意 Python 语句，原样写入 forward()', 'Any Python statements, written into forward() as is')}
              className={itemClass}
            >
              <Glyph text="{}" color="var(--nl-cat-code)" />
              <span className="min-w-0">
                <span className="block truncate font-mono text-[12.5px] leading-tight">Python</span>
                <span className="block truncate text-[11px] leading-tight text-muted">{tr('代码块', 'Code block')}</span>
              </span>
            </button>
          </section>
        )}
      </div>
    </>
  );
}

/** The module outline (PRD F4.2): where each module is used, and every definition. */
function ModuleOutline() {
  const project = useStudio((s) => s.project);
  const view = useStudio((s) => s.view);
  const openModule = useStudio((s) => s.openModule);
  const addAtCenter = useAddAtCenter();
  const placeable = new Set(placeableModules(project, view.moduleId).map((m) => m.id));

  const rows: { depth: number; path: NodeId[]; moduleId: ModuleId; label: string; detail: string }[] = [];
  const walk = (moduleId: ModuleId, path: NodeId[], depth: number) => {
    const mod = project.modules[moduleId];
    if (!mod || depth > 12) return;
    for (const node of Object.values(mod.nodes)) {
      const def = node.kind === 'module' ? project.modules[node.type] : undefined;
      if (!def) continue;
      const repeat = node.repeat ? ' ×' : '';
      rows.push({ depth, path: [...path, node.id], moduleId: def.id, label: node.name + repeat, detail: def.sequential ? 'Sequential' : def.className });
      if (!path.includes(node.id)) walk(def.id, [...path, node.id], depth + 1);
    }
  };
  walk(project.entry, [], 1);
  const entry = project.modules[project.entry]!;
  const defs = Object.values(project.modules).filter((m) => !m.sequential);
  const samePath = (p: NodeId[]) => p.length === view.path.length && p.every((id, i) => id === view.path[i]);

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-2 pt-2 pb-4">
      <div className="px-2 py-1 text-[11px] text-muted">{tr('层级', 'Hierarchy')}</div>
      <button
        type="button"
        onClick={() => openModule(project.entry, [])}
        className={`flex w-full items-center gap-2 rounded-md px-2 py-1 text-left font-mono text-[12.5px] hover:bg-surface-2 ${view.moduleId === project.entry ? 'bg-primary-soft' : ''}`}
      >
        <span className="text-[10px] text-muted">▣</span>
        {entry.className}
      </button>
      {rows.map((r) => (
        <button
          key={r.path.join('/')}
          type="button"
          onClick={() => openModule(r.moduleId, r.path)}
          style={{ paddingLeft: 8 + r.depth * 14 }}
          className={`flex w-full items-center gap-1.5 rounded-md py-0.5 pr-2 text-left text-[12px] hover:bg-surface-2 ${samePath(r.path) ? 'bg-primary-soft' : ''}`}
        >
          <span className="text-[9px] text-muted">└</span>
          <span className="truncate font-mono">{r.label}</span>
          <span className="truncate text-[11px] text-muted">{r.detail}</span>
        </button>
      ))}

      <div className="mt-3 flex items-center justify-between px-2 py-1 text-[11px] text-muted">
        <span>
          {tr('模块定义', 'Module definitions')} <span className="opacity-60">{defs.length}</span>
        </span>
        <button type="button" className="rounded px-1.5 py-0.5 text-primary hover:bg-surface-2" onClick={newModule}>
          {tr('+ 新建', '+ New')}
        </button>
      </div>
      {defs.map((m) => {
        const uses = instancesOf(project, m.id).length;
        const kind = m.id === project.entry ? tr('入口', 'entry') : m.opaque ? (m.opaque.source ? tr('源码', 'source') : tr('外部', 'external')) : '';
        const canPlace = placeable.has(m.id);
        return (
          <div key={m.id} className={`group flex items-center gap-1 rounded-lg pr-1 hover:bg-surface-2 ${view.moduleId === m.id ? 'bg-primary-soft' : ''}`}>
            <button
              type="button"
              draggable={canPlace}
              onDragStart={canPlace ? dragType(moduleType(m.id)) : undefined}
              onClick={() => openModule(m.id)}
              title={canPlace ? tr('单击打开，拖到画布上添加实例', 'Click to open, drag onto the canvas to add an instance') : tr('单击打开', 'Click to open')}
              className={`flex min-w-0 flex-1 items-center gap-2.5 px-2 py-1 text-left ${canPlace ? 'cursor-grab active:cursor-grabbing' : ''}`}
            >
              <Glyph text={m.opaque ? '⟨⟩' : 'M'} color="var(--nl-cat-module)" />
              <span className="min-w-0">
                <span className="block truncate font-mono text-[12.5px] leading-tight">{m.className}</span>
                <span className="block truncate text-[11px] leading-tight text-muted">
                  {[kind, m.id === project.entry ? '' : tr(`${uses} 处使用`, `${uses} use${uses === 1 ? '' : 's'}`)].filter(Boolean).join(' · ')}
                </span>
              </span>
            </button>
            {canPlace && (
              <button type="button" className="hidden rounded px-1 text-[13px] text-muted group-hover:block hover:text-fg" title={tr('添加到当前画布', 'Add to the canvas')} onClick={() => addAtCenter(moduleType(m.id))}>
                +
              </button>
            )}
            {!m.opaque && (
              <button type="button" className="hidden rounded px-1 text-[11px] text-muted group-hover:block hover:text-fg" title={tr('重命名', 'Rename')} onClick={() => renameModulePrompt(m.id)}>
                ✎
              </button>
            )}
            {uses === 0 && m.id !== project.entry && (
              <button
                type="button"
                className="hidden rounded px-1 text-[12px] text-muted group-hover:block hover:text-error"
                title={tr('删除未使用的模块', 'Delete this unused module')}
                onClick={() => attempt(() => useStudio.getState().deleteModule(m.id))}
              >
                ×
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function Palette() {
  const [tab, setTab] = useState<'nodes' | 'modules'>('nodes');
  const moduleCount = useStudio((s) => Object.values(s.project.modules).filter((m) => !m.sequential).length);
  const tabClass = (active: boolean) =>
    `flex-1 border-b-2 px-2 py-2 text-[12.5px] ${active ? 'border-primary font-semibold text-fg' : 'border-transparent text-muted hover:text-fg'}`;
  return (
    <aside className="flex min-h-0 flex-col border-r border-line bg-surface">
      <div role="tablist" className="flex border-b border-line">
        <button type="button" role="tab" aria-selected={tab === 'nodes'} className={tabClass(tab === 'nodes')} onClick={() => setTab('nodes')}>
          {tr('节点', 'Nodes')}
        </button>
        <button type="button" role="tab" aria-selected={tab === 'modules'} className={tabClass(tab === 'modules')} onClick={() => setTab('modules')}>
          {tr('模块', 'Modules')} <span className="text-[11px] text-muted">{moduleCount}</span>
        </button>
      </div>
      {tab === 'nodes' ? <NodeLibrary /> : <ModuleOutline />}
      <p className="border-t border-line px-4 py-3 text-[11px] leading-relaxed text-muted">
        {tab === 'nodes'
          ? tr('拖到画布，或单击添加到视图中心。选中后按 Delete 删除。', 'Drag onto the canvas, or click to add at the centre. Delete removes the selection.')
          : tr('双击模块节点进入；Ctrl+G 把选中节点打包为模块，Alt+← 返回上一级。', 'Double-click a module node to open it; Ctrl+G packs the selection, Alt+← goes up.')}
      </p>
    </aside>
  );
}
