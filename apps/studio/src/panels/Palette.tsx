import { allSpecs, type Category, type NodeSpec } from '@neuroloom/core';
import { useReactFlow } from '@xyflow/react';
import { useMemo, useState, type CSSProperties, type DragEvent } from 'react';
import { NODE_DRAG_TYPE } from '../canvas/FlowCanvas';
import { CATEGORY_LABEL, categoryColor, glyphOf } from '../canvas/glyph';
import { useStudio } from '../store';

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

export function Palette() {
  const addNodeAt = useStudio((s) => s.addNodeAt);
  const { screenToFlowPosition, getNodes } = useReactFlow();
  const [query, setQuery] = useState('');
  const specs = useMemo(() => allSpecs().filter((s) => matches(s, query)), [query]);

  const addAtCenter = (spec: NodeSpec) => {
    const rect = document.querySelector('.react-flow')?.getBoundingClientRect();
    const p = rect
      ? screenToFlowPosition({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 })
      : { x: 0, y: 0 };
    // Step down until the spot is free, so a new node never hides an existing one.
    const pos = { x: Math.round(p.x - 84), y: Math.round(p.y - 30) };
    const NEW_HEIGHT = 90;
    const blocker = (y: number) =>
      getNodes().find((n) => {
        const h = n.measured?.height ?? 80;
        return Math.abs(n.position.x - pos.x) < 190 && y < n.position.y + h + 16 && n.position.y < y + NEW_HEIGHT;
      });
    for (let b = blocker(pos.y); b; b = blocker(pos.y)) pos.y = b.position.y + (b.measured?.height ?? 80) + 24;
    addNodeAt(spec.type, pos);
  };

  const onDragStart = (e: DragEvent, spec: NodeSpec) => {
    e.dataTransfer.setData(NODE_DRAG_TYPE, spec.type);
    e.dataTransfer.effectAllowed = 'copy';
  };

  return (
    <aside className="flex min-h-0 flex-col border-r border-line bg-surface">
      <div className="px-3 pt-3 pb-2">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={`搜索 ${allSpecs().length} 个节点…`}
          className="w-full rounded-md border border-line bg-bg px-2 py-1.5 text-[12.5px] outline-none focus:border-primary"
        />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-4">
        {specs.length === 0 && <p className="px-2 py-4 text-[12px] text-muted">没有匹配的节点</p>}
        {ORDER.map((category) => {
          const group = specs.filter((s) => s.category === category);
          if (group.length === 0) return null;
          return (
            <section key={category} className="mb-2">
              <div className="sticky top-0 bg-surface px-2 py-1 text-[11px] text-muted">
                {CATEGORY_LABEL[category]} <span className="opacity-60">{group.length}</span>
              </div>
              {group.map((spec) => (
                <button
                  key={spec.type}
                  type="button"
                  draggable
                  onDragStart={(e) => onDragStart(e, spec)}
                  onClick={() => addAtCenter(spec)}
                  title={`${spec.summary.zh}\n${spec.type}`}
                  className="group flex w-full cursor-grab items-center gap-2.5 rounded-lg px-2 py-1 text-left hover:bg-surface-2 active:cursor-grabbing"
                >
                  <span
                    className="grid size-6 flex-none place-items-center rounded-md font-mono text-[10px] font-bold text-white"
                    style={{ background: categoryColor(spec) } as CSSProperties}
                  >
                    {glyphOf(spec)}
                  </span>
                  <span className="min-w-0">
                    <span className="block truncate font-mono text-[12.5px] leading-tight">{spec.displayName.en}</span>
                    <span className="block truncate text-[11px] leading-tight text-muted">{spec.displayName.zh}</span>
                  </span>
                </button>
              ))}
            </section>
          );
        })}
      </div>
      <p className="border-t border-line px-4 py-3 text-[11px] leading-relaxed text-muted">拖到画布，或单击添加到视图中心。选中后按 Delete 删除。</p>
    </aside>
  );
}
