import { allSpecs, type Category, type NodeSpec } from '@neuroloom/core';
import { useReactFlow } from '@xyflow/react';
import type { CSSProperties, DragEvent } from 'react';
import { NODE_DRAG_TYPE } from '../canvas/FlowCanvas';
import { categoryColor, glyphOf } from '../canvas/glyph';
import { useStudio } from '../store';

const GROUPS: { category: Category; label: string }[] = [
  { category: 'io', label: '输入输出' },
  { category: 'linear', label: '线性' },
  { category: 'activation', label: '激活' },
];

export function Palette() {
  const addNodeAt = useStudio((s) => s.addNodeAt);
  const { screenToFlowPosition, getNodes } = useReactFlow();
  const specs = allSpecs();

  const addAtCenter = (spec: NodeSpec) => {
    const rect = document.querySelector('.react-flow')?.getBoundingClientRect();
    const p = rect
      ? screenToFlowPosition({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 })
      : { x: 0, y: 0 };
    // Step down until the spot is free, so a new node never hides an existing one.
    const pos = { x: Math.round(p.x - 84), y: Math.round(p.y - 30) };
    const taken = (y: number) =>
      getNodes().some((n) => Math.abs(n.position.x - pos.x) < 190 && Math.abs(n.position.y - y) < 90);
    while (taken(pos.y)) pos.y += 100;
    addNodeAt(spec.type, pos);
  };

  const onDragStart = (e: DragEvent, spec: NodeSpec) => {
    e.dataTransfer.setData(NODE_DRAG_TYPE, spec.type);
    e.dataTransfer.effectAllowed = 'copy';
  };

  return (
    <aside className="flex min-h-0 flex-col border-r border-line bg-surface">
      <div className="px-4 pt-4 pb-2 text-xs font-semibold tracking-wide text-muted">节点库</div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-4">
        {GROUPS.map(({ category, label }) => (
          <section key={category} className="mb-3">
            <div className="px-2 py-1 text-[11px] text-muted">{label}</div>
            {specs
              .filter((s) => s.category === category)
              .map((spec) => (
                <button
                  key={spec.type}
                  type="button"
                  draggable
                  onDragStart={(e) => onDragStart(e, spec)}
                  onClick={() => addAtCenter(spec)}
                  title={spec.summary.zh}
                  className="group flex w-full cursor-grab items-center gap-2.5 rounded-lg px-2 py-1.5 text-left hover:bg-surface-2 active:cursor-grabbing"
                >
                  <span
                    className="grid size-6 flex-none place-items-center rounded-md font-mono text-[10px] font-bold text-white"
                    style={{ background: categoryColor(spec) } as CSSProperties}
                  >
                    {glyphOf(spec)}
                  </span>
                  <span className="min-w-0">
                    <span className="block text-[13px] leading-tight">{spec.displayName.zh}</span>
                    <span className="block font-mono text-[11px] leading-tight text-muted">{spec.displayName.en}</span>
                  </span>
                </button>
              ))}
          </section>
        ))}
      </div>
      <p className="border-t border-line px-4 py-3 text-[11px] leading-relaxed text-muted">拖到画布，或单击添加到视图中心。选中后按 Delete 删除。</p>
    </aside>
  );
}
