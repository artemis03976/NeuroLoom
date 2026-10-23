import { CODE_TYPE, allSpecs } from '@neuroloom/core';
import { useReactFlow } from '@xyflow/react';
import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { categoryColor, glyphOf } from '../canvas/glyph';
import { attempt } from '../commands';
import { tr } from '../i18n';
import { moduleType, useStudio } from '../store';
import { useUi, type QuickAdd as QuickAddState } from '../ui';
import { matches, placeableModules } from './Palette';

const WIDTH = 300;
const MAX_RESULTS = 12;

interface Choice {
  type: string;
  glyph: string;
  color: string;
  title: string;
  subtitle: string;
}

/** Search-and-add popover (PRD F1.6): Tab, double-click on empty canvas, or an edge dropped on empty canvas. */
export function QuickAdd({ state }: { state: QuickAddState }) {
  const close = () => useUi.getState().setQuickAdd(null);
  const addNodeAt = useStudio((s) => s.addNodeAt);
  const project = useStudio((s) => s.project);
  const moduleId = useStudio((s) => s.view.moduleId);
  const { screenToFlowPosition } = useReactFlow();
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);

  // When wiring from a port, only offer nodes that can be connected on that side.
  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    const choices: Choice[] = [];
    for (const m of placeableModules(project, moduleId)) {
      const ports = m.interface;
      const usable = !state.link || (state.link.side === 'source' ? ports.inputs.length > 0 : ports.outputs.length > 0);
      if (usable && (!q || m.className.toLowerCase().includes(q))) {
        choices.push({ type: moduleType(m.id), glyph: m.opaque ? '⟨⟩' : 'M', color: 'var(--nl-cat-module)', title: m.className, subtitle: tr('模块', 'module') });
      }
    }
    for (const s of allSpecs()) {
      const usable = !state.link || (state.link.side === 'source' ? s.inputs.length > 0 : s.outputs.length > 0 || !!s.dynamicOutputs);
      if (usable && matches(s, query)) choices.push({ type: s.type, glyph: glyphOf(s), color: categoryColor(s), title: s.displayName.en, subtitle: s.displayName.zh });
    }
    if (!q || 'python code 代码块'.includes(q)) {
      choices.push({ type: CODE_TYPE, glyph: '{}', color: 'var(--nl-cat-code)', title: 'Python', subtitle: tr('代码块', 'code block') });
    }
    return choices.slice(0, MAX_RESULTS);
  }, [query, state.link, project, moduleId]);

  useEffect(() => setActive(0), [query]);
  useEffect(() => {
    const onDown = (e: MouseEvent) => !root.current?.contains(e.target as Node) && close();
    window.addEventListener('mousedown', onDown);
    return () => window.removeEventListener('mousedown', onDown);
  }, []);

  const choose = (choice: Choice | undefined) => {
    if (!choice) return;
    const at = screenToFlowPosition(state.screen);
    attempt(() => addNodeAt(choice.type, { x: Math.round(at.x), y: Math.round(at.y - 30) }, state.link));
    close();
  };

  const left = Math.min(state.screen.x, window.innerWidth - WIDTH - 12);
  const top = Math.min(state.screen.y, window.innerHeight - 380);
  return (
    <div
      ref={root}
      role="dialog"
      aria-label={tr('添加节点', 'Add node')}
      className="fixed z-50 rounded-lg border border-line bg-surface shadow-2xl"
      style={{ left, top, width: WIDTH }}
    >
      <input
        autoFocus
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') setActive((i) => Math.min(i + 1, results.length - 1));
          else if (e.key === 'ArrowUp') setActive((i) => Math.max(i - 1, 0));
          else if (e.key === 'Enter') choose(results[active]);
          else if (e.key === 'Escape') close();
          else return;
          e.preventDefault();
        }}
        placeholder={state.link ? tr('添加并连接节点…', 'Add and connect a node…') : tr('添加节点…', 'Add a node…')}
        className="w-full rounded-t-lg border-b border-line bg-transparent px-3 py-2 text-[13px] outline-none"
      />
      <ul role="listbox" className="max-h-80 overflow-y-auto py-1">
        {results.length === 0 && <li className="px-3 py-2 text-[12px] text-muted">{tr('没有匹配的节点', 'No matching nodes')}</li>}
        {results.map((choice, i) => (
          <li
            key={choice.type}
            role="option"
            aria-selected={i === active}
            onMouseEnter={() => setActive(i)}
            onMouseDown={(e) => {
              e.preventDefault();
              choose(choice);
            }}
            className={`flex cursor-pointer items-center gap-2.5 px-3 py-1.5 ${i === active ? 'bg-primary-soft' : ''}`}
          >
            <span
              className="grid size-5 flex-none place-items-center rounded font-mono text-[9.5px] font-bold text-white"
              style={{ background: choice.color } as CSSProperties}
            >
              {choice.glyph}
            </span>
            <span className="min-w-0 flex-1 truncate font-mono text-[12.5px]">{choice.title}</span>
            <span className="truncate text-[11px] text-muted">{choice.subtitle}</span>
          </li>
        ))}
      </ul>
      <div className="border-t border-line px-3 py-1.5 text-[10.5px] text-muted">{tr('↑↓ 选择 · Enter 添加 · Esc 关闭', '↑↓ choose · Enter add · Esc close')}</div>
    </div>
  );
}
