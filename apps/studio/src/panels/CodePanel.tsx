import { findPath, type ModuleId, type NodeId } from '@neuroloom/core';
import { useReactFlow } from '@xyflow/react';
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { runCommand } from '../commands';
import { tr } from '../i18n';
import { tokenizeLine } from '../lib/highlight';
import { useSelectedNodeId, useStudio } from '../store';

export function CodePanel({ tabs }: { tabs: ReactNode }) {
  const { code, lineMap, classLines, todos } = useStudio((s) => s.codegen);
  const project = useStudio((s) => s.project);
  const moduleId = useStudio((s) => s.view.moduleId);
  const className = project.modules[project.entry]!.className;
  const selected = useSelectedNodeId();
  const [copied, setCopied] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);

  const lines = useMemo(() => code.replace(/\n$/, '').split('\n').map(tokenizeLine), [code]);
  const nodeOfLine = useMemo(() => {
    const map = new Map<number, { moduleId: ModuleId; nodeId: NodeId }>();
    for (const [mod, nodes] of Object.entries(lineMap)) {
      for (const [nodeId, nums] of Object.entries(nodes)) for (const n of nums) if (!map.has(n)) map.set(n, { moduleId: mod, nodeId });
    }
    return map;
  }, [lineMap]);
  const { fitView } = useReactFlow();

  /** Clicking a line selects its node, switching to its module first when needed (PRD F5.2). */
  const reveal = (line: number) => {
    const target = nodeOfLine.get(line);
    if (!target) return;
    const studio = useStudio.getState();
    if (target.moduleId !== studio.view.moduleId) studio.openModule(target.moduleId, findPath(studio.project, target.moduleId));
    useStudio.getState().selectNode(target.nodeId);
    setTimeout(() => void fitView({ nodes: [{ id: target.nodeId }], duration: 300, maxZoom: 1.2, padding: 0.6 }), 80);
  };
  const highlighted = useMemo(() => new Set(selected ? (lineMap[moduleId]?.[selected] ?? []) : []), [selected, lineMap, moduleId]);

  // Bring the selected node's code into view, or the class shown on the canvas.
  useEffect(() => {
    const first = highlighted.size > 0 ? Math.min(...highlighted) : classLines[moduleId];
    if (first === undefined || !Number.isFinite(first)) return;
    scroller.current?.querySelector(`[data-line="${first}"]`)?.scrollIntoView({ block: highlighted.size > 0 ? 'nearest' : 'start', behavior: 'smooth' });
  }, [highlighted, classLines, moduleId]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  const button = 'rounded-md border border-line px-2.5 py-1 text-[12px] hover:bg-surface-2';
  return (
    <section className="flex min-h-0 flex-col border-t border-line bg-surface">
      <header className="flex items-center gap-3 border-b border-line px-4 py-2">
        {tabs}
        <span className="font-mono text-[11px] text-muted">
          {className.toLowerCase()}.py · {tr('实时生成', 'live')}
        </span>
        {todos > 0 && <span className="text-[11px] text-warning">{tr(`${todos} 处 TODO`, `${todos} TODO${todos === 1 ? '' : 's'}`)}</span>}
        <span className="flex-1" />
        <button type="button" className={button} onClick={copy}>
          {copied ? tr('已复制', 'Copied') : tr('复制', 'Copy')}
        </button>
        <button type="button" className={button} onClick={() => runCommand('exportPy')}>
          {tr('导出 .py', 'Export .py')}
        </button>
      </header>
      <div ref={scroller} className="min-h-0 flex-1 overflow-auto bg-code py-2">
        <pre className="nl-code m-0">
          {lines.map((tokens, i) => (
            <div
              key={i}
              data-line={i + 1}
              onClick={() => reveal(i + 1)}
              title={nodeOfLine.has(i + 1) ? tr('在画布中定位该节点', 'Show this node on the canvas') : undefined}
              className={`px-3 ${highlighted.has(i + 1) ? 'hl' : ''} ${nodeOfLine.has(i + 1) ? 'cursor-pointer hover:bg-surface-2' : ''}`}
            >
              <span className="ln">{i + 1}</span>
              {tokens.map((t, j) => (
                <span key={j} className={t.kind ? `tok-${t.kind}` : undefined}>
                  {t.text}
                </span>
              ))}
              {tokens.length === 0 && ' '}
            </div>
          ))}
        </pre>
      </div>
    </section>
  );
}
