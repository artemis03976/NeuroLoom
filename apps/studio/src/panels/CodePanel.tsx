import { entryModule } from '@neuroloom/core';
import { useEffect, useMemo, useRef, useState } from 'react';
import { tokenizeLine } from '../lib/highlight';
import { useSelectedNodeId, useStudio } from '../store';

export function downloadCode(code: string, className: string) {
  const url = URL.createObjectURL(new Blob([code], { type: 'text/x-python' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `${className.replace(/[^A-Za-z0-9_]/g, '').toLowerCase() || 'model'}.py`;
  a.click();
  URL.revokeObjectURL(url);
}

export function CodePanel() {
  const { code, lineMap, todos } = useStudio((s) => s.codegen);
  const className = useStudio((s) => entryModule(s.project).className);
  const selected = useSelectedNodeId();
  const [copied, setCopied] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);

  const lines = useMemo(() => code.replace(/\n$/, '').split('\n').map(tokenizeLine), [code]);
  const highlighted = useMemo(() => new Set(selected ? lineMap[selected] ?? [] : []), [selected, lineMap]);

  // Bring the selected node's code into view (PRD F5.2).
  useEffect(() => {
    const first = Math.min(...highlighted);
    if (!Number.isFinite(first)) return;
    scroller.current?.querySelector(`[data-line="${first}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [highlighted]);

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
        <span className="text-[12px] font-semibold">代码预览</span>
        <span className="font-mono text-[11px] text-muted">{className.toLowerCase()}.py · 实时生成</span>
        {todos > 0 && <span className="text-[11px] text-warning">{todos} 处 TODO</span>}
        <span className="flex-1" />
        <button type="button" className={button} onClick={copy}>
          {copied ? '已复制' : '复制'}
        </button>
        <button type="button" className={button} onClick={() => downloadCode(code, className)}>
          下载 .py
        </button>
      </header>
      <div ref={scroller} className="min-h-0 flex-1 overflow-auto bg-code py-2">
        <pre className="nl-code m-0">
          {lines.map((tokens, i) => (
            <div key={i} data-line={i + 1} className={`px-3 ${highlighted.has(i + 1) ? 'hl' : ''}`}>
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
