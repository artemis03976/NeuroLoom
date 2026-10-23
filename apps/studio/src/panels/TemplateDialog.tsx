import { analyzeModule, entryModule, type LoomProject } from '@neuroloom/core';
import { useMemo } from 'react';
import { host } from '../host';
import { formatCount } from '../lib/format';
import { blankProject } from '../lib/sample';
import { useStudio } from '../store';
import { TEMPLATES } from '../templates';

interface Choice {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  make: () => LoomProject;
}

/** "New": a blank project or one of the built-in templates (PRD F7.2). */
export function TemplateDialog({ onClose }: { onClose: () => void }) {
  const loadProject = useStudio((s) => s.loadProject);

  const choices = useMemo<(Choice & { nodes: number; params: number })[]>(() => {
    const list: Choice[] = [
      { id: 'blank', title: '空白项目', subtitle: 'Blank', description: '一个输入和一个输出，从零开始搭建。', make: blankProject },
      ...TEMPLATES.map((t) => ({
        id: t.id,
        title: t.titleZh,
        subtitle: t.title,
        description: t.descriptionZh,
        make: () => structuredClone(t.project),
      })),
    ];
    return list.map((c) => {
      const mod = entryModule(c.make());
      return { ...c, nodes: Object.keys(mod.nodes).length, params: analyzeModule(mod).totalParams };
    });
  }, []);

  const choose = (c: Choice) => {
    const project = c.make();
    loadProject(project, { name: c.id === 'blank' ? '未命名' : c.title, relayout: c.id !== 'blank' });
    void host.recovery.clear();
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-6" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="new-title"
        className="flex max-h-full w-[min(860px,100%)] flex-col rounded-xl border border-line bg-surface shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="border-b border-line px-5 py-4">
          <h2 id="new-title" className="text-[15px] font-semibold">
            新建项目
          </h2>
          <p className="mt-1 text-[12px] text-muted">从空白开始，或选一个模板。模板都可以继续编辑和导出。</p>
        </header>
        <div className="grid min-h-0 flex-1 grid-cols-2 gap-3 overflow-auto p-5 md:grid-cols-3">
          {choices.map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => choose(c)}
              className="flex flex-col rounded-lg border border-line p-4 text-left transition-colors hover:border-primary hover:bg-primary-soft"
            >
              <span className="text-[13.5px] font-semibold">{c.title}</span>
              <span className="font-mono text-[11px] text-muted">{c.subtitle}</span>
              <span className="mt-2 flex-1 text-[12px] leading-relaxed text-muted">{c.description}</span>
              <span className="mt-3 font-mono text-[11px] text-muted">
                {c.nodes} 个节点 · {formatCount(c.params)} 参数
              </span>
            </button>
          ))}
        </div>
        <footer className="flex justify-end border-t border-line px-5 py-3">
          <button type="button" className="rounded-md border border-line px-3 py-1.5 text-[13px] hover:bg-surface-2" onClick={onClose}>
            取消
          </button>
        </footer>
      </div>
    </div>
  );
}
