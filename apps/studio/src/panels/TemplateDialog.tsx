import { analyzeProject, type LoomProject } from '@neuroloom/core';
import { useMemo } from 'react';
import { host } from '../host';
import { tr } from '../i18n';
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
      {
        id: 'blank',
        title: tr('空白项目', 'Blank project'),
        subtitle: 'Blank',
        description: tr('一个输入和一个输出，从零开始搭建。', 'One input and one output; build from scratch.'),
        make: blankProject,
      },
      ...TEMPLATES.map((t) => ({
        id: t.id,
        title: tr(t.titleZh, t.title),
        subtitle: t.title,
        description: tr(t.descriptionZh, t.description),
        make: () => structuredClone(t.project),
      })),
    ];
    return list.map((c) => {
      const project = c.make();
      const nodes = Object.values(project.modules).reduce((n, m) => n + Object.keys(m.nodes).length, 0);
      return { ...c, nodes, params: analyzeProject(project).entry.totalParams };
    });
  }, []);

  const choose = (c: Choice) => {
    const project = c.make();
    loadProject(project, { name: c.id === 'blank' ? tr('未命名', 'Untitled') : c.title, relayout: c.id !== 'blank' });
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
            {tr('新建项目', 'New project')}
          </h2>
          <p className="mt-1 text-[12px] text-muted">
            {tr('从空白开始，或选一个模板。模板都可以继续编辑和导出。', 'Start blank or from a template; every template can be edited and exported.')}
          </p>
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
                {tr(`${c.nodes} 个节点 · ${formatCount(c.params)} 参数`, `${c.nodes} nodes · ${formatCount(c.params)} parameters`)}
              </span>
            </button>
          ))}
        </div>
        <footer className="flex justify-end border-t border-line px-5 py-3">
          <button type="button" className="rounded-md border border-line px-3 py-1.5 text-[13px] hover:bg-surface-2" onClick={onClose}>
            {tr('取消', 'Cancel')}
          </button>
        </footer>
      </div>
    </div>
  );
}
