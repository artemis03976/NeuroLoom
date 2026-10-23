import { getLocale, parseProject } from '@neuroloom/core';
import { useState } from 'react';
import { fillShapesByRunning, hasHiddenShapes } from '../commands';
import { errorMessage } from '../host';
import { tr } from '../i18n';
import { rpc, RpcError, type ImportReportEntry, type ImportResult } from '../lib/rpc';
import { SAMPLE_IMPORT_CODE } from '../lib/sample';
import { useStudio } from '../store';
import { notify } from '../ui';

/** "x=B,784; mask=B,16" -> { x: "B,784", mask: "B,16" } */
function parseShapes(text: string): Record<string, string> {
  const shapes: Record<string, string> = {};
  for (const part of text.split(/[;\n]/)) {
    const [name, dims] = part.split('=');
    if (name?.trim() && dims?.trim()) shapes[name.trim()] = dims.trim();
  }
  return shapes;
}

type Mode = 'static' | 'trace';

const LEVEL: Record<ImportReportEntry['level'], { zh: string; en: string; tone: string }> = {
  L1: { zh: '完整结构化', en: 'fully structured', tone: 'text-success' },
  'L1+L3': { zh: '部分保留为代码块', en: 'partly kept as code', tone: 'text-warning' },
  L3: { zh: '整个类保留为源码', en: 'kept as source', tone: 'text-error' },
};

function Report({ report, warnings }: { report: ImportReportEntry[]; warnings: string[] }) {
  return (
    <div className="space-y-3">
      <ul className="space-y-2">
        {report.map((r) => (
          <li key={r.className} className="rounded-md border border-line px-3 py-2">
            <div className="flex items-center justify-between gap-3">
              <span className="font-mono text-[13px] font-semibold">{r.className}</span>
              <span className={`text-[12px] ${LEVEL[r.level].tone}`}>
                {r.level} · {tr(LEVEL[r.level].zh, LEVEL[r.level].en)}
                {r.codeBlocks > 0 && tr(`，${r.codeBlocks} 个代码块`, `, ${r.codeBlocks} code block${r.codeBlocks === 1 ? '' : 's'}`)}
              </span>
            </div>
            {r.notes.map((n, i) => (
              <p key={i} className="mt-1 text-[12px] leading-relaxed text-muted">
                {n}
              </p>
            ))}
          </li>
        ))}
      </ul>
      {warnings.length > 0 && (
        <ul className="space-y-1 text-[12px] leading-relaxed text-warning">
          {warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ImportDialog({ onClose }: { onClose: () => void }) {
  const loadProject = useStudio((s) => s.loadProject);
  const [source, setSource] = useState(SAMPLE_IMPORT_CODE);
  const [className, setClassName] = useState('');
  const [shapes, setShapes] = useState('x=B,1,28,28');
  const [mode, setMode] = useState<Mode>('static');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<{ name: string; report: ImportReportEntry[]; warnings: string[] } | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await rpc<ImportResult>('import_code', {
        source,
        className: className.trim() || null,
        inputShapes: parseShapes(shapes),
        mode,
        locale: getLocale(),
      });
      const project = parseProject(result.project);
      loadProject(project, { name: project.name, saved: false, relayout: true });
      if (result.report) setDone({ name: project.name, report: result.report, warnings: result.warnings });
      else {
        notify(tr(`已导入 ${project.name}`, `Imported ${project.name}`), 'success', result.warnings);
        onClose();
      }
    } catch (e) {
      setError(e instanceof RpcError ? e.message : errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const field = 'rounded-md border border-line bg-bg px-2 py-1.5 font-mono text-[12.5px] outline-none focus:border-primary';
  const modeButton = (m: Mode, title: string, detail: string) => (
    <label className={`flex flex-1 cursor-pointer gap-2 rounded-md border px-3 py-2 ${mode === m ? 'border-primary bg-primary-soft' : 'border-line'}`}>
      <input type="radio" name="import-mode" checked={mode === m} onChange={() => setMode(m)} className="mt-0.5" />
      <span>
        <span className="block text-[12.5px] font-medium">{title}</span>
        <span className="block text-[11.5px] leading-relaxed text-muted">{detail}</span>
      </span>
    </label>
  );
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-6" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="import-title"
        className="flex max-h-full w-[min(780px,100%)] flex-col rounded-xl border border-line bg-surface shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="border-b border-line px-5 py-4">
          <h2 id="import-title" className="text-[15px] font-semibold">
            {done ? tr(`已导入 ${done.name}`, `Imported ${done.name}`) : tr('从 PyTorch 代码导入', 'Import from PyTorch code')}
          </h2>
          <p className="mt-1 text-[12px] text-muted">
            {done
              ? tr('每个类的导入方式如下。保留为代码块或源码的部分会原样导出。', 'How each class was imported. Parts kept as code or source are exported unchanged.')
              : tr('粘贴一个或多个 nn.Module 类。子模块、超参数表达式和 nn.ModuleList 循环都会保留；无法结构化的语句保留为代码块。', 'Paste one or more nn.Module classes. Submodules, hyperparameter expressions and nn.ModuleList loops are kept; statements that cannot be structured become code blocks.')}
          </p>
        </header>
        <div className="min-h-0 flex-1 overflow-auto px-5 py-4">
          {done ? (
            <Report report={done.report} warnings={done.warnings} />
          ) : (
            <>
              <textarea
                className={`${field} h-64 w-full resize-y leading-relaxed`}
                value={source}
                onChange={(e) => setSource(e.target.value)}
                spellCheck={false}
              />
              <div className="mt-3 grid grid-cols-2 gap-3">
                <label className="flex flex-col gap-1 text-[12px] text-muted">
                  {tr('入口类名（留空 = 自动选择）', 'Entry class (empty = automatic)')}
                  <input className={field} value={className} onChange={(e) => setClassName(e.target.value)} placeholder="Classifier" />
                </label>
                <label className="flex flex-col gap-1 text-[12px] text-muted">
                  {tr('输入形状（留空 = 从第一层推断）', 'Input shapes (empty = guess from the first layer)')}
                  <input className={field} value={shapes} onChange={(e) => setShapes(e.target.value)} placeholder="x=B,784" />
                </label>
              </div>
              <div className="mt-3 flex gap-3">
                {modeButton(
                  'static',
                  tr('静态解析（推荐）', 'Read the code (recommended)'),
                  tr('不执行代码；保留层级、变量名和超参数表达式。', 'Does not run the code; keeps hierarchy, names and expressions.'),
                )}
                {modeButton(
                  'trace',
                  tr('动态追踪', 'Trace by running'),
                  tr('在子进程中执行一次并用 torch.fx 追踪；适合循环构造的模型，层级会被展平。', 'Runs once in a subprocess under torch.fx; for models built in loops. Flattens the hierarchy.'),
                )}
              </div>
              {error && <pre className="mt-3 whitespace-pre-wrap rounded-md bg-error/10 px-3 py-2 text-[12px] text-error">{error}</pre>}
            </>
          )}
        </div>
        <footer className="flex justify-end gap-2 border-t border-line px-5 py-3">
          {done ? (
            <>
              {hasHiddenShapes() && (
                <button
                  type="button"
                  disabled={busy}
                  title={tr('在 Python 子进程中运行一次模型，声明代码块和外部模块的输出形状', 'Run the model once in a Python subprocess to declare the output shapes of code blocks and external modules')}
                  className="rounded-md border border-line px-3 py-1.5 text-[13px] hover:bg-surface-2 disabled:opacity-60"
                  onClick={() => {
                    setBusy(true);
                    void fillShapesByRunning().finally(() => {
                      setBusy(false);
                      onClose();
                    });
                  }}
                >
                  {busy ? tr('正在运行…', 'Running…') : tr('运行一次，补全形状', 'Run once to fill in shapes')}
                </button>
              )}
              <button type="button" className="rounded-md bg-primary px-3 py-1.5 text-[13px] font-medium text-primary-fg hover:bg-primary-hover" onClick={onClose}>
                {tr('完成', 'Done')}
              </button>
            </>
          ) : (
            <>
              <button type="button" className="rounded-md border border-line px-3 py-1.5 text-[13px] hover:bg-surface-2" onClick={onClose}>
                {tr('取消', 'Cancel')}
              </button>
              <button
                type="button"
                disabled={busy || !source.trim()}
                onClick={run}
                className="rounded-md bg-primary px-3 py-1.5 text-[13px] font-medium text-primary-fg hover:bg-primary-hover disabled:opacity-60"
              >
                {busy ? (mode === 'trace' ? tr('正在追踪…', 'Tracing…') : tr('正在解析…', 'Reading…')) : mode === 'trace' ? tr('执行并导入', 'Run and import') : tr('导入', 'Import')}
              </button>
            </>
          )}
        </footer>
      </div>
    </div>
  );
}
