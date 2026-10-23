import { parseProject } from '@neuroloom/core';
import { useState } from 'react';
import { errorMessage } from '../host';
import { rpc, RpcError, type ImportResult } from '../lib/rpc';
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

export function ImportDialog({ onClose }: { onClose: () => void }) {
  const loadProject = useStudio((s) => s.loadProject);
  const [source, setSource] = useState(SAMPLE_IMPORT_CODE);
  const [className, setClassName] = useState('');
  const [shapes, setShapes] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await rpc<ImportResult>('import_code', {
        source,
        className: className.trim() || null,
        inputShapes: parseShapes(shapes),
      });
      const project = parseProject(result.project);
      loadProject(project, { name: project.name, saved: false, relayout: true });
      notify(`已导入 ${project.name}`, 'success', result.warnings);
      onClose();
    } catch (e) {
      setError(e instanceof RpcError ? e.message : errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const field = 'rounded-md border border-line bg-bg px-2 py-1.5 font-mono text-[12.5px] outline-none focus:border-primary';
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-6" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="import-title"
        className="flex max-h-full w-[min(760px,100%)] flex-col rounded-xl border border-line bg-surface shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="border-b border-line px-5 py-4">
          <h2 id="import-title" className="text-[15px] font-semibold">
            从 PyTorch 代码导入
          </h2>
          <p className="mt-1 text-[12px] text-muted">
            粘贴一个 nn.Module 定义。代码会在独立的 Python 子进程中执行一次，用 torch.fx 追踪出数据流（M0 支持 Linear、ReLU、F.relu）。
          </p>
        </header>
        <div className="min-h-0 flex-1 overflow-auto px-5 py-4">
          <textarea
            className={`${field} h-72 w-full resize-y leading-relaxed`}
            value={source}
            onChange={(e) => setSource(e.target.value)}
            spellCheck={false}
          />
          <div className="mt-3 grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1 text-[12px] text-muted">
              类名（留空 = 最后定义的类）
              <input className={field} value={className} onChange={(e) => setClassName(e.target.value)} placeholder="Autoencoder" />
            </label>
            <label className="flex flex-col gap-1 text-[12px] text-muted">
              输入形状（留空 = 从首个 Linear 推断）
              <input className={field} value={shapes} onChange={(e) => setShapes(e.target.value)} placeholder="x=B,784" />
            </label>
          </div>
          {error && <pre className="mt-3 whitespace-pre-wrap rounded-md bg-error/10 px-3 py-2 text-[12px] text-error">{error}</pre>}
        </div>
        <footer className="flex justify-end gap-2 border-t border-line px-5 py-3">
          <button type="button" className="rounded-md border border-line px-3 py-1.5 text-[13px] hover:bg-surface-2" onClick={onClose}>
            取消
          </button>
          <button
            type="button"
            disabled={busy || !source.trim()}
            onClick={run}
            className="rounded-md bg-primary px-3 py-1.5 text-[13px] font-medium text-primary-fg hover:bg-primary-hover disabled:opacity-60"
          >
            {busy ? '正在追踪…' : '执行并导入'}
          </button>
        </footer>
      </div>
    </div>
  );
}
