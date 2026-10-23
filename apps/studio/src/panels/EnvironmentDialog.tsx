import type { PythonCandidateInfo } from '@neuroloom/host-api';
import { useEffect, useRef, useState } from 'react';
import { errorMessage, host } from '../host';
import { tr } from '../i18n';
import { rpc, type EnvInfo } from '../lib/rpc';

const STATUS = (): Record<PythonCandidateInfo['status'], { label: string; tone: string }> => ({
  ready: { label: tr('可用', 'ready'), tone: 'text-success' },
  'no-torch': { label: tr('未安装 torch', 'no torch'), tone: 'text-warning' },
  'old-torch': { label: tr('torch 版本过低（需 ≥ 2.4）', 'torch too old (needs ≥ 2.4)'), tone: 'text-warning' },
  'old-python': { label: tr('Python 版本过低（需 ≥ 3.10）', 'Python too old (needs ≥ 3.10)'), tone: 'text-error' },
  broken: { label: tr('无法运行', 'does not run'), tone: 'text-error' },
});

const SOURCE = (): Record<PythonCandidateInfo['source'], string> => ({
  configured: tr('已选择', 'selected'),
  env: 'NEUROLOOM_PYTHON',
  managed: tr('NeuroLoom 创建', 'created by NeuroLoom'),
  project: tr('项目 venv', 'project venv'),
  conda: 'conda',
  launcher: tr('py 启动器', 'py launcher'),
  path: 'PATH',
});

/** Python environment picker and one-click setup (PRD F10.1). */
export function EnvironmentDialog({ onClose, onChanged }: { onClose: () => void; onChanged: () => void }) {
  const [current, setCurrent] = useState<EnvInfo | null>(null);
  const [candidates, setCandidates] = useState<PythonCandidateInfo[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const logEnd = useRef<HTMLDivElement>(null);

  const refresh = async () => {
    setError(null);
    rpc<EnvInfo>('env_info').then(setCurrent, () => setCurrent(null));
    if (host.env) setCandidates(await host.env.list());
  };

  useEffect(() => {
    void refresh();
    return host.env?.onProgress((line) => setLog((l) => [...l.slice(-400), line]));
  }, []);

  useEffect(() => logEnd.current?.scrollIntoView({ block: 'end' }), [log]);

  const run = async (label: string, action: () => Promise<unknown>) => {
    setBusy(label);
    setError(null);
    try {
      await action();
      await refresh();
      onChanged();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(null);
    }
  };

  const button = 'rounded-md border border-line px-2.5 py-1 text-[12px] hover:bg-surface-2 disabled:opacity-50';
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-6" onMouseDown={() => !busy && onClose()}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="env-title"
        className="flex max-h-full w-[min(760px,100%)] flex-col rounded-xl border border-line bg-surface shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="border-b border-line px-5 py-4">
          <h2 id="env-title" className="text-[15px] font-semibold">
            {tr('Python 环境', 'Python environment')}
          </h2>
          <p className="mt-1 text-[12px] text-muted">
            {tr(
              '编辑、形状推断和导出不需要 Python；从代码导入需要一个装有 PyTorch 2.4+ 的环境，CPU 版即可。',
              'Editing, shape inference and export need no Python; importing code needs an environment with PyTorch 2.4+ (CPU is enough).',
            )}
          </p>
        </header>

        <div className="min-h-0 flex-1 overflow-auto px-5 py-4">
          <section className="mb-4 rounded-lg border border-line bg-bg px-4 py-3 text-[12.5px]">
            <div className="mb-1 text-[11px] font-semibold text-muted">{tr('当前使用', 'In use')}</div>
            {current ? (
              <>
                <div className="font-mono break-all">{current.executable}</div>
                <div className="mt-1 text-muted">
                  Python {current.python} · torch {current.torch ?? tr('未安装', 'not installed')}
                  {current.cuda ? tr(' · CUDA 可用', ' · CUDA available') : ''}
                </div>
              </>
            ) : (
              <div className="text-muted">{tr('后端未连接', 'Backend not connected')}</div>
            )}
          </section>

          {host.env ? (
            <>
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-[12px] font-semibold">{tr('本机找到的解释器', 'Interpreters on this machine')}</h3>
                <button type="button" className={button} disabled={!!busy} onClick={() => void refresh()}>
                  {tr('重新扫描', 'Rescan')}
                </button>
              </div>
              {candidates === null ? (
                <p className="text-[12px] text-muted">{tr('正在扫描…', 'Scanning…')}</p>
              ) : candidates.length === 0 ? (
                <p className="text-[12px] text-muted">
                  {tr('没有找到 Python。可以先安装 Python 3.10+，再一键创建环境。', 'No Python found. Install Python 3.10+, then create an environment here.')}
                </p>
              ) : (
                <ul className="divide-y divide-line rounded-lg border border-line">
                  {candidates.map((c) => {
                    const inUse = current?.executable && c.path.toLowerCase() === current.executable.toLowerCase();
                    return (
                      <li key={c.path} className="flex items-center gap-3 px-3 py-2">
                        <div className="min-w-0 flex-1">
                          <div className="truncate font-mono text-[12px]" title={c.path}>
                            {c.path}
                          </div>
                          <div className="text-[11px] text-muted">
                            {SOURCE()[c.source]}
                            {c.version && ` · Python ${c.version}`}
                            {c.torch && ` · torch ${c.torch}`}
                            {' · '}
                            <span className={STATUS()[c.status].tone}>{STATUS()[c.status].label}</span>
                          </div>
                        </div>
                        {inUse ? (
                          <span className="text-[12px] text-success">{tr('使用中', 'In use')}</span>
                        ) : (
                          <button
                            type="button"
                            className={button}
                            disabled={!!busy || c.status !== 'ready'}
                            onClick={() => void run('switch', () => host.env!.select(c.path))}
                          >
                            {tr('使用', 'Use')}
                          </button>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}

              <section className="mt-4 rounded-lg border border-line px-4 py-3">
                <div className="flex items-center gap-3">
                  <div className="flex-1 text-[12px]">
                    <div className="font-semibold">{tr('一键创建环境', 'Create an environment')}</div>
                    <div className="text-muted">
                      {tr('在应用数据目录新建虚拟环境并安装 CPU 版 PyTorch，需下载约 200 MB。', 'Creates a virtual environment in the app data folder with CPU PyTorch (about 200 MB to download).')}
                    </div>
                  </div>
                  <button
                    type="button"
                    disabled={!!busy}
                    onClick={() => void run('create', () => host.env!.create())}
                    className="rounded-md bg-primary px-3 py-1.5 text-[13px] font-medium text-primary-fg hover:bg-primary-hover disabled:opacity-60"
                  >
                    {busy === 'create' ? tr('正在创建…', 'Creating…') : tr('创建', 'Create')}
                  </button>
                </div>
                {log.length > 0 && (
                  <pre className="mt-3 max-h-40 overflow-auto rounded-md bg-code px-3 py-2 font-mono text-[11px] leading-relaxed text-muted">
                    {log.join('\n')}
                    <div ref={logEnd} />
                  </pre>
                )}
              </section>
            </>
          ) : (
            <p className="text-[12px] leading-relaxed text-muted">
              {tr(
                '浏览器开发模式下，后端使用 NEUROLOOM_PYTHON 指定的解释器，否则使用 python/.venv。切换和创建环境需要在桌面版中进行。',
                'In the browser dev mode the backend uses the interpreter in NEUROLOOM_PYTHON, or python/.venv. Switching and creating environments needs the desktop app.',
              )}
            </p>
          )}
          {error && <pre className="mt-3 whitespace-pre-wrap rounded-md bg-error/10 px-3 py-2 text-[12px] text-error">{error}</pre>}
        </div>

        <footer className="flex justify-end border-t border-line px-5 py-3">
          <button type="button" disabled={!!busy} className={button} onClick={onClose}>
            {tr('关闭', 'Close')}
          </button>
        </footer>
      </div>
    </div>
  );
}
