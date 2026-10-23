import type { MenuCommand } from '@neuroloom/host-api';
import { useEffect, useRef, useState } from 'react';
import { runCommand } from '../commands';
import { host } from '../host';
import { rpc, type EnvInfo } from '../lib/rpc';
import type { ThemeChoice } from '../lib/theme';
import { useDirty, useStudio } from '../store';
import { useUi } from '../ui';

type Backend = { state: 'checking' } | { state: 'ok'; env: EnvInfo } | { state: 'down'; reason: string };

const THEME_LABEL: Record<ThemeChoice, string> = { system: '跟随系统', light: '浅色', dark: '深色' };
const NEXT_THEME: Record<ThemeChoice, ThemeChoice> = { system: 'light', light: 'dark', dark: 'system' };
const MOD = navigator.platform.toLowerCase().includes('mac') ? '⌘' : 'Ctrl+';

const button = 'rounded-md px-2.5 py-1.5 text-[13px] hover:bg-surface-2 disabled:opacity-40 disabled:hover:bg-transparent';

function FileMenu() {
  const [open, setOpen] = useState(false);
  const [recent, setRecent] = useState<string[]>([]);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    void host.recent().then(setRecent);
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === 'Escape' : !root.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener('mousedown', close);
    window.addEventListener('keydown', close);
    return () => {
      window.removeEventListener('mousedown', close);
      window.removeEventListener('keydown', close);
    };
  }, [open]);

  const item = (label: string, command: MenuCommand, keys?: string) => (
    <button
      key={command}
      type="button"
      role="menuitem"
      className="flex w-full items-center justify-between gap-6 px-3 py-1.5 text-left text-[13px] hover:bg-surface-2"
      onClick={() => {
        setOpen(false);
        runCommand(command);
      }}
    >
      <span>{label}</span>
      {keys && <span className="font-mono text-[11px] text-muted">{keys}</span>}
    </button>
  );

  return (
    <div ref={root} className="relative">
      <button type="button" className={button} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        文件
      </button>
      {open && (
        <div role="menu" className="absolute top-full left-0 z-40 mt-1 w-64 rounded-lg border border-line bg-surface py-1 shadow-xl">
          {item('新建', 'new', `${MOD}N`)}
          {item('打开…', 'open', `${MOD}O`)}
          {recent.length > 0 && (
            <>
              <div className="px-3 pt-2 pb-1 text-[11px] text-muted">最近打开</div>
              {recent.slice(0, 6).map((path) => (
                <button
                  key={path}
                  type="button"
                  role="menuitem"
                  title={path}
                  className="block w-full truncate px-3 py-1 text-left font-mono text-[12px] hover:bg-surface-2"
                  onClick={() => {
                    setOpen(false);
                    runCommand(`openRecent:${path}`);
                  }}
                >
                  {path.split(/[\\/]/).pop()}
                </button>
              ))}
            </>
          )}
          <div className="my-1 border-t border-line" />
          {item('保存', 'save', `${MOD}S`)}
          {item('另存为…', 'saveAs', `${MOD}Shift+S`)}
          <div className="my-1 border-t border-line" />
          {item('从代码导入…', 'import', `${MOD}I`)}
          {item('导出 .py…', 'exportPy', `${MOD}E`)}
          <div className="my-1 border-t border-line" />
          {item('Python 环境…', 'environment')}
        </div>
      )}
    </div>
  );
}

export function TopBar({ theme, onTheme }: { theme: ThemeChoice; onTheme: (t: ThemeChoice) => void }) {
  const docName = useStudio((s) => s.doc.name);
  const dirty = useDirty();
  const canUndo = useStudio((s) => s.history.past.length > 0);
  const canRedo = useStudio((s) => s.history.future.length > 0);
  const backendVersion = useUi((s) => s.backendVersion);
  const [backend, setBackend] = useState<Backend>({ state: 'checking' });

  useEffect(() => {
    setBackend({ state: 'checking' });
    rpc<EnvInfo>('env_info')
      .then((env) => setBackend(env.torch ? { state: 'ok', env } : { state: 'down', reason: '当前 Python 环境没有安装 torch' }))
      .catch((e: Error) => setBackend({ state: 'down', reason: e.message }));
  }, [backendVersion]);

  return (
    <header className="flex items-center gap-1 border-b border-line bg-surface px-3">
      <img src="/neuroloom-mark.svg" alt="" className="mr-1 size-7" />
      <span className="mr-3 text-[15px] font-bold tracking-tight">
        Neuro<span className="text-primary">Loom</span>
      </span>
      <FileMenu />
      <span className="mx-2 h-5 w-px bg-line" />
      <span className="mr-2 flex max-w-64 items-center gap-1.5 truncate text-[13px]" title={dirty ? '有未保存的更改' : '已保存'}>
        <span className="truncate font-medium">{docName}</span>
        {dirty && <span aria-label="未保存" className="size-1.5 flex-none rounded-full bg-warning" />}
      </span>
      <button type="button" className={button} disabled={!canUndo} onClick={() => runCommand('undo')} title={`撤销（${MOD}Z）`}>
        ↶
      </button>
      <button type="button" className={button} disabled={!canRedo} onClick={() => runCommand('redo')} title={`重做（${MOD}Shift+Z）`}>
        ↷
      </button>
      <span className="mx-2 h-5 w-px bg-line" />
      <button type="button" className={button} onClick={() => runCommand('import')}>
        导入代码
      </button>
      <button type="button" className={button} onClick={() => runCommand('autoLayout')} title={`${MOD}L`}>
        自动布局
      </button>
      <span className="flex-1" />
      <button
        type="button"
        className={`${button} flex items-center gap-1.5 text-[12px] text-muted`}
        onClick={() => runCommand('environment')}
        title={backend.state === 'down' ? backend.reason : backend.state === 'ok' ? backend.env.executable : ''}
      >
        <span
          className={`size-2 rounded-full ${backend.state === 'ok' ? 'bg-success' : backend.state === 'down' ? 'bg-line-strong' : 'bg-warning'}`}
        />
        {backend.state === 'ok'
          ? `Python ${backend.env.python} · torch ${backend.env.torch}`
          : backend.state === 'down'
            ? 'Python 后端未连接'
            : '正在连接后端…'}
      </button>
      <button type="button" className={button} onClick={() => onTheme(NEXT_THEME[theme])} title="切换主题">
        {THEME_LABEL[theme]}
      </button>
    </header>
  );
}
