import type { MenuCommand } from '@neuroloom/host-api';
import { useEffect, useRef, useState } from 'react';
import { useReplay } from '../canvas/replay';
import { runCommand } from '../commands';
import { host } from '../host';
import { tr, useLocale } from '../i18n';
import { rpc, type EnvInfo } from '../lib/rpc';
import type { ThemeChoice } from '../lib/theme';
import { useDirty, useStudio } from '../store';
import { useUi } from '../ui';

type Backend = { state: 'checking' } | { state: 'ok'; env: EnvInfo } | { state: 'down'; reason: string };

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
        {tr('文件', 'File')}
      </button>
      {open && (
        <div role="menu" className="absolute top-full left-0 z-40 mt-1 w-64 rounded-lg border border-line bg-surface py-1 shadow-xl">
          {item(tr('新建', 'New'), 'new', `${MOD}N`)}
          {item(tr('打开…', 'Open…'), 'open', `${MOD}O`)}
          {recent.length > 0 && (
            <>
              <div className="px-3 pt-2 pb-1 text-[11px] text-muted">{tr('最近打开', 'Recent')}</div>
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
          {item(tr('保存', 'Save'), 'save', `${MOD}S`)}
          {item(tr('另存为…', 'Save As…'), 'saveAs', `${MOD}Shift+S`)}
          <div className="my-1 border-t border-line" />
          {item(tr('从代码导入…', 'Import from Code…'), 'import', `${MOD}I`)}
          {item(tr('导出 .py…', 'Export .py…'), 'exportPy', `${MOD}E`)}
          {item(tr('导出选项…', 'Export Options…'), 'exportOptions')}
          {item(tr('导出图片…', 'Export Image…'), 'exportImage')}
          <div className="my-1 border-t border-line" />
          {item(tr('Python 环境…', 'Python Environment…'), 'environment')}
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
  const snap = useUi((s) => s.snapToGrid);
  const replaying = useReplay((s) => s.open);
  const locale = useLocale((s) => s.locale);
  const [backend, setBackend] = useState<Backend>({ state: 'checking' });
  const themeLabel: Record<ThemeChoice, string> = { system: tr('跟随系统', 'System'), light: tr('浅色', 'Light'), dark: tr('深色', 'Dark') };

  useEffect(() => {
    setBackend({ state: 'checking' });
    rpc<EnvInfo>('env_info')
      .then((env) => setBackend(env.torch ? { state: 'ok', env } : { state: 'down', reason: tr('当前 Python 环境没有安装 torch', 'The Python environment has no torch') }))
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
      <span className="mr-2 flex max-w-64 items-center gap-1.5 truncate text-[13px]" title={dirty ? tr('有未保存的更改', 'Unsaved changes') : tr('已保存', 'Saved')}>
        <span className="truncate font-medium">{docName}</span>
        {dirty && <span aria-label={tr('未保存', 'unsaved')} className="size-1.5 flex-none rounded-full bg-warning" />}
      </span>
      <button type="button" className={button} disabled={!canUndo} onClick={() => runCommand('undo')} title={tr(`撤销（${MOD}Z）`, `Undo (${MOD}Z)`)}>
        ↶
      </button>
      <button type="button" className={button} disabled={!canRedo} onClick={() => runCommand('redo')} title={tr(`重做（${MOD}Shift+Z）`, `Redo (${MOD}Shift+Z)`)}>
        ↷
      </button>
      <span className="mx-2 h-5 w-px bg-line" />
      <button type="button" className={button} onClick={() => runCommand('import')}>
        {tr('导入代码', 'Import code')}
      </button>
      <button type="button" className={button} onClick={() => runCommand('autoLayout')} title={`${MOD}L`}>
        {tr('自动布局', 'Auto layout')}
      </button>
      <button
        type="button"
        className={`${button} ${snap ? 'bg-primary-soft text-primary' : ''}`}
        aria-pressed={snap}
        onClick={() => runCommand('toggleSnap')}
        title={tr('拖动节点时吸附到 20 px 网格', 'Snap nodes to a 20 px grid while dragging')}
      >
        {tr('网格吸附', 'Snap')}
      </button>
      <button
        type="button"
        className={`${button} ${replaying ? 'bg-primary-soft text-primary' : ''}`}
        aria-pressed={replaying}
        onClick={() => runCommand('replay')}
        title={tr('按拓扑顺序逐层演示前向传播', 'Step through the forward pass in topological order')}
      >
        {tr('▶ 回放', '▶ Replay')}
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
            ? tr('Python 后端未连接', 'Python backend not connected')
            : tr('正在连接后端…', 'Connecting…')}
      </button>
      <button
        type="button"
        className={button}
        onClick={() => runCommand(locale === 'zh' ? 'locale:en' : 'locale:zh')}
        title={tr('切换界面语言', 'Switch the interface language')}
      >
        {locale === 'zh' ? 'EN' : '中文'}
      </button>
      <button type="button" className={button} onClick={() => onTheme(NEXT_THEME[theme])} title={tr('切换主题', 'Switch theme')}>
        {themeLabel[theme]}
      </button>
    </header>
  );
}
