import { entryModule } from '@neuroloom/core';
import { useEffect, useState } from 'react';
import { rpc, type EnvInfo } from '../lib/rpc';
import type { ThemeChoice } from '../lib/theme';
import { useStudio } from '../store';
import { downloadCode } from './CodePanel';

type Backend = { state: 'checking' } | { state: 'ok'; env: EnvInfo } | { state: 'down'; reason: string };

const THEME_LABEL: Record<ThemeChoice, string> = { system: '跟随系统', light: '浅色', dark: '深色' };
const NEXT_THEME: Record<ThemeChoice, ThemeChoice> = { system: 'light', light: 'dark', dark: 'system' };

export function TopBar({ theme, onTheme, onImport }: { theme: ThemeChoice; onTheme: (t: ThemeChoice) => void; onImport: () => void }) {
  const requestLayout = useStudio((s) => s.requestLayout);
  const className = useStudio((s) => entryModule(s.project).className);
  const code = useStudio((s) => s.codegen.code);
  const [backend, setBackend] = useState<Backend>({ state: 'checking' });

  useEffect(() => {
    rpc<EnvInfo>('env_info')
      .then((env) => setBackend(env.torch ? { state: 'ok', env } : { state: 'down', reason: '当前 Python 环境没有安装 torch' }))
      .catch((e: Error) => setBackend({ state: 'down', reason: e.message }));
  }, []);

  const button = 'rounded-md px-2.5 py-1.5 text-[13px] hover:bg-surface-2';
  return (
    <header className="flex items-center gap-2 border-b border-line bg-surface px-3">
      <img src="/neuroloom-mark.svg" alt="" className="size-7" />
      <span className="mr-2 text-[15px] font-bold tracking-tight">
        Neuro<span className="text-primary">Loom</span>
      </span>
      <span className="mr-3 h-5 w-px bg-line" />
      <span className="mr-3 font-mono text-[12.5px] text-muted">{className}</span>
      <button type="button" className={button} onClick={onImport}>
        导入代码
      </button>
      <button type="button" className={button} onClick={requestLayout}>
        自动布局
      </button>
      <button type="button" className={button} onClick={() => downloadCode(code, className)}>
        导出 .py
      </button>
      <span className="flex-1" />
      <span
        className="flex items-center gap-1.5 text-[12px] text-muted"
        title={backend.state === 'down' ? backend.reason : backend.state === 'ok' ? backend.env.python : ''}
      >
        <span
          className={`size-2 rounded-full ${backend.state === 'ok' ? 'bg-success' : backend.state === 'down' ? 'bg-line-strong' : 'bg-warning'}`}
        />
        {backend.state === 'ok'
          ? `Python ${backend.env.python} · torch ${backend.env.torch}`
          : backend.state === 'down'
            ? 'Python 后端未连接'
            : '正在连接后端…'}
      </span>
      <button type="button" className={button} onClick={() => onTheme(NEXT_THEME[theme])} title="切换主题">
        {THEME_LABEL[theme]}
      </button>
    </header>
  );
}
