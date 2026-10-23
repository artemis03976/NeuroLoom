import { ReactFlowProvider } from '@xyflow/react';
import { useEffect, useState } from 'react';
import { FlowCanvas } from './canvas/FlowCanvas';
import { discardRecovery, restoreRecovery } from './commands';
import { useClipboard, useCommands, useDocumentLifecycle } from './hooks';
import { applyTheme, loadTheme, type ThemeChoice } from './lib/theme';
import { CodePanel } from './panels/CodePanel';
import { EnvironmentDialog } from './panels/EnvironmentDialog';
import { ImportDialog } from './panels/ImportDialog';
import { Inspector } from './panels/Inspector';
import { Palette } from './panels/Palette';
import { QuickAdd } from './panels/QuickAdd';
import { TemplateDialog } from './panels/TemplateDialog';
import { TopBar } from './panels/TopBar';
import { useUi } from './ui';

function Toasts() {
  const toasts = useUi((s) => s.toasts);
  return (
    <div className="pointer-events-none absolute top-3 left-1/2 z-20 flex max-w-[80%] -translate-x-1/2 flex-col gap-2">
      {toasts.map((t) => (
        <div key={t.id} role="status" className="rounded-lg border border-line bg-surface px-4 py-2 text-[12px] shadow-lg">
          <div className={`font-medium ${t.tone === 'error' ? 'text-error' : t.tone === 'success' ? 'text-success' : ''}`}>{t.message}</div>
          {t.details?.map((d, i) => (
            <div key={i} className="text-muted">
              {d}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function RecoveryBanner() {
  const recovery = useUi((s) => s.recovery);
  if (!recovery) return null;
  return (
    <div className="absolute inset-x-3 top-3 z-20 flex items-center gap-3 rounded-lg border border-warning/50 bg-surface px-4 py-2 text-[12.5px] shadow-lg">
      <span className="size-2 flex-none rounded-full bg-warning" />
      <span className="flex-1">
        发现 {new Date(recovery.savedAt).toLocaleString()} 自动保存的未保存工作{recovery.path ? `（${recovery.path}）` : ''}。
      </span>
      <button type="button" className="rounded-md bg-primary px-3 py-1 text-primary-fg hover:bg-primary-hover" onClick={restoreRecovery}>
        恢复
      </button>
      <button type="button" className="rounded-md border border-line px-3 py-1 hover:bg-surface-2" onClick={discardRecovery}>
        丢弃
      </button>
    </div>
  );
}

export function App() {
  const [theme, setTheme] = useState<ThemeChoice>(loadTheme);
  const dialog = useUi((s) => s.dialog);
  const openDialog = useUi((s) => s.openDialog);
  const backendChanged = useUi((s) => s.backendChanged);
  const quickAdd = useUi((s) => s.quickAdd);

  useEffect(() => applyTheme(theme), [theme]);
  useCommands();
  useClipboard();
  useDocumentLifecycle();

  return (
    <ReactFlowProvider>
      <div className="grid h-full grid-rows-[48px_minmax(0,1fr)_minmax(160px,34%)] bg-bg text-fg">
        <TopBar theme={theme} onTheme={setTheme} />
        <div className="grid min-h-0 grid-cols-[196px_minmax(0,1fr)_292px]">
          <Palette />
          <main className="relative min-h-0">
            <FlowCanvas colorMode={theme} />
            <RecoveryBanner />
            <Toasts />
          </main>
          <Inspector />
        </div>
        <CodePanel />
      </div>
      {quickAdd && <QuickAdd key={`${quickAdd.screen.x},${quickAdd.screen.y}`} state={quickAdd} />}
      {dialog === 'new' && <TemplateDialog onClose={() => openDialog(null)} />}
      {dialog === 'import' && <ImportDialog onClose={() => openDialog(null)} />}
      {dialog === 'environment' && <EnvironmentDialog onClose={() => openDialog(null)} onChanged={backendChanged} />}
    </ReactFlowProvider>
  );
}
