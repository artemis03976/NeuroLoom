import { ReactFlowProvider } from '@xyflow/react';
import { useEffect, useState } from 'react';
import { FlowCanvas } from './canvas/FlowCanvas';
import { Breadcrumb, ContextMenuView, SelectionBar, SourceView } from './canvas/Overlays';
import { ReplayBar } from './canvas/ReplayBar';
import { discardRecovery, restoreRecovery } from './commands';
import { host } from './host';
import { useClipboard, useCommands, useDocumentLifecycle } from './hooks';
import { tr, useLocale } from './i18n';
import { applyTheme, loadTheme, type ThemeChoice } from './lib/theme';
import { BottomPanel } from './panels/BottomPanel';
import { EnvironmentDialog } from './panels/EnvironmentDialog';
import { ImportDialog } from './panels/ImportDialog';
import { Inspector } from './panels/Inspector';
import { Palette } from './panels/Palette';
import { PromptDialog } from './panels/PromptDialog';
import { QuickAdd } from './panels/QuickAdd';
import { TemplateDialog } from './panels/TemplateDialog';
import { TopBar } from './panels/TopBar';
import { useStudio } from './store';
import { useUi } from './ui';

function Toasts() {
  const toasts = useUi((s) => s.toasts);
  return (
    <div className="pointer-events-none absolute top-14 left-1/2 z-20 flex max-w-[80%] -translate-x-1/2 flex-col gap-2">
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
  const when = new Date(recovery.savedAt).toLocaleString();
  return (
    <div className="absolute inset-x-3 top-14 z-20 flex items-center gap-3 rounded-lg border border-warning/50 bg-surface px-4 py-2 text-[12.5px] shadow-lg">
      <span className="size-2 flex-none rounded-full bg-warning" />
      <span className="flex-1">
        {tr(`发现 ${when} 自动保存的未保存工作`, `Found unsaved work autosaved at ${when}`)}
        {recovery.path ? `（${recovery.path}）` : ''}
      </span>
      <button type="button" className="rounded-md bg-primary px-3 py-1 text-primary-fg hover:bg-primary-hover" onClick={restoreRecovery}>
        {tr('恢复', 'Restore')}
      </button>
      <button type="button" className="rounded-md border border-line px-3 py-1 hover:bg-surface-2" onClick={discardRecovery}>
        {tr('丢弃', 'Discard')}
      </button>
    </div>
  );
}

export function App() {
  const [theme, setTheme] = useState<ThemeChoice>(loadTheme);
  const locale = useLocale((s) => s.locale);
  const dialog = useUi((s) => s.dialog);
  const prompt = useUi((s) => s.prompt);
  const openDialog = useUi((s) => s.openDialog);
  const backendChanged = useUi((s) => s.backendChanged);
  const quickAdd = useUi((s) => s.quickAdd);

  useEffect(() => applyTheme(theme), [theme]);
  // A new language: native menus follow, and diagnostics are produced again in it.
  useEffect(() => {
    host.setLocale(locale);
    useStudio.getState().relocalize();
  }, [locale]);
  useCommands();
  useClipboard();
  useDocumentLifecycle();

  return (
    <ReactFlowProvider>
      {/* Keyed by language so every label re-renders in it. */}
      <div key={locale} className="grid h-full grid-rows-[48px_minmax(0,1fr)_minmax(160px,34%)] bg-bg text-fg">
        <TopBar theme={theme} onTheme={setTheme} />
        <div className="grid min-h-0 grid-cols-[208px_minmax(0,1fr)_300px]">
          <Palette />
          <main className="relative min-h-0">
            <FlowCanvas colorMode={theme} />
            <div className="pointer-events-none absolute inset-x-3 top-3 z-10 flex items-start justify-between gap-3">
              <Breadcrumb />
              <SelectionBar />
            </div>
            <SourceView />
            <ReplayBar />
            <RecoveryBanner />
            <Toasts />
          </main>
          <Inspector />
        </div>
        <BottomPanel />
      </div>
      {quickAdd && <QuickAdd key={`${quickAdd.screen.x},${quickAdd.screen.y}`} state={quickAdd} />}
      <ContextMenuView />
      {prompt && <PromptDialog prompt={prompt} onClose={() => useUi.getState().setPrompt(null)} />}
      {dialog === 'new' && <TemplateDialog onClose={() => openDialog(null)} />}
      {dialog === 'import' && <ImportDialog onClose={() => openDialog(null)} />}
      {dialog === 'environment' && <EnvironmentDialog onClose={() => openDialog(null)} onChanged={backendChanged} />}
    </ReactFlowProvider>
  );
}
