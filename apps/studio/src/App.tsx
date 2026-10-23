import { ReactFlowProvider } from '@xyflow/react';
import { useEffect, useState } from 'react';
import { FlowCanvas } from './canvas/FlowCanvas';
import { applyTheme, loadTheme, type ThemeChoice } from './lib/theme';
import { CodePanel } from './panels/CodePanel';
import { ImportDialog } from './panels/ImportDialog';
import { Inspector } from './panels/Inspector';
import { Palette } from './panels/Palette';
import { TopBar } from './panels/TopBar';

export function App() {
  const [theme, setTheme] = useState<ThemeChoice>(loadTheme);
  const [importing, setImporting] = useState(false);
  const [notice, setNotice] = useState<string[] | null>(null);

  useEffect(() => applyTheme(theme), [theme]);
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 6000);
    return () => clearTimeout(t);
  }, [notice]);

  return (
    <ReactFlowProvider>
      <div className="grid h-full grid-rows-[48px_minmax(0,1fr)_minmax(160px,34%)] bg-bg text-fg">
        <TopBar theme={theme} onTheme={setTheme} onImport={() => setImporting(true)} />
        <div className="grid min-h-0 grid-cols-[196px_minmax(0,1fr)_292px]">
          <Palette />
          <main className="relative min-h-0">
            <FlowCanvas colorMode={theme} />
            {notice && (
              <div role="status" className="absolute top-3 left-1/2 z-10 max-w-[80%] -translate-x-1/2 rounded-lg border border-line bg-surface px-4 py-2 text-[12px] shadow-lg">
                <div className="font-medium text-success">导入完成</div>
                {notice.map((w, i) => (
                  <div key={i} className="text-muted">
                    {w}
                  </div>
                ))}
              </div>
            )}
          </main>
          <Inspector />
        </div>
        <CodePanel />
      </div>
      {importing && <ImportDialog onClose={() => setImporting(false)} onImported={setNotice} />}
    </ReactFlowProvider>
  );
}
