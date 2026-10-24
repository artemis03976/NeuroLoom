import { useState } from 'react';
import { runCommand } from '../commands';
import { tr } from '../i18n';
import { useUi } from '../ui';
import { AnalysisPanel } from './AnalysisPanel';
import { CodePanel } from './CodePanel';

type Tab = 'code' | 'analysis';
const KEY = 'neuroloom.bottomTab';
const MOD = navigator.platform.toLowerCase().includes('mac') ? '⌘' : 'Ctrl+';

function loadTab(): Tab {
  try {
    return localStorage.getItem(KEY) === 'analysis' ? 'analysis' : 'code';
  } catch {
    return 'code';
  }
}

/**
 * The panel under the canvas: live code (PRD F5.2) or the model analysis (F8.2).
 * Collapsed, only its header stays and the canvas takes the room (Ctrl+J).
 */
export function BottomPanel() {
  const [tab, setTab] = useState<Tab>(loadTab);
  const collapsed = useUi((s) => s.bottomCollapsed);
  const choose = (next: Tab) => {
    setTab(next);
    // Picking a tab of the collapsed panel opens it there.
    if (collapsed) runCommand('toggleBottomPanel');
    try {
      localStorage.setItem(KEY, next);
    } catch {
      // Not remembered; the choice still applies now.
    }
  };
  const tabClass = (active: boolean) => `rounded-md px-2 py-0.5 text-[12px] ${active ? 'bg-surface-2 font-semibold text-fg' : 'text-muted hover:text-fg'}`;
  const tabs = (
    <span role="tablist" className="flex items-center gap-1">
      <button type="button" role="tab" aria-selected={tab === 'code'} className={tabClass(tab === 'code')} onClick={() => choose('code')}>
        {tr('代码预览', 'Code preview')}
      </button>
      <button type="button" role="tab" aria-selected={tab === 'analysis'} className={tabClass(tab === 'analysis')} onClick={() => choose('analysis')}>
        {tr('分析', 'Analysis')}
      </button>
    </span>
  );
  const panelToggle = (
    <button
      type="button"
      aria-expanded={!collapsed}
      aria-label={collapsed ? tr('展开面板', 'Expand panel') : tr('收起面板', 'Collapse panel')}
      title={collapsed ? tr(`展开面板（${MOD}J）`, `Expand panel (${MOD}J)`) : tr(`收起面板（${MOD}J）`, `Collapse panel (${MOD}J)`)}
      className="-mr-1.5 flex size-7 flex-none items-center justify-center rounded-md text-muted hover:bg-surface-2 hover:text-fg"
      onClick={() => runCommand('toggleBottomPanel')}
    >
      <svg viewBox="0 0 16 16" aria-hidden="true" className={`size-4 ${collapsed ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 6l4 4 4-4" />
      </svg>
    </button>
  );
  return tab === 'code' ? <CodePanel tabs={tabs} panelToggle={panelToggle} collapsed={collapsed} /> : <AnalysisPanel tabs={tabs} panelToggle={panelToggle} collapsed={collapsed} />;
}
