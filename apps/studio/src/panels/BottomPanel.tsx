import { useState } from 'react';
import { tr } from '../i18n';
import { AnalysisPanel } from './AnalysisPanel';
import { CodePanel } from './CodePanel';

type Tab = 'code' | 'analysis';
const KEY = 'neuroloom.bottomTab';

function loadTab(): Tab {
  try {
    return localStorage.getItem(KEY) === 'analysis' ? 'analysis' : 'code';
  } catch {
    return 'code';
  }
}

/** The panel under the canvas: live code (PRD F5.2) or the model analysis (F8.2). */
export function BottomPanel() {
  const [tab, setTab] = useState<Tab>(loadTab);
  const choose = (next: Tab) => {
    setTab(next);
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
  return tab === 'code' ? <CodePanel tabs={tabs} /> : <AnalysisPanel tabs={tabs} />;
}
