// Model analysis (PRD F8.2, 12.5): parameters, FLOPs and memory per module
// instance, sortable, with the symbolic dimensions (batch size, sequence length)
// set here. Clicking a row shows its node on the canvas.
import { costOf, defaultSymbolValue, symbolsOf, type CostRow, type SymbolValues } from '@neuroloom/core';
import { useReactFlow } from '@xyflow/react';
import { useMemo, useState, type ReactNode } from 'react';
import { create } from 'zustand';
import { tr } from '../i18n';
import { formatBytes, formatCount, formatFlops } from '../lib/format';
import { useStudio } from '../store';

interface SymbolState {
  /** Values the user set; other symbols use defaultSymbolValue. */
  values: SymbolValues;
  set(symbol: string, value: number): void;
}

export const useSymbolValues = create<SymbolState>()((set) => ({
  values: {},
  set: (symbol, value) => set((s) => ({ values: { ...s.values, [symbol]: value } })),
}));

/** A value for each of `symbols`: the user's, or the default. */
export function useSymbols(symbols: string[]): SymbolValues {
  const values = useSymbolValues((s) => s.values);
  const key = symbols.join(',');
  // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the symbol names
  return useMemo(() => Object.fromEntries(symbols.map((s) => [s, values[s] ?? defaultSymbolValue(s)])), [key, values]);
}

type SortKey = 'order' | 'name' | 'params' | 'flops' | 'activationBytes';

function sorted(rows: CostRow[], key: SortKey): CostRow[] {
  if (key === 'order') return rows;
  if (key === 'name') return [...rows].sort((a, b) => a.name.localeCompare(b.name));
  return [...rows].sort((a, b) => b[key] - a[key]);
}

const share = (part: number, whole: number) => (whole > 0 ? `${((part / whole) * 100).toFixed(1)}%` : '');

export function AnalysisPanel({ tabs, panelToggle, collapsed }: { tabs: ReactNode; panelToggle: ReactNode; collapsed: boolean }) {
  const project = useStudio((s) => s.project);
  const entry = useStudio((s) => s.analysis.entry);
  const symbols = useMemo(() => symbolsOf(entry), [entry]);
  const values = useSymbols(symbols);
  const setSymbol = useSymbolValues((s) => s.set);
  const cost = useMemo(() => costOf(project, entry, values), [project, entry, values]);
  const [sort, setSort] = useState<SortKey>('order');
  const [open, setOpen] = useState<Set<string>>(new Set());
  const { fitView } = useReactFlow();

  const jump = (row: CostRow) => {
    const studio = useStudio.getState();
    studio.openModule(row.moduleId, row.path);
    useStudio.getState().selectNode(row.nodeId);
    setTimeout(() => void fitView({ nodes: [{ id: row.nodeId }], duration: 300, maxZoom: 1.2, padding: 0.6 }), 80);
  };
  const toggle = (name: string) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  const lines: { row: CostRow; depth: number }[] = [];
  const walk = (rows: CostRow[], depth: number) => {
    for (const row of sorted(rows, sort)) {
      lines.push({ row, depth });
      if (row.children && open.has(row.name)) walk(row.children, depth + 1);
    }
  };
  walk(cost.rows, 0);

  const header = (key: SortKey, label: string, align = 'text-right') => (
    <button type="button" onClick={() => setSort(sort === key ? 'order' : key)} className={`${align} hover:text-fg ${sort === key ? 'font-semibold text-fg' : ''}`}>
      {label}
      {sort === key && (key === 'name' ? ' ↑' : ' ↓')}
    </button>
  );
  const approx = cost.partial ? '≥ ' : '';
  const grid = 'grid grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)_88px_96px_88px] gap-x-3';

  return (
    <section className="flex min-h-0 flex-col border-t border-line bg-surface">
      <header className={`flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2 text-[12px] ${collapsed ? '' : 'border-b border-line'}`}>
        {tabs}
        <span title={tr('参数按 float32 计', 'Parameters as float32')}>
          {tr('参数', 'Params')} <b>{formatCount(cost.params)}</b> <span className="text-muted">({formatBytes(cost.paramBytes)})</span>
        </span>
        <span title={tr('一次前向传播；乘加计 2 次，与 FlopCounterMode 相同', 'One forward pass; a multiply-add counts 2, as in FlopCounterMode')}>
          {tr('计算量', 'Compute')} <b>{approx}{formatFlops(cost.flops)}</b> <span className="text-muted">({approx}{formatCount(cost.flops / 2)} MACs)</span>
        </span>
        <span title={tr('前向传播中产生的所有张量', 'Every tensor the forward pass produces')}>
          {tr('激活', 'Activations')} <b>{approx}{formatBytes(cost.activationBytes)}</b>
        </span>
        <span title={tr('参数内存 + 激活内存（推理时的粗略上限，不含梯度和优化器状态）', 'Parameter memory + activation memory (a rough bound for inference; no gradients or optimizer state)')}>
          {tr('显存约', 'Memory ≈')} <b>{approx}{formatBytes(cost.paramBytes + cost.activationBytes)}</b>
        </span>
        <span className="flex-1" />
        {symbols.length > 0 && (
          <span className="flex items-center gap-2 text-muted">
            {symbols.map((s) => (
              <label key={s} className="flex items-center gap-1 font-mono">
                {s} =
                <input
                  type="number"
                  min={1}
                  value={values[s]}
                  onChange={(e) => {
                    const n = Math.floor(Number(e.target.value));
                    if (n >= 1) setSymbol(s, n);
                  }}
                  className="w-16 rounded border border-line bg-bg px-1.5 py-0.5 text-fg outline-none focus:border-primary"
                />
              </label>
            ))}
          </span>
        )}
        {panelToggle}
      </header>
      {!collapsed && (
        <div className="min-h-0 flex-1 overflow-auto px-2 py-1 text-[12px]">
          <div className={`${grid} sticky top-0 z-[1] bg-surface px-2 py-1 text-[11px] text-muted`}>
            {header('name', tr('名称', 'Name'), 'text-left')}
            <span>{tr('类型', 'Type')}</span>
            {header('params', tr('参数', 'Params'))}
            {header('flops', 'FLOPs')}
            {header('activationBytes', tr('激活', 'Activations'))}
          </div>
          {lines.map(({ row, depth }) => (
            <div
              key={row.name}
              role="button"
              tabIndex={0}
              onClick={() => jump(row)}
              onKeyDown={(e) => e.key === 'Enter' && jump(row)}
              title={row.partial ? tr('包含无法计算的部分：代码块、外部模块或未知形状', 'Includes parts that cannot be counted: code blocks, external modules or unknown shapes') : undefined}
              className={`${grid} cursor-pointer rounded px-2 py-0.5 hover:bg-surface-2`}
            >
              <span className="flex min-w-0 items-center font-mono" style={{ paddingLeft: depth * 14 }}>
                {row.children ? (
                  <button
                    type="button"
                    aria-label={open.has(row.name) ? tr('收起', 'Collapse') : tr('展开', 'Expand')}
                    className="mr-1 w-3 flex-none text-muted hover:text-fg"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggle(row.name);
                    }}
                  >
                    {open.has(row.name) ? '▾' : '▸'}
                  </button>
                ) : (
                  <span className="mr-1 w-3 flex-none" />
                )}
                <span className="truncate">{row.name.split('.').pop()}</span>
              </span>
              <span className="truncate text-muted">{row.type}</span>
              <span className="text-right font-mono">{row.params > 0 ? formatCount(row.params) : ''}</span>
              <span className="text-right font-mono" title={share(row.flops, cost.flops)}>
                {row.flops > 0 ? formatFlops(row.flops) : row.partial ? '?' : ''}
              </span>
              <span className="text-right font-mono" title={share(row.activationBytes, cost.activationBytes)}>
                {row.activationBytes > 0 ? formatBytes(row.activationBytes) : row.partial ? '?' : ''}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
