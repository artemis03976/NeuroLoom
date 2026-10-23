import {
  entryModule,
  formatShape,
  getSpec,
  isAuto,
  literalOf,
  withDefaults,
  type NodeInst,
  type ParamSpec,
  type ParamValue,
} from '@neuroloom/core';
import { useState, type CSSProperties, type ReactNode } from 'react';
import { categoryColor, glyphOf } from '../canvas/glyph';
import { formatCount } from '../lib/format';
import { useSelectedNodeId, useStudio } from '../store';

const field =
  'w-full rounded-md border border-line bg-bg px-2 py-1 font-mono text-[12.5px] outline-none focus:border-primary focus:ring-2 focus:ring-primary-soft';

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="border-b border-line px-4 py-3">
      <h3 className="mb-2 text-[11px] font-semibold tracking-wide text-muted">{title}</h3>
      {children}
    </section>
  );
}

/** A text field that only commits values that parse; keeps the raw text while typing. */
function IntField({ value, onCommit, disabled }: { value: number | undefined; onCommit: (n: number) => void; disabled?: boolean }) {
  const [draft, setDraft] = useState<string | null>(null);
  const text = draft ?? (value === undefined ? '' : String(value));
  const valid = /^\d+$/.test(text.trim());
  return (
    <input
      className={`${field} ${!valid && !disabled ? 'border-error' : ''}`}
      inputMode="numeric"
      value={text}
      disabled={disabled}
      onChange={(e) => {
        setDraft(e.target.value);
        if (/^\d+$/.test(e.target.value.trim())) onCommit(Number(e.target.value));
      }}
      onBlur={() => setDraft(null)}
    />
  );
}

function ParamRow({ node, spec, resolved }: { node: NodeInst; spec: ParamSpec; resolved?: Record<string, unknown> }) {
  const setParam = useStudio((s) => s.setParam);
  const value: ParamValue | undefined = node.params[spec.name] ?? (spec.default !== undefined ? { literal: spec.default } : undefined);
  const auto = isAuto(value);
  const lit = literalOf(value);
  const changed = spec.default !== undefined && lit !== undefined && lit !== spec.default;
  const set = (v: ParamValue) => setParam(node.id, spec.name, v);

  return (
    <div className="mb-2.5">
      <label className="mb-1 flex items-center justify-between gap-2 text-[12px]">
        <span className={changed ? 'font-semibold' : ''}>
          <span className="font-mono">{spec.name}</span>
          <span className="ml-1.5 text-muted">{spec.label.zh}</span>
        </span>
        {spec.allowAuto && (
          <span className="flex items-center gap-1 text-[11px] text-muted">
            <input
              type="checkbox"
              checked={auto}
              onChange={(e) =>
                set(e.target.checked ? { auto: true } : { literal: (resolved?.[spec.name] as number | undefined) ?? 1 })
              }
            />
            自动推断
          </span>
        )}
      </label>
      {spec.type === 'bool' ? (
        <label className="flex items-center gap-2 text-[12.5px]">
          <input type="checkbox" checked={lit === true} onChange={(e) => set({ literal: e.target.checked })} />
          <span className="font-mono">{lit ? 'True' : 'False'}</span>
        </label>
      ) : spec.type === 'shape' ? (
        <input className={field} value={String(lit ?? '')} onChange={(e) => set({ literal: e.target.value })} spellCheck={false} />
      ) : (
        <IntField
          key={`${node.id}:${spec.name}:${auto}`}
          value={auto ? (resolved?.[spec.name] as number | undefined) : (lit as number | undefined)}
          disabled={auto}
          onCommit={(n) => set({ literal: n })}
        />
      )}
    </div>
  );
}

function NodeInspector({ nodeId }: { nodeId: string }) {
  const node = useStudio((s) => entryModule(s.project).nodes[nodeId]);
  const info = useStudio((s) => s.analysis.nodes[nodeId]);
  const diagnostics = useStudio((s) => s.analysis.diagnostics);
  const rename = useStudio((s) => s.rename);
  if (!node) return null;
  const spec = getSpec(node.type);
  if (!spec) return <Section title="节点">未知类型 {node.type}</Section>;
  const params = withDefaults(spec, node.params);
  const mine = diagnostics.filter((d) => d.nodeId === nodeId);
  const shapeOut = info?.outputs?.[0];

  return (
    <>
      <div className="flex items-center gap-3 border-b border-line px-4 py-3">
        <span
          className="grid size-8 flex-none place-items-center rounded-lg font-mono text-[11px] font-bold text-white"
          style={{ background: categoryColor(spec) } as CSSProperties}
        >
          {glyphOf(spec)}
        </span>
        <div className="min-w-0">
          <div className="text-[13px] font-semibold">{spec.displayName.zh}</div>
          <div className="truncate font-mono text-[11px] text-muted">{spec.type}</div>
        </div>
      </div>
      <Section title="变量名">
        <input className={field} value={node.name} onChange={(e) => rename(nodeId, e.target.value)} spellCheck={false} />
      </Section>
      {spec.params.length > 0 && (
        <Section title="参数">
          {spec.params.map((p) => (
            <ParamRow key={p.name} node={{ ...node, params }} spec={p} resolved={info?.resolved} />
          ))}
        </Section>
      )}
      <Section title="张量">
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[12px]">
          {spec.inputs.length > 0 && (
            <>
              <dt className="text-muted">输入</dt>
              <dd>{info?.inputs[0] ? formatShape(info.inputs[0].shape) : '—'}</dd>
            </>
          )}
          {node.kind !== 'output' && (
            <>
              <dt className="text-muted">输出</dt>
              <dd>{shapeOut ? formatShape(shapeOut.shape) : '—'}</dd>
            </>
          )}
          {spec.kind === 'layer' && (
            <>
              <dt className="text-muted">参数量</dt>
              <dd>{(info?.paramCount ?? 0).toLocaleString()}</dd>
            </>
          )}
        </dl>
      </Section>
      {mine.length > 0 && (
        <Section title="问题">
          {mine.map((d, i) => (
            <p key={i} className={`text-[12px] leading-relaxed ${d.severity === 'error' ? 'text-error' : 'text-warning'}`}>
              {d.message}
            </p>
          ))}
        </Section>
      )}
      <Section title="说明">
        <p className="text-[12px] leading-relaxed text-muted">{spec.summary.zh}</p>
        {spec.docUrl && (
          <a className="mt-1 inline-block text-[12px] text-primary hover:underline" href={spec.docUrl} target="_blank" rel="noreferrer">
            PyTorch 官方文档 ↗
          </a>
        )}
      </Section>
    </>
  );
}

function ModelInspector() {
  const mod = useStudio((s) => entryModule(s.project));
  const analysis = useStudio((s) => s.analysis);
  const setClassName = useStudio((s) => s.setClassName);
  const selectNode = useStudio((s) => s.selectNode);
  const errors = analysis.diagnostics.filter((d) => d.severity === 'error').length;

  return (
    <>
      <Section title="模型">
        <label className="mb-1 block text-[12px] text-muted">类名</label>
        <input className={field} value={mod.className} onChange={(e) => setClassName(e.target.value)} spellCheck={false} />
        <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[12px]">
          <dt className="text-muted">节点</dt>
          <dd className="font-mono">{Object.keys(mod.nodes).length}</dd>
          <dt className="text-muted">参数量</dt>
          <dd className="font-mono">
            {analysis.totalParams.toLocaleString()} <span className="text-muted">({formatCount(analysis.totalParams)})</span>
          </dd>
        </dl>
      </Section>
      <Section title={`问题（${analysis.diagnostics.length}）`}>
        {analysis.diagnostics.length === 0 ? (
          <p className="text-[12px] text-success">形状检查通过，可以导出。</p>
        ) : (
          <ul className="space-y-1.5">
            {analysis.diagnostics.map((d, i) => (
              <li key={i}>
                <button
                  type="button"
                  disabled={!d.nodeId}
                  onClick={() => d.nodeId && selectNode(d.nodeId)}
                  className={`text-left text-[12px] leading-relaxed ${d.severity === 'error' ? 'text-error' : 'text-warning'} enabled:hover:underline`}
                >
                  {d.nodeId && <span className="font-mono">{mod.nodes[d.nodeId]?.name}：</span>}
                  {d.message}
                </button>
              </li>
            ))}
          </ul>
        )}
        {errors > 0 && <p className="mt-2 text-[11px] text-muted">有错误时仍可导出，代码中会留下 TODO(neuroloom) 标记。</p>}
      </Section>
      <Section title="提示">
        <p className="text-[12px] leading-relaxed text-muted">选中一个节点查看和修改参数。拖动输出端口到另一个节点的输入端口即可连线。</p>
      </Section>
    </>
  );
}

export function Inspector() {
  const selected = useSelectedNodeId();
  return (
    <aside className="min-h-0 overflow-y-auto border-l border-line bg-surface">
      {selected ? <NodeInspector key={selected} nodeId={selected} /> : <ModelInspector />}
    </aside>
  );
}
