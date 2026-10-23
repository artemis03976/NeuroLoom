import {
  exprOf,
  formatShape,
  getSpec,
  isAuto,
  literalOf,
  literalSource,
  parseShape,
  portsOf,
  withDefaults,
  type Literal,
  type NodeAnalysis,
  type NodeInst,
  type ParamSpec,
  type ParamValue,
  type Repeat,
} from '@neuroloom/core';
import { Fragment, useState, type CSSProperties, type ReactNode } from 'react';
import { nodeBadge } from '../canvas/glyph';
import { attempt, unpackSelection } from '../commands';
import { pick, tr } from '../i18n';
import { fallbackValue } from '../lib/paramText';
import { useCurrentModule, useSelectedNodeId, useStudio } from '../store';
import { CodeArea, ExprParam, field, NamesField, Section, TextParam, ValueField } from './inspector/fields';
import { ModelInspector } from './inspector/ModelInspector';

function ParamRow({ node, spec, resolved }: { node: NodeInst; spec: ParamSpec; resolved?: Record<string, Literal> }) {
  const setParam = useStudio((s) => s.setParam);
  const value: ParamValue | undefined = node.params[spec.name] ?? (spec.default !== undefined ? { literal: spec.default } : undefined);
  const auto = isAuto(value);
  const expr = exprOf(value);
  const lit = literalOf(value);
  const isNone = !auto && expr === undefined && lit === null;
  const changed = expr !== undefined || (spec.default !== undefined && lit !== undefined && JSON.stringify(lit) !== JSON.stringify(spec.default));
  const set = (v: ParamValue) => setParam(node.id, spec.name, v);

  let control: ReactNode;
  if (expr !== undefined) {
    control = <ExprParam text={expr} resolved={resolved?.[spec.name]} onCommit={set} />;
  } else if (spec.type === 'bool') {
    control = (
      <label className="flex items-center gap-2 text-[12.5px]">
        <input type="checkbox" checked={lit === true} onChange={(e) => set({ literal: e.target.checked })} />
        <span className="font-mono">{lit ? 'True' : 'False'}</span>
      </label>
    );
  } else if (spec.type === 'enum') {
    control = (
      <select className={field} value={String(lit ?? '')} disabled={isNone} onChange={(e) => set({ literal: e.target.value })}>
        {spec.options?.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    );
  } else if (spec.type === 'shape') {
    control = <input className={field} value={String(lit ?? '')} onChange={(e) => set({ literal: e.target.value })} spellCheck={false} />;
  } else {
    control = (
      <TextParam
        key={`${node.id}:${spec.name}:${auto}:${isNone}`}
        spec={spec}
        value={auto ? resolved?.[spec.name] : lit}
        disabled={auto || isNone}
        onCommit={set}
      />
    );
  }

  // ƒx switches between a literal and an expression over the module's hyperparameters.
  const toggleExpr = () =>
    set(expr !== undefined ? { literal: resolved?.[spec.name] ?? fallbackValue(spec) } : { expr: literalSource(resolved?.[spec.name] ?? lit ?? fallbackValue(spec)) });

  return (
    <div className="mb-2.5">
      <label className="mb-1 flex items-center justify-between gap-2 text-[12px]">
        <span className={changed ? 'font-semibold' : ''}>
          <span className="font-mono">{spec.name}</span>
          <span className="ml-1.5 text-muted">{pick(spec.label)}</span>
        </span>
        <span className="flex items-center gap-2 text-[11px] text-muted">
          {spec.allowAuto && expr === undefined && (
            <span className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={auto}
                onChange={(e) => set(e.target.checked ? { auto: true } : { literal: resolved?.[spec.name] ?? fallbackValue(spec) })}
              />
              {tr('自动推断', 'Auto')}
            </span>
          )}
          {spec.nullable && expr === undefined && (
            <span className="flex items-center gap-1">
              <input type="checkbox" checked={isNone} onChange={(e) => set({ literal: e.target.checked ? null : fallbackValue(spec) })} />
              None
            </span>
          )}
          {spec.type !== 'shape' && (
            <button
              type="button"
              onClick={toggleExpr}
              title={expr !== undefined ? tr('改回固定值', 'Use a fixed value') : tr('改用表达式（可引用模块超参数）', 'Use an expression over the module hyperparameters')}
              className={`rounded px-1 font-mono italic ${expr !== undefined ? 'bg-primary-soft text-primary' : 'hover:bg-surface-2'}`}
            >
              ƒx
            </button>
          )}
        </span>
      </label>
      {control}
    </div>
  );
}

function Header({ node, children }: { node: NodeInst; children?: ReactNode }) {
  const project = useStudio((s) => s.project);
  const badge = nodeBadge(project, node);
  return (
    <div className="flex items-center gap-3 border-b border-line px-4 py-3">
      <span
        className="grid size-8 flex-none place-items-center rounded-lg font-mono text-[11px] font-bold text-white"
        style={{ background: badge.color } as CSSProperties}
      >
        {badge.glyph}
      </span>
      <div className="min-w-0 flex-1">
        <div className="truncate text-[13px] font-semibold">{badge.title}</div>
        <div className="truncate font-mono text-[11px] text-muted">{node.kind === 'module' ? badge.type : node.type}</div>
      </div>
      {children}
    </div>
  );
}

function NameSection({ node }: { node: NodeInst }) {
  const rename = useStudio((s) => s.rename);
  return (
    <Section title={node.kind === 'code' ? tr('名称', 'Name') : tr('变量名', 'Variable name')}>
      <input className={field} value={node.name} onChange={(e) => rename(node.id, e.target.value)} spellCheck={false} />
    </Section>
  );
}

function TensorSection({ node, info }: { node: NodeInst; info?: NodeAnalysis }) {
  const project = useStudio((s) => s.project);
  const ports = portsOf(project, node);
  const inputs = ports?.inputs ?? [];
  const outputs = node.kind === 'output' ? [] : (ports?.outputs ?? []);
  const shown = (i: number) => (info?.outputs?.[i] ? formatShape(info.outputs[i]!.shape) : '—');
  return (
    <Section title={tr('张量', 'Tensors')}>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[12px]">
        {info?.inputs.map((t, i) => (
          <Fragment key={`in${i}`}>
            <dt className="text-muted">{info.inputs.length > 1 ? `${tr('输入', 'in')} ${inputs[i]?.label ?? i}` : tr('输入', 'in')}</dt>
            <dd>{t ? formatShape(t.shape) : '—'}</dd>
          </Fragment>
        ))}
        {outputs.map((o, i) => (
          <Fragment key={o.name}>
            <dt className="text-muted">{outputs.length > 1 ? `${tr('输出', 'out')} ${o.label}` : tr('输出', 'out')}</dt>
            <dd>{shown(i)}</dd>
          </Fragment>
        ))}
        {(node.kind === 'layer' || node.kind === 'module') && (
          <>
            <dt className="text-muted">{tr('参数量', 'Parameters')}</dt>
            <dd>{(info?.paramCount ?? 0).toLocaleString()}</dd>
          </>
        )}
      </dl>
    </Section>
  );
}

function Issues({ nodeId }: { nodeId: string }) {
  const all = useStudio((s) => s.viewAnalysis.diagnostics);
  const diagnostics = all.filter((d) => d.nodeId === nodeId);
  if (diagnostics.length === 0) return null;
  return (
    <Section title={tr('问题', 'Problems')}>
      {diagnostics.map((d, i) => (
        <p key={i} className={`text-[12px] leading-relaxed ${d.severity === 'error' ? 'text-error' : 'text-warning'}`}>
          {d.message}
        </p>
      ))}
    </Section>
  );
}

/**
 * Repeat ×N (PRD F4.4): the count may be an expression, and the first copy may
 * take different values (a ResNet stage's stride=2).
 */
function RepeatSection({ node, fields }: { node: NodeInst; fields: { name: string; label?: string }[] }) {
  const setRepeat = useStudio((s) => s.setRepeat);
  const [showFirst, setShowFirst] = useState(!!node.repeat?.first);
  const repeat = node.repeat;
  const update = (patch: Partial<Repeat>) => {
    if (!repeat) return;
    const next: Repeat = { ...repeat, ...patch };
    if (next.first && Object.keys(next.first).length === 0) delete next.first;
    setRepeat(node.id, next);
  };
  return (
    <Section
      title={tr('重复', 'Repeat')}
      aside={
        <label className="flex items-center gap-1 font-normal">
          <input type="checkbox" checked={!!repeat} onChange={(e) => setRepeat(node.id, e.target.checked ? { count: { literal: 2 } } : undefined)} />
          {tr('串联重复', 'Repeat in series')}
        </label>
      }
    >
      {!repeat ? (
        <p className="text-[11.5px] leading-relaxed text-muted">
          {tr('把这个节点串联重复 N 次，导出为 nn.ModuleList 加 for 循环。', 'Chain N copies of this node, exported as an nn.ModuleList with a for loop.')}
        </p>
      ) : (
        <>
          <label className="mb-1 block text-[12px]">{tr('次数（整数或表达式）', 'Count (integer or expression)')}</label>
          <ValueField value={repeat.count} onCommit={(v) => v && update({ count: v })} placeholder="4" />
          <button type="button" className="mt-2 text-[11.5px] text-primary hover:underline" onClick={() => setShowFirst((s) => !s)}>
            {showFirst ? '▾' : '▸'} {tr('第一份使用不同的值', 'Different values for the first copy')}
            {repeat.first && ` (${Object.keys(repeat.first).length})`}
          </button>
          {showFirst && (
            <div className="mt-2 space-y-2">
              {fields.map((f) => (
                <div key={f.name}>
                  <label className="mb-0.5 block font-mono text-[11.5px] text-muted">{f.label ?? f.name}</label>
                  <ValueField
                    value={repeat.first?.[f.name]}
                    placeholder={tr('同其余份', 'same as the rest')}
                    onCommit={(v) => {
                      const first = { ...repeat.first };
                      if (v) first[f.name] = v;
                      else delete first[f.name];
                      update({ first });
                    }}
                  />
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </Section>
  );
}

function SpecNodeInspector({ node, info }: { node: NodeInst; info?: NodeAnalysis }) {
  const mod = useCurrentModule();
  const isEntry = useStudio((s) => s.project.entry === s.view.moduleId);
  const spec = getSpec(node.type);
  if (!spec) return <Section title={tr('节点', 'Node')}>{tr('未知类型', 'Unknown type')} {node.type}</Section>;
  const params = withDefaults(spec, node.params);
  const fromCallSite = node.kind === 'input' && !isEntry;
  return (
    <>
      <Header node={node} />
      <NameSection node={node} />
      {spec.params.length > 0 && (
        <Section title={tr('参数', 'Parameters')}>
          {fromCallSite && (
            <p className="mb-2 text-[11.5px] leading-relaxed text-muted">
              {tr('子模块的输入形状来自调用处；这里的形状只在单独查看本模块时使用，可留空。', 'A submodule input takes its shape from the call site; this one is only used when the module is viewed on its own and may be empty.')}
            </p>
          )}
          {spec.params.map((p) => (
            <ParamRow key={p.name} node={{ ...node, params }} spec={p} resolved={info?.resolved} />
          ))}
        </Section>
      )}
      {node.kind === 'layer' && !mod.sequential && <RepeatSection node={node} fields={spec.params.map((p) => ({ name: p.name }))} />}
      <TensorSection node={node} info={info} />
      <Issues nodeId={node.id} />
      <Section title={tr('说明', 'About')}>
        <p className="text-[12px] leading-relaxed text-muted">{pick(spec.summary)}</p>
        {spec.docUrl && (
          <a className="mt-1 inline-block text-[12px] text-primary hover:underline" href={spec.docUrl} target="_blank" rel="noreferrer">
            {tr('PyTorch 官方文档 ↗', 'PyTorch documentation ↗')}
          </a>
        )}
      </Section>
    </>
  );
}

function ModuleNodeInspector({ node, info }: { node: NodeInst; info?: NodeAnalysis }) {
  const project = useStudio((s) => s.project);
  const mod = useCurrentModule();
  const { setParam, enterModule, makeUnique } = useStudio.getState();
  const def = project.modules[node.type];
  if (!def) return <Section title={tr('模块', 'Module')}>{tr('模块定义不存在', 'The module definition is missing')}</Section>;
  const uses = Object.values(project.modules).reduce((n, m) => n + Object.values(m.nodes).filter((x) => x.kind === 'module' && x.type === def.id).length, 0);
  const verbatim = !!def.opaque && (!def.opaque.source || !!def.opaque.verbatimArgs);
  const button = 'rounded-md border border-line px-2.5 py-1 text-[12px] hover:bg-surface-2 disabled:opacity-40';

  return (
    <>
      <Header node={node}>
        <button type="button" className={button} onClick={() => enterModule(node.id)}>
          {def.opaque ? tr('源码', 'Source') : tr('进入', 'Open')}
        </button>
      </Header>
      <NameSection node={node} />
      {def.sequential ? (
        <Section title={tr('参数', 'Parameters')}>
          <p className="text-[11.5px] leading-relaxed text-muted">
            {tr('nn.Sequential 没有自己的超参数，里面的层直接使用当前模块的超参数。', 'An nn.Sequential has no hyperparameters of its own; its layers use those of this module.')}
          </p>
        </Section>
      ) : (
        <Section title={tr('构造参数', 'Constructor arguments')}>
          {def.hyperparams.length === 0 && <p className="text-[12px] text-muted">{tr(`${def.className} 没有构造参数`, `${def.className} takes no arguments`)}</p>}
          {def.hyperparams.map((hp) => {
            const resolved = info?.resolved?.[hp.name];
            const required = !hp.default && !verbatim;
            const label = hp.name.startsWith('#') ? tr(`第 ${Number(hp.name.slice(1)) + 1} 个位置参数`, `positional ${Number(hp.name.slice(1)) + 1}`) : hp.name;
            return (
              <div key={hp.name} className="mb-2.5">
                <label className="mb-1 flex items-center justify-between text-[12px]">
                  <span className="font-mono">
                    {label}
                    {hp.annotation && <span className="ml-1 text-muted">: {hp.annotation}</span>}
                  </span>
                  {required && <span className="text-[11px] text-warning">{tr('必填', 'required')}</span>}
                </label>
                <ValueField
                  value={node.params[hp.name]}
                  placeholder={hp.default ? ('literal' in hp.default ? literalSource(hp.default.literal) : hp.default.expr) : ''}
                  hint={resolved === undefined || resolved === null ? undefined : literalSource(resolved)}
                  onCommit={(v) => setParam(node.id, hp.name, v)}
                />
              </div>
            );
          })}
          {def.hyperparams.length > 0 && (
            <p className="text-[11px] leading-relaxed text-muted">
              {tr('可填数值，或用表达式引用当前模块的超参数（如 width * 2）。留空使用默认值。', 'Enter a value, or an expression over this module’s hyperparameters (such as width * 2). Empty means the default.')}
            </p>
          )}
        </Section>
      )}
      {!def.opaque && !mod.sequential && !def.sequential && <RepeatSection node={node} fields={def.hyperparams.map((h) => ({ name: h.name }))} />}
      <Section title={tr('模块', 'Module')}>
        <p className="mb-2 text-[11.5px] leading-relaxed text-muted">
          {uses > 1
            ? tr(`${def.className} 在项目中有 ${uses} 个实例，修改定义会影响全部实例。`, `${def.className} has ${uses} instances; editing the definition changes all of them.`)
            : tr(`这是 ${def.className} 唯一的实例。`, `This is the only instance of ${def.className}.`)}
        </p>
        <div className="flex flex-wrap gap-1.5">
          <button type="button" className={button} disabled={uses < 2} onClick={() => attempt(() => makeUnique(node.id))}>
            {tr('转为独立副本', 'Make independent copy')}
          </button>
          <button type="button" className={button} disabled={!!def.opaque || !!node.repeat} onClick={() => unpackSelection(node.id)}>
            {tr('解包', 'Unpack')}
          </button>
        </div>
      </Section>
      {def.opaque && <DeclaredShapes node={node} ports={def.interface.outputs.map((id) => ({ name: id, label: def.nodes[id]?.name ?? id }))} />}
      <TensorSection node={node} info={info} />
      <Issues nodeId={node.id} />
    </>
  );
}

/** Output shapes for what inference cannot see into: code blocks and opaque modules. */
function DeclaredShapes({ node, ports }: { node: NodeInst; ports: { name: string; label: string }[] }) {
  const setShapes = useStudio((s) => s.setShapes);
  if (ports.length === 0) return null;
  const set = (port: string, text: string) => {
    const shapes = { ...node.shapes };
    if (text.trim()) shapes[port] = text;
    else delete shapes[port];
    setShapes(node.id, Object.keys(shapes).length > 0 ? shapes : undefined);
  };
  return (
    <Section title={tr('输出形状（声明）', 'Output shapes (declared)')}>
      <p className="mb-2 text-[11.5px] leading-relaxed text-muted">
        {tr('形状推断看不到其中的计算；声明形状后，下游节点才能继续推断，例如 B, 512。', 'Inference cannot see inside; declare the shapes so downstream nodes can continue, e.g. B, 512.')}
      </p>
      {ports.map((p) => {
        const text = node.shapes?.[p.name] ?? '';
        const parsed = text.trim() ? parseShape(text.split(':')[0] ?? '') : undefined;
        return (
          <div key={p.name} className="mb-2">
            <label className="mb-0.5 block font-mono text-[11.5px] text-muted">{p.label}</label>
            <input className={`${field} ${parsed && !parsed.ok ? 'border-error' : ''}`} value={text} placeholder="B, 512" spellCheck={false} onChange={(e) => set(p.name, e.target.value)} />
          </div>
        );
      })}
    </Section>
  );
}

function CodeNodeInspector({ node, info }: { node: NodeInst; info?: NodeAnalysis }) {
  const setCode = useStudio((s) => s.setCode);
  const code = node.code ?? { source: '', inputs: [], outputs: [] };
  return (
    <>
      <Header node={node} />
      <NameSection node={node} />
      <Section title={tr('Python 代码', 'Python code')}>
        <CodeArea value={code.source} rows={Math.min(14, Math.max(4, code.source.split('\n').length + 1))} onCommit={(source) => setCode(node.id, { ...code, source })} />
        <p className="mt-1 text-[11px] leading-relaxed text-muted">
          {tr('原样写入 forward()。可以用 self.<名称> 调用本模块的层。', 'Written into forward() as is. Layers of this module are available as self.<name>.')}
        </p>
      </Section>
      <Section title={tr('变量', 'Variables')}>
        <label className="mb-1 block text-[12px]">{tr('读取（输入端口）', 'Reads (inputs)')}</label>
        <NamesField names={code.inputs} onCommit={(inputs) => setCode(node.id, { ...code, inputs })} />
        <label className="mt-2 mb-1 block text-[12px]">{tr('定义（输出端口）', 'Defines (outputs)')}</label>
        <NamesField names={code.outputs} onCommit={(outputs) => setCode(node.id, { ...code, outputs })} />
      </Section>
      <DeclaredShapes node={node} ports={code.outputs.map((name) => ({ name, label: name }))} />
      <TensorSection node={node} info={info} />
      <Issues nodeId={node.id} />
    </>
  );
}

function NodeInspector({ nodeId }: { nodeId: string }) {
  const node = useStudio((s) => s.project.modules[s.view.moduleId]?.nodes[nodeId]);
  const info = useStudio((s) => s.viewAnalysis.nodes[nodeId]);
  if (!node) return null;
  if (node.kind === 'module') return <ModuleNodeInspector node={node} info={info} />;
  if (node.kind === 'code') return <CodeNodeInspector node={node} info={info} />;
  return <SpecNodeInspector node={node} info={info} />;
}

export function Inspector() {
  const selected = useSelectedNodeId();
  return (
    <aside className="min-h-0 overflow-y-auto border-l border-line bg-surface">
      {selected ? <NodeInspector key={selected} nodeId={selected} /> : <ModelInspector />}
    </aside>
  );
}
