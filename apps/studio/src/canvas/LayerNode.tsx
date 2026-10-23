import { formatShape, getSpec, literalOf, portsOf, variadicIndex, withDefaults, type Literal } from '@neuroloom/core';
import { Handle, Position, useUpdateNodeInternals, type NodeProps } from '@xyflow/react';
import { memo, useEffect, useRef, type CSSProperties } from 'react';
import { tr } from '../i18n';
import { formatCount } from '../lib/format';
import { useStudio, type FlowNode } from '../store';
import { nodeBadge } from './glyph';

interface HandleInfo {
  id: string;
  label: string;
  optional?: boolean;
}

function LayerNodeView({ id, selected }: NodeProps<FlowNode>) {
  const project = useStudio((s) => s.project);
  const node = useStudio((s) => s.project.modules[s.view.moduleId]?.nodes[id]);
  const info = useStudio((s) => s.viewAnalysis.nodes[id]);
  const error = useStudio((s) => s.viewAnalysis.diagnostics.find((d) => d.nodeId === id && d.severity === 'error')?.message);
  // Ports used by edges into this node; variadic inputs grow one handle past the last connected one.
  const targetPorts = useStudio((s) =>
    (s.project.modules[s.view.moduleId]?.edges ?? [])
      .filter((e) => e.target.node === id)
      .map((e) => e.target.port)
      .sort()
      .join('|'),
  );
  const ports = node ? portsOf(project, node) : undefined;

  const inputs: HandleInfo[] = (ports?.inputs ?? []).flatMap((p) => {
    if (!p.variadic) return [{ id: p.name, label: p.label, optional: p.optional }];
    const used = targetPorts.split('|').filter((port) => variadicIndex(port, p.name) !== undefined).length;
    return Array.from({ length: used + 1 }, (_, i) => ({ id: `${p.name}${i}`, label: `${p.label}${i}`, optional: i === used }));
  });
  const outputs: HandleInfo[] = (ports?.outputs ?? []).map((o) => ({ id: o.name, label: o.label }));
  const labelled = inputs.length > 1 || outputs.length > 1 || node?.kind === 'code';

  // React Flow must re-measure when handles appear or disappear (variadic inputs,
  // chunk outputs). Not on mount: that would disturb the initial fitView.
  const updateInternals = useUpdateNodeInternals();
  const handleKey = `${inputs.map((h) => h.id).join(',')}/${outputs.map((h) => h.id).join(',')}`;
  const previousKey = useRef(handleKey);
  useEffect(() => {
    if (previousKey.current === handleKey) return;
    previousKey.current = handleKey;
    updateInternals(id);
  }, [handleKey, id, updateInternals]);

  if (!node) return null;
  const badge = nodeBadge(project, node);
  if (!ports) {
    return <div className="nl-node has-error nl-node-error">{tr('未知节点', 'Unknown node')} {node.type}</div>;
  }

  let summary: string | undefined;
  if (node.kind === 'output') {
    summary = info?.inputs[0] ? formatShape(info.inputs[0].shape) : undefined;
  } else if (node.kind === 'input' && info?.outputs?.[0]) {
    // In a submodule the type comes from the call site, not the declared shape.
    const t = info.outputs[0];
    summary = `${formatShape(t.shape)}${t.dtype !== 'float32' ? ` ${t.dtype}` : ''}`;
  } else if (node.kind === 'module') {
    const values = Object.entries(info?.resolved ?? {}).filter(([k]) => !k.startsWith('#'));
    summary = values.map(([k, v]) => `${k}=${Array.isArray(v) ? v.join('×') : String(v)}`).join(' ');
  } else if (node.kind === 'code') {
    summary = node.code?.source.split('\n').find((l) => l.trim())?.trim();
  } else {
    // Show resolved values when inference succeeded, otherwise what the user typed.
    const spec = getSpec(node.type)!;
    const params = withDefaults(spec, node.params);
    const typed = Object.fromEntries(Object.entries(params).map(([k, v]) => [k, literalOf(v) ?? ('auto' as Literal)]));
    summary = spec.describe?.({ ...typed, ...info?.resolved });
  }
  const copies = node.repeat ? (info?.children?.length ?? (node.kind === 'layer' ? Number(literalOf(node.repeat.count)) || undefined : undefined)) : undefined;
  const rows = Math.max(inputs.length, outputs.length);
  const kindClass = node.kind === 'module' ? ' nl-node-module' : node.kind === 'code' ? ' nl-node-code' : '';

  return (
    <div
      className={`nl-node${kindClass}${node.repeat ? ' nl-node-repeat' : ''}${selected ? ' selected' : ''}${error ? ' has-error' : ''}`}
      style={{ '--cat': badge.color } as CSSProperties}
      title={node.kind === 'module' ? tr('双击进入模块', 'Double-click to open the module') : undefined}
    >
      {!labelled &&
        inputs.map((h) => <Handle key={h.id} id={h.id} type="target" position={Position.Left} className="nl-handle" />)}
      <div className="nl-node-head">
        <span className="nl-node-glyph">{badge.glyph}</span>
        <div className="min-w-0 flex-1">
          <div className="nl-node-name">{node.name}</div>
          <div className="nl-node-type">{badge.type}</div>
        </div>
        {node.repeat && <span className="nl-repeat-badge">×{copies ?? '?'}</span>}
      </div>
      {labelled &&
        Array.from({ length: rows }, (_, i) => {
          const input = inputs[i];
          const output = outputs[i];
          // Each row holds its own handles, so they line up with their labels.
          return (
            <div key={i} className="nl-node-port">
              <span className={input?.optional ? 'opacity-50' : ''}>{input?.label}</span>
              <span>{output?.label}</span>
              {input && (
                <Handle id={input.id} type="target" position={Position.Left} className={`nl-handle${input.optional ? ' optional' : ''}`} />
              )}
              {output && <Handle id={output.id} type="source" position={Position.Right} className="nl-handle" />}
            </div>
          );
        })}
      {(summary || (info?.paramCount ?? 0) > 0) && (
        <div className={`nl-node-body${node.kind === 'code' ? ' nl-node-code-line' : ''}`}>
          <span className="truncate">{summary}</span>
          {info && info.paramCount > 0 && <span className="nl-badge">{formatCount(info.paramCount)}</span>}
        </div>
      )}
      {error && <div className="nl-node-error">{error}</div>}
      {!labelled &&
        outputs.map((h) => <Handle key={h.id} id={h.id} type="source" position={Position.Right} className="nl-handle" />)}
    </div>
  );
}

export const LayerNode = memo(LayerNodeView);
