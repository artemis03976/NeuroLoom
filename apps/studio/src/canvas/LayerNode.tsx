import { entryModule, formatShape, getSpec, literalOf, outputsOf, variadicIndex, withDefaults, type Literal } from '@neuroloom/core';
import { Handle, Position, useUpdateNodeInternals, type NodeProps } from '@xyflow/react';
import { memo, useEffect, useRef, type CSSProperties } from 'react';
import { formatCount } from '../lib/format';
import { useStudio, type FlowNode } from '../store';
import { categoryColor, glyphOf } from './glyph';

interface HandleInfo {
  id: string;
  label: string;
  optional?: boolean;
}

function LayerNodeView({ id, selected }: NodeProps<FlowNode>) {
  const node = useStudio((s) => entryModule(s.project).nodes[id]);
  const info = useStudio((s) => s.analysis.nodes[id]);
  const error = useStudio((s) => s.analysis.diagnostics.find((d) => d.nodeId === id && d.severity === 'error')?.message);
  // Ports used by edges into this node; variadic inputs grow one handle past the last connected one.
  const targetPorts = useStudio((s) =>
    entryModule(s.project)
      .edges.filter((e) => e.target.node === id)
      .map((e) => e.target.port)
      .sort()
      .join('|'),
  );
  const spec = getSpec(node?.type);
  const params = node && spec ? withDefaults(spec, node.params) : {};

  const inputs: HandleInfo[] = (spec?.inputs ?? []).flatMap((p) => {
    if (!p.variadic) return [{ id: p.name, label: p.name, optional: p.optional }];
    const used = targetPorts.split('|').filter((port) => variadicIndex(port, p.name) !== undefined).length;
    return Array.from({ length: used + 1 }, (_, i) => ({ id: `${p.name}${i}`, label: `${p.name}${i}`, optional: i === used }));
  });
  const outputs: HandleInfo[] = spec ? outputsOf(spec, params).map((o) => ({ id: o.name, label: o.name })) : [];
  const labelled = inputs.length > 1 || outputs.length > 1;

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
  if (!spec) return <div className="nl-node has-error nl-node-error">未知节点 {node.type}</div>;

  // Show resolved values when inference succeeded, otherwise what the user typed.
  const typed = Object.fromEntries(Object.entries(params).map(([k, v]) => [k, literalOf(v) ?? ('auto' as Literal)]));
  const summary =
    node.kind === 'output' ? info?.outputs?.[0] && formatShape(info.outputs[0].shape) : spec.describe?.({ ...typed, ...info?.resolved });
  const rows = Math.max(inputs.length, outputs.length);

  return (
    <div
      className={`nl-node${selected ? ' selected' : ''}${error ? ' has-error' : ''}`}
      style={{ '--cat': categoryColor(spec) } as CSSProperties}
    >
      {!labelled &&
        inputs.map((h) => <Handle key={h.id} id={h.id} type="target" position={Position.Left} className="nl-handle" />)}
      <div className="nl-node-head">
        <span className="nl-node-glyph">{glyphOf(spec)}</span>
        <div className="min-w-0">
          <div className="nl-node-name">{node.name}</div>
          <div className="nl-node-type">{spec.displayName.en}</div>
        </div>
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
        <div className="nl-node-body">
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
