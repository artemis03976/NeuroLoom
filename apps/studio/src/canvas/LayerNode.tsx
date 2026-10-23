import { entryModule, formatShape, getSpec, literalOf } from '@neuroloom/core';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { memo, type CSSProperties } from 'react';
import { formatCount } from '../lib/format';
import { useStudio, type FlowNode } from '../store';
import { categoryColor, glyphOf } from './glyph';

function LayerNodeView({ id, selected }: NodeProps<FlowNode>) {
  const node = useStudio((s) => entryModule(s.project).nodes[id]);
  const info = useStudio((s) => s.analysis.nodes[id]);
  const error = useStudio((s) => s.analysis.diagnostics.find((d) => d.nodeId === id && d.severity === 'error')?.message);
  if (!node) return null;
  const spec = getSpec(node.type);
  if (!spec) return <div className="nl-node has-error nl-node-error">未知节点 {node.type}</div>;

  // Show resolved values when inference succeeded, otherwise what the user typed.
  const typed = Object.fromEntries(Object.entries(node.params).map(([k, v]) => [k, literalOf(v) ?? 'auto']));
  const summary =
    node.kind === 'output'
      ? info?.outputs?.[0] && formatShape(info.outputs[0].shape)
      : spec.describe?.({ ...typed, ...info?.resolved });

  return (
    <div
      className={`nl-node${selected ? ' selected' : ''}${error ? ' has-error' : ''}`}
      style={{ '--cat': categoryColor(spec) } as CSSProperties}
    >
      {spec.inputs.map((p) => (
        <Handle key={p.name} id={p.name} type="target" position={Position.Left} className="nl-handle" />
      ))}
      <div className="nl-node-head">
        <span className="nl-node-glyph">{glyphOf(spec)}</span>
        <div className="min-w-0">
          <div className="nl-node-name">{node.name}</div>
          <div className="nl-node-type">{spec.displayName.en}</div>
        </div>
      </div>
      {(summary || (info?.paramCount ?? 0) > 0) && (
        <div className="nl-node-body">
          <span>{summary}</span>
          {info && info.paramCount > 0 && <span className="nl-badge">{formatCount(info.paramCount)}</span>}
        </div>
      )}
      {error && <div className="nl-node-error">{error}</div>}
      {spec.outputs.map((p) => (
        <Handle key={p.name} id={p.name} type="source" position={Position.Right} className="nl-handle" />
      ))}
    </div>
  );
}

export const LayerNode = memo(LayerNodeView);
