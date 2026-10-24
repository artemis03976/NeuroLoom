import { Handle, Position, useUpdateNodeInternals, type NodeProps } from '@xyflow/react';
import { memo, useEffect, useRef, type CSSProperties } from 'react';
import { tr } from '../i18n';
import { formatCount } from '../lib/format';
import { useStudio, type FlowNode } from '../store';
import { unknownLabel } from './face';
import { useReplayStatus } from './replay';

/**
 * A node draws its precomputed face (canvas/face.ts), so it renders again only
 * when what it shows changes. Simplified drawing when zoomed out and the
 * replay's glow are classes; styles.css does the rest.
 */
function LayerNodeView({ id, selected }: NodeProps<FlowNode>) {
  const face = useStudio((s) => s.faces[id]);
  const replay = useReplayStatus(id);

  // React Flow must re-measure when handles appear or disappear (variadic inputs,
  // chunk outputs). Not on mount: that would disturb the initial fitView.
  const updateInternals = useUpdateNodeInternals();
  const handleKey = face ? `${face.inputs.map((h) => h.id).join(',')}/${face.outputs.map((h) => h.id).join(',')}` : '';
  const previousKey = useRef(handleKey);
  useEffect(() => {
    if (previousKey.current === handleKey) return;
    previousKey.current = handleKey;
    updateInternals(id);
  }, [handleKey, id, updateInternals]);

  if (!face) return null;
  if (face.unknown) {
    return (
      <div className="nl-node has-error nl-node-error">
        {unknownLabel()} {face.type}
      </div>
    );
  }

  const { inputs, outputs, labelled } = face;
  const rows = Math.max(inputs.length, outputs.length);
  const kindClass = face.kind === 'module' ? ' nl-node-module' : face.kind === 'code' ? ' nl-node-code' : '';
  const replayClass = replay === 'idle' ? '' : ` replay-${replay}`;

  return (
    <div
      className={`nl-node${kindClass}${face.repeat ? ' nl-node-repeat' : ''}${selected ? ' selected' : ''}${face.error ? ' has-error' : ''}${replayClass}`}
      style={{ '--cat': face.color } as CSSProperties}
      title={face.kind === 'module' ? tr('双击进入模块', 'Double-click to open the module') : undefined}
    >
      {!labelled && inputs.map((h) => <Handle key={h.id} id={h.id} type="target" position={Position.Left} className="nl-handle" />)}
      <div className="nl-node-head">
        <span className="nl-node-glyph">{face.glyph}</span>
        <div className="min-w-0 flex-1">
          <div className="nl-node-name">{face.name}</div>
          <div className="nl-node-type">{face.type}</div>
        </div>
        {face.repeat && <span className="nl-repeat-badge">×{face.copies ?? '?'}</span>}
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
              {input && <Handle id={input.id} type="target" position={Position.Left} className={`nl-handle${input.optional ? ' optional' : ''}`} />}
              {output && <Handle id={output.id} type="source" position={Position.Right} className="nl-handle" />}
            </div>
          );
        })}
      {(face.summary || face.params > 0) && (
        <div className={`nl-node-body${face.kind === 'code' ? ' nl-node-code-line' : ''}`}>
          <span className="truncate">{face.summary}</span>
          {face.params > 0 && <span className="nl-badge">{formatCount(face.params)}</span>}
        </div>
      )}
      {face.error && <div className="nl-node-error">{face.error}</div>}
      {!labelled && outputs.map((h) => <Handle key={h.id} id={h.id} type="source" position={Position.Right} className="nl-handle" />)}
    </div>
  );
}

export const LayerNode = memo(LayerNodeView);
