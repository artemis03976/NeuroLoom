import { BaseEdge, EdgeLabelRenderer, getBezierPath, type Edge, type EdgeProps } from '@xyflow/react';
import { memo } from 'react';
import { useFlow } from './flowMode';
import { useEdgeReplay } from './replay';

/** `label` is the formatted shape, absent while the shape is unknown. */
export type TensorEdgeType = Edge<{ label?: string }, 'tensor'>;

const FLOW_SECONDS = 1.8;
const PARTICLES = [0, 1 / 3, 2 / 3];

/**
 * An edge is a tensor: it shows its shape and, when the shape is known, data
 * flowing from source to target (PRD 8.1–8.2): particles, or dashes for large
 * or zoomed-out views. Edges without a known shape are drawn dashed and still.
 * During a replay (F3.5) only the edges into the running step move, and an
 * edge's shape appears once its source has run.
 */
function TensorEdgeView({ id, source, target, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, selected, data }: EdgeProps<TensorEdgeType>) {
  const [path, labelX, labelY] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  const mode = useFlow((s) => s.mode);
  const inLineage = useFlow((s) => !s.lineage || s.lineage.has(id));
  const replay = useEdgeReplay(source, target);
  const label = data?.label;
  const live = label !== undefined;

  let motion = live && inLineage ? mode : 'still';
  if (replay !== 'idle') motion = replay === 'flowing' && mode !== 'still' ? 'particles' : 'still';
  const faded = (!inLineage && replay === 'idle') || replay === 'waiting';
  const seconds = replay === 'flowing' ? FLOW_SECONDS / 2 : FLOW_SECONDS;

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        className={motion === 'dashes' ? 'nl-edge-dashes' : undefined}
        style={{
          stroke: live ? 'var(--nl-flow)' : 'var(--nl-border-strong)',
          strokeWidth: selected ? 2.75 : 1.75,
          strokeDasharray: !live ? '5 5' : motion === 'dashes' ? '6 6' : undefined,
          opacity: faded ? 0.2 : live ? 0.85 : 1,
        }}
      />
      {motion === 'particles' &&
        PARTICLES.map((offset) => (
          <circle key={offset} r={2.75} fill="var(--nl-flow)" style={{ pointerEvents: 'none' }}>
            <animateMotion dur={`${seconds}s`} begin={`${-offset * seconds}s`} repeatCount="indefinite" path={path} />
          </circle>
        ))}
      {live && replay !== 'waiting' && (
        <EdgeLabelRenderer>
          <div
            className={`nl-edge-label${replay === 'ready' ? ' nl-pop' : ''}${faded ? ' opacity-30' : ''}`}
            style={{ transform: `translate(-50%, -100%) translate(${labelX}px, ${labelY - 6}px)` }}
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

export const TensorEdge = memo(TensorEdgeView);
