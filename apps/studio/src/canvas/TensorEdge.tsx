import { formatShape, type Dim } from '@neuroloom/core';
import { BaseEdge, EdgeLabelRenderer, getBezierPath, type Edge, type EdgeProps } from '@xyflow/react';
import { memo } from 'react';

export type TensorEdgeType = Edge<{ shape?: Dim[] }, 'tensor'>;

const reducedMotion =
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

const FLOW_SECONDS = 1.8;
const PARTICLES = [0, 1 / 3, 2 / 3];

/**
 * An edge is a tensor: it shows its shape and, when the shape is known, particles
 * flowing from source to target (PRD 8.1–8.2). Edges without a known shape are
 * drawn dashed and still.
 */
function TensorEdgeView({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, selected, data }: EdgeProps<TensorEdgeType>) {
  const [path, labelX, labelY] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  const shape = data?.shape;
  const live = shape !== undefined;

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={{
          stroke: live ? 'var(--nl-flow)' : 'var(--nl-border-strong)',
          strokeWidth: selected ? 2.75 : 1.75,
          strokeDasharray: live ? undefined : '5 5',
          opacity: live ? 0.85 : 1,
        }}
      />
      {live &&
        !reducedMotion &&
        PARTICLES.map((offset) => (
          <circle key={offset} r={2.75} fill="var(--nl-flow)" style={{ pointerEvents: 'none' }}>
            <animateMotion dur={`${FLOW_SECONDS}s`} begin={`${-offset * FLOW_SECONDS}s`} repeatCount="indefinite" path={path} />
          </circle>
        ))}
      {live && (
        <EdgeLabelRenderer>
          <div className="nl-edge-label" style={{ transform: `translate(-50%, -100%) translate(${labelX}px, ${labelY - 6}px)` }}>
            {formatShape(shape)}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

export const TensorEdge = memo(TensorEdgeView);
