import type { ModuleDef, NodeId } from '@neuroloom/core';
import type { ELK as ElkInstance, ElkNode } from 'elkjs/lib/elk-api';

// ELK is large (~1.4 MB); load it the first time a layout is requested.
let elk: Promise<ElkInstance> | undefined;
const getElk = () => (elk ??= import('elkjs/lib/elk.bundled.js').then(({ default: ELK }) => new ELK()));

/** Below this zoom, node text gets hard to read, so a single row is wrapped. */
const READABLE_ZOOM = 0.6;

/**
 * Layered left-to-right layout (PRD F1.5). Sizes come from React Flow's
 * measurements. Long chains (a deep CNN, a U-Net) wrap into rows so the whole
 * graph fits the canvas at a readable size; short ones stay on one row.
 */
export async function computeLayout(
  mod: ModuleDef,
  sizes: Map<NodeId, { width: number; height: number }>,
  canvas: { width: number; height: number } = { width: 1200, height: 750 },
): Promise<Record<NodeId, { x: number; y: number }>> {
  const aspectRatio = canvas.width / Math.max(canvas.height, 1);
  const fitZoom = (r: ElkNode) => Math.min(canvas.width / Math.max(r.width ?? 1, 1), canvas.height / Math.max(r.height ?? 1, 1));
  const engine = await getElk();
  const run = (wrap: boolean) =>
    engine.layout({
      id: 'root',
      layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': 'RIGHT',
        'elk.layered.spacing.nodeNodeBetweenLayers': '90',
        'elk.spacing.nodeNode': '48',
        'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
        ...(wrap ? { 'elk.layered.wrapping.strategy': 'MULTI_EDGE', 'elk.aspectRatio': String(aspectRatio) } : {}),
      },
      children: Object.keys(mod.nodes).map((id) => ({
        id,
        width: sizes.get(id)?.width ?? 180,
        height: sizes.get(id)?.height ?? 72,
      })),
      edges: mod.edges.map((e) => ({ id: e.id, sources: [e.source.node], targets: [e.target.node] })),
    });

  let result: ElkNode = await run(false);
  if (fitZoom(result) < READABLE_ZOOM) {
    const wrapped = await run(true);
    if (fitZoom(wrapped) > fitZoom(result)) result = wrapped;
  }
  return Object.fromEntries((result.children ?? []).map((c) => [c.id, { x: Math.round(c.x ?? 0), y: Math.round(c.y ?? 0) }]));
}
