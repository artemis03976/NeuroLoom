import type { ModuleDef, NodeId } from '@neuroloom/core';
import type { ELK as ElkInstance } from 'elkjs/lib/elk-api';

// ELK is large (~1.4 MB); load it the first time a layout is requested.
let elk: Promise<ElkInstance> | undefined;
const getElk = () => (elk ??= import('elkjs/lib/elk.bundled.js').then(({ default: ELK }) => new ELK()));

/** Layered left-to-right layout (PRD F1.5). Sizes come from React Flow's measurements. */
export async function computeLayout(
  mod: ModuleDef,
  sizes: Map<NodeId, { width: number; height: number }>,
): Promise<Record<NodeId, { x: number; y: number }>> {
  const result = await (await getElk()).layout({
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'RIGHT',
      'elk.layered.spacing.nodeNodeBetweenLayers': '90',
      'elk.spacing.nodeNode': '48',
      'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
    },
    children: Object.keys(mod.nodes).map((id) => ({
      id,
      width: sizes.get(id)?.width ?? 180,
      height: sizes.get(id)?.height ?? 72,
    })),
    edges: mod.edges.map((e) => ({ id: e.id, sources: [e.source.node], targets: [e.target.node] })),
  });
  return Object.fromEntries((result.children ?? []).map((c) => [c.id, { x: Math.round(c.x ?? 0), y: Math.round(c.y ?? 0) }]));
}
