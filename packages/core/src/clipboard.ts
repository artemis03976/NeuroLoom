// Copy/paste of node selections (PRD F1.2). The clipboard holds an IR fragment,
// so it works across projects and windows; names are made unique on paste.
import { newId, uniqueName } from './graph';
import { SCHEMA_VERSION, type Edge, type ModuleDef, type NodeId, type NodeInst } from './ir';
import { getSpec } from './specs';

export const CLIP_KIND = 'neuroloom/clip';

export interface Clip {
  kind: typeof CLIP_KIND;
  schemaVersion: string;
  nodes: NodeInst[];
  /** Only edges between copied nodes. */
  edges: Edge[];
  layout: Record<NodeId, { x: number; y: number }>;
}

export function copyNodes(mod: ModuleDef, ids: NodeId[]): Clip {
  const picked = new Set(ids.filter((id) => mod.nodes[id]));
  return {
    kind: CLIP_KIND,
    schemaVersion: SCHEMA_VERSION,
    nodes: [...picked].map((id) => structuredClone(mod.nodes[id]!)),
    edges: mod.edges.filter((e) => picked.has(e.source.node) && picked.has(e.target.node)).map((e) => structuredClone(e)),
    layout: Object.fromEntries([...picked].map((id) => [id, mod.layout[id] ?? { x: 0, y: 0 }])),
  };
}

/** A clip from clipboard text, or null when the text is something else. */
export function parseClip(text: string): Clip | null {
  try {
    const value = JSON.parse(text);
    if (value?.kind !== CLIP_KIND || value.schemaVersion !== SCHEMA_VERSION || !Array.isArray(value.nodes)) return null;
    if (!value.nodes.every((n: NodeInst) => getSpec(n.type) && typeof n.name === 'string')) return null;
    return value as Clip;
  } catch {
    return null;
  }
}

/** Pastes with fresh ids, moved by `offset`; clashing names get the next free one. */
export function pasteClip(mod: ModuleDef, clip: Clip, offset: { x: number; y: number }): { mod: ModuleDef; nodeIds: NodeId[] } {
  const idMap = new Map<NodeId, NodeId>();
  const next: ModuleDef = { ...mod, nodes: { ...mod.nodes }, layout: { ...mod.layout }, edges: [...mod.edges], interface: { inputs: [...mod.interface.inputs], outputs: [...mod.interface.outputs] } };

  for (const node of clip.nodes) {
    const spec = getSpec(node.type)!;
    const id = newId('n');
    idMap.set(node.id, id);
    const taken = Object.values(next.nodes).some((n) => n.name === node.name);
    const style = spec.kind === 'op' ? 'fx' : spec.kind === 'layer' ? 'numbered' : 'bare';
    const name = taken ? uniqueName(next, spec.kind === 'layer' ? spec.defaultName : node.name.replace(/_?\d+$/, '') || spec.defaultName, style) : node.name;
    next.nodes[id] = { ...structuredClone(node), id, name, kind: spec.kind };
    const at = clip.layout[node.id] ?? { x: 0, y: 0 };
    next.layout[id] = { x: at.x + offset.x, y: at.y + offset.y };
    if (spec.kind === 'input') next.interface.inputs.push(id);
    if (spec.kind === 'output') next.interface.outputs.push(id);
  }
  for (const e of clip.edges) {
    const source = idMap.get(e.source.node);
    const target = idMap.get(e.target.node);
    if (source && target) next.edges.push({ id: newId('e'), source: { ...e.source, node: source }, target: { ...e.target, node: target } });
  }
  return { mod: next, nodeIds: [...idMap.values()] };
}
