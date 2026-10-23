// Pure editing operations on a module. Every function returns a new ModuleDef.
import { newId, uniqueName, wouldCreateCycle } from './graph';
import type { ModuleDef, NodeId, ParamValue, PortRef } from './ir';
import { getSpec, initialParams } from './specs';

export function addNode(mod: ModuleDef, type: string, position: { x: number; y: number }): { mod: ModuleDef; nodeId: NodeId } {
  const spec = getSpec(type);
  if (!spec) throw new Error(`Unknown node type ${type}`);
  const id = newId('n');
  const name = uniqueName(mod, spec.defaultName, spec.kind === 'input' || spec.kind === 'output');
  return {
    nodeId: id,
    mod: {
      ...mod,
      nodes: { ...mod.nodes, [id]: { id, kind: spec.kind, type, name, params: initialParams(spec) } },
      layout: { ...mod.layout, [id]: position },
    },
  };
}

export function removeNodes(mod: ModuleDef, ids: NodeId[]): ModuleDef {
  const drop = new Set(ids);
  const nodes = { ...mod.nodes };
  const layout = { ...mod.layout };
  for (const id of drop) {
    delete nodes[id];
    delete layout[id];
  }
  return { ...mod, nodes, layout, edges: mod.edges.filter((e) => !drop.has(e.source.node) && !drop.has(e.target.node)) };
}

export function removeEdges(mod: ModuleDef, ids: string[]): ModuleDef {
  const drop = new Set(ids);
  return { ...mod, edges: mod.edges.filter((e) => !drop.has(e.id)) };
}

/**
 * Connects two ports. An input port holds one edge, so a new connection replaces
 * the old one. Returns null when the edge would create a cycle.
 */
export function connect(mod: ModuleDef, source: PortRef, target: PortRef): ModuleDef | null {
  if (wouldCreateCycle(mod, source.node, target.node)) return null;
  const edges = mod.edges.filter((e) => !(e.target.node === target.node && e.target.port === target.port));
  return { ...mod, edges: [...edges, { id: newId('e'), source, target }] };
}

export function setParam(mod: ModuleDef, nodeId: NodeId, name: string, value: ParamValue): ModuleDef {
  const node = mod.nodes[nodeId];
  if (!node) return mod;
  return { ...mod, nodes: { ...mod.nodes, [nodeId]: { ...node, params: { ...node.params, [name]: value } } } };
}

export function renameNode(mod: ModuleDef, nodeId: NodeId, name: string): ModuleDef {
  const node = mod.nodes[nodeId];
  if (!node) return mod;
  return { ...mod, nodes: { ...mod.nodes, [nodeId]: { ...node, name } } };
}

export function setLayout(mod: ModuleDef, positions: Record<NodeId, { x: number; y: number }>): ModuleDef {
  return { ...mod, layout: { ...mod.layout, ...positions } };
}
