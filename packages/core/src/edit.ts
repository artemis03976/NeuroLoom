// Pure editing operations on a module. Every function returns a new ModuleDef.
import { variadicEdges } from './analyze';
import { newId, uniqueName, wouldCreateCycle } from './graph';
import type { ModuleDef, NodeId, ParamValue, PortRef } from './ir';
import { getSpec, initialParams } from './specs';

export function addNode(mod: ModuleDef, type: string, position: { x: number; y: number }): { mod: ModuleDef; nodeId: NodeId } {
  const spec = getSpec(type);
  if (!spec) throw new Error(`Unknown node type ${type}`);
  const id = newId('n');
  const style = spec.kind === 'op' ? 'fx' : spec.kind === 'layer' ? 'numbered' : 'bare';
  const name = uniqueName(mod, spec.defaultName, style);
  // New inputs and outputs go to the end of the forward() signature / return tuple.
  const { inputs, outputs } = mod.interface;
  return {
    nodeId: id,
    mod: {
      ...mod,
      nodes: { ...mod.nodes, [id]: { id, kind: spec.kind, type, name, params: initialParams(spec) } },
      layout: { ...mod.layout, [id]: position },
      interface: {
        ...mod.interface,
        inputs: spec.kind === 'input' ? [...inputs, id] : inputs,
        outputs: spec.kind === 'output' ? [...outputs, id] : outputs,
      },
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
  return compactVariadic({
    ...mod,
    nodes,
    layout,
    edges: mod.edges.filter((e) => !drop.has(e.source.node) && !drop.has(e.target.node)),
    interface: {
      inputs: mod.interface.inputs.filter((id) => !drop.has(id)),
      outputs: mod.interface.outputs.filter((id) => !drop.has(id)),
      ...(mod.interface.defaults ? { defaults: Object.fromEntries(Object.entries(mod.interface.defaults).filter(([id]) => !drop.has(id))) } : {}),
    },
  });
}

export function removeEdges(mod: ModuleDef, ids: string[]): ModuleDef {
  const drop = new Set(ids);
  return compactVariadic({ ...mod, edges: mod.edges.filter((e) => !drop.has(e.id)) });
}

/** Renumbers variadic ports (in0, in2 -> in0, in1) so the list has no gaps after a removal. */
export function compactVariadic(mod: ModuleDef): ModuleDef {
  const renamed = new Map<string, string>();
  for (const node of Object.values(mod.nodes)) {
    for (const port of getSpec(node.type)?.inputs ?? []) {
      if (!port.variadic) continue;
      variadicEdges(mod, node.id, port.name).forEach((e, i) => {
        if (e.target.port !== `${port.name}${i}`) renamed.set(e.id, `${port.name}${i}`);
      });
    }
  }
  if (renamed.size === 0) return mod;
  return { ...mod, edges: mod.edges.map((e) => (renamed.has(e.id) ? { ...e, target: { ...e.target, port: renamed.get(e.id)! } } : e)) };
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

/** Sets a parameter; `undefined` removes it (a module argument back to its default). */
export function setParam(mod: ModuleDef, nodeId: NodeId, name: string, value: ParamValue | undefined): ModuleDef {
  const node = mod.nodes[nodeId];
  if (!node) return mod;
  const params = { ...node.params };
  if (value === undefined) delete params[name];
  else params[name] = value;
  return { ...mod, nodes: { ...mod.nodes, [nodeId]: { ...node, params } } };
}

export function renameNode(mod: ModuleDef, nodeId: NodeId, name: string): ModuleDef {
  const node = mod.nodes[nodeId];
  if (!node) return mod;
  return { ...mod, nodes: { ...mod.nodes, [nodeId]: { ...node, name } } };
}

export function setLayout(mod: ModuleDef, positions: Record<NodeId, { x: number; y: number }>): ModuleDef {
  return { ...mod, layout: { ...mod.layout, ...positions } };
}
