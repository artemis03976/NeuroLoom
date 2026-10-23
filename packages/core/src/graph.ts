import type { Edge, ModuleDef, NodeId, NodeInst } from './ir';

export const portKey = (node: NodeId, port: string): string => `${node}:${port}`;

/** Each input port accepts at most one edge, so incoming edges are keyed by port. */
export function incomingByPort(mod: ModuleDef): Map<string, Edge> {
  const map = new Map<string, Edge>();
  for (const e of mod.edges) map.set(portKey(e.target.node, e.target.port), e);
  return map;
}

export function consumers(mod: ModuleDef, node: NodeId, port: string): Edge[] {
  return mod.edges.filter((e) => e.source.node === node && e.source.port === port);
}

/**
 * Topological order. Ties between ready nodes are broken by canvas position
 * (left to right, then top to bottom) and finally by insertion order, so the
 * generated code reads in the same order as the picture.
 */
export function topoOrder(mod: ModuleDef): { order: NodeId[]; cyclic: NodeId[] } {
  const ids = Object.keys(mod.nodes);
  const index = new Map(ids.map((id, i) => [id, i]));
  const indegree = new Map(ids.map((id) => [id, 0]));
  const successors = new Map<NodeId, NodeId[]>(ids.map((id) => [id, []]));
  for (const e of mod.edges) {
    if (!indegree.has(e.source.node) || !indegree.has(e.target.node)) continue;
    indegree.set(e.target.node, indegree.get(e.target.node)! + 1);
    successors.get(e.source.node)!.push(e.target.node);
  }

  const sortKey = (id: NodeId): [number, number, number] => {
    const pos = mod.layout[id];
    return [pos?.x ?? 0, pos?.y ?? 0, index.get(id)!];
  };
  const compare = (a: NodeId, b: NodeId): number => {
    const ka = sortKey(a);
    const kb = sortKey(b);
    return ka[0] - kb[0] || ka[1] - kb[1] || ka[2] - kb[2];
  };

  const ready = ids.filter((id) => indegree.get(id) === 0);
  const order: NodeId[] = [];
  while (ready.length > 0) {
    ready.sort(compare);
    const id = ready.shift()!;
    order.push(id);
    for (const next of successors.get(id)!) {
      const d = indegree.get(next)! - 1;
      indegree.set(next, d);
      if (d === 0) ready.push(next);
    }
  }
  const placed = new Set(order);
  return { order, cyclic: ids.filter((id) => !placed.has(id)) };
}

/** forward() arguments or return values, in creation order (the IR's insertion order). */
export function interfaceOrder(mod: ModuleDef, kind: 'input' | 'output'): NodeInst[] {
  return Object.values(mod.nodes).filter((n) => n.kind === kind);
}

/** True if adding source -> target would close a cycle. */
export function wouldCreateCycle(mod: ModuleDef, source: NodeId, target: NodeId): boolean {
  if (source === target) return true;
  const stack = [target];
  const seen = new Set<NodeId>();
  while (stack.length > 0) {
    const id = stack.pop()!;
    if (id === source) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    for (const e of mod.edges) if (e.source.node === id) stack.push(e.target.node);
  }
  return false;
}

const PY_KEYWORDS = new Set(
  'False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield self'.split(
    ' ',
  ),
);

export function isValidIdentifier(name: string): boolean {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) && !PY_KEYWORDS.has(name);
}

/** "fc" -> "fc1", "fc2", ...; with `bare`, the first one is just "x". */
export function uniqueName(mod: ModuleDef, base: string, bare = false): string {
  const taken = new Set(Object.values(mod.nodes).map((n) => n.name));
  if (bare && !taken.has(base)) return base;
  for (let i = 1; ; i++) {
    const name = `${base}${i}`;
    if (!taken.has(name)) return name;
  }
}

let counter = 0;
/** Short random ids; stable once assigned. */
export function newId(prefix: string): string {
  counter = (counter + 1) % 1296;
  return `${prefix}_${Date.now().toString(36).slice(-4)}${Math.random().toString(36).slice(2, 6)}${counter.toString(36)}`;
}
