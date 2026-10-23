// Structural equivalence used by the round-trip gate (PRD section 11.5).
import { analyzeModule, withDefaults, type Analysis } from './analyze';
import { interfaceOrder } from './graph';
import { literalOf, type Literal, type ModuleDef, type NodeId } from './ir';
import { parseShape, shapeText } from './shape';
import { getSpec } from './specs';

/**
 * A sorted list of lines describing the module. Layers and inputs are named by
 * their variable names, which survive a round trip. Ops and outputs have no
 * name in the code, so they are labelled by type and topological position.
 */
export function structuralSignature(mod: ModuleDef, analysis: Analysis = analyzeModule(mod)): string[] {
  const label = new Map<NodeId, string>();
  // Outputs are positional (the return tuple), so their label is their index in it.
  interfaceOrder(mod, 'output').forEach((n, i) => label.set(n.id, `return[${i}]`));
  const perType = new Map<string, number>();
  for (const id of analysis.order) {
    const node = mod.nodes[id]!;
    if (node.kind === 'layer' || node.kind === 'input') {
      label.set(id, node.name);
    } else if (node.kind === 'op') {
      const k = perType.get(node.type) ?? 0;
      perType.set(node.type, k + 1);
      label.set(id, `${node.type}#${k}`);
    }
  }

  const lines = [`module ${mod.className}`];
  for (const id of analysis.order) {
    const node = mod.nodes[id]!;
    const spec = getSpec(node.type);
    const resolved = analysis.nodes[id]?.resolved;
    const params: Record<string, Literal | undefined> = {};
    if (spec) {
      const full = withDefaults(spec, node.params);
      for (const p of spec.params) {
        let value = resolved?.[p.name] ?? literalOf(full[p.name]);
        if (p.type === 'shape' && typeof value === 'string') {
          const parsed = parseShape(value);
          if (parsed.ok) value = shapeText(parsed.shape);
        }
        params[p.name] = value;
      }
    }
    lines.push(`node ${label.get(id)} ${node.type} ${JSON.stringify(params)}`);
  }
  for (const e of mod.edges) {
    lines.push(`edge ${label.get(e.source.node)}.${e.source.port} -> ${label.get(e.target.node)}.${e.target.port}`);
  }
  return lines.sort();
}

export function diffSignatures(a: string[], b: string[]): { onlyInA: string[]; onlyInB: string[] } {
  const setA = new Set(a);
  const setB = new Set(b);
  return { onlyInA: a.filter((l) => !setB.has(l)), onlyInB: b.filter((l) => !setA.has(l)) };
}
