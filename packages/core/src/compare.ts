// Structural equivalence used by the round-trip gate (PRD section 11.5).
import { analyzeModule, analyzeProject, withDefaults, type Analysis, type ProjectAnalysis } from './analyze';
import { canonicalValue } from './codegen';
import { interfaceOrder } from './graph';
import { exprOf, literalOf, type Literal, type LoomProject, type ModuleDef, type ModuleId, type NodeId } from './ir';
import { parseShape, shapeText } from './shape';
import { getSpec } from './specs';

/**
 * A sorted list of lines describing the project, independent of ids and layout.
 * Modules are named by class, layers, module instances and inputs by their
 * variable names, which survive a round trip. Ops and code blocks have no name
 * in the code, so they are labelled by type and topological position.
 */
export function structuralSignature(project: LoomProject, analysis: ProjectAnalysis = analyzeProject(project)): string[] {
  const lines: string[] = [];
  if (project.preamble?.trim()) lines.push(`preamble ${JSON.stringify(project.preamble.trim())}`);
  if (project.epilogue?.trim()) lines.push(`epilogue ${JSON.stringify(project.epilogue.trim())}`);
  lines.push(`entry ${project.modules[project.entry]?.className}`);
  // An inline nn.Sequential has no name in the code; it is known by where it is used.
  const names = new Map<ModuleId, string>();
  for (const mod of Object.values(project.modules)) {
    for (const n of Object.values(mod.nodes)) {
      if (n.kind === 'module' && project.modules[n.type]?.sequential && !names.has(n.type)) names.set(n.type, `${mod.className}.${n.name}`);
    }
  }
  const nameOf = (id: ModuleId) => names.get(id) ?? project.modules[id]?.className ?? id;
  for (const mod of Object.values(project.modules)) {
    const a = analysis.byModule[mod.id]?.[0] ?? analyzeModule(project, mod.id);
    lines.push(...moduleSignature(project, mod, a, mod.id === project.entry, nameOf));
  }
  return lines.sort();
}

function moduleSignature(project: LoomProject, mod: ModuleDef, analysis: Analysis, isEntry: boolean, nameOf: (id: ModuleId) => string): string[] {
  const prefix = `[${nameOf(mod.id)}]`;
  const lines = [
    `${prefix} module ${stable({
      hyperparams: mod.hyperparams,
      constants: mod.constants ?? [],
      sequential: !!mod.sequential,
      opaque: mod.opaque ?? null,
      initCode: mod.initCode?.trim() ?? '',
      members: mod.members?.trim() ?? '',
    })}`,
  ];
  if (mod.opaque) {
    interfaceOrder(mod, 'input').forEach((n, i) => lines.push(`${prefix} arg[${i}] ${n.name}`));
    lines.push(`${prefix} returns ${mod.interface.outputs.length}`);
    return lines;
  }

  const label = new Map<NodeId, string>();
  // Outputs are positional (the return tuple), so their label is their index in it.
  interfaceOrder(mod, 'output').forEach((n, i) => label.set(n.id, `return[${i}]`));
  const perType = new Map<string, number>();
  const order = [...analysis.order, ...Object.keys(mod.nodes).filter((id) => !analysis.order.includes(id))];
  for (const id of order) {
    const node = mod.nodes[id]!;
    if (node.kind === 'layer' || node.kind === 'input' || node.kind === 'module') {
      label.set(id, node.name);
    } else if (node.kind === 'op' || node.kind === 'code') {
      const k = perType.get(node.type) ?? 0;
      perType.set(node.type, k + 1);
      label.set(id, `${node.kind === 'code' ? 'code' : node.type}#${k}`);
    }
  }

  /** Module ports are node ids of the referenced definition; compare them by name. */
  const portName = (nodeId: NodeId, port: string): string => {
    const node = mod.nodes[nodeId];
    if (node?.kind !== 'module') return port;
    const def = project.modules[node.type];
    const out = def?.interface.outputs.indexOf(port) ?? -1;
    return out >= 0 ? `return[${out}]` : (def?.nodes[port]?.name ?? port);
  };

  for (const id of order) {
    const node = mod.nodes[id]!;
    let detail: unknown;
    if (node.kind === 'module') {
      const def = project.modules[node.type];
      // Values equal to the default are not written in the code, so they are not part of the structure.
      const params = Object.fromEntries(
        Object.entries(node.params).filter(([k, v]) => JSON.stringify(def?.hyperparams.find((h) => h.name === k)?.default) !== JSON.stringify(v)),
      );
      // Declared shapes are editor annotations, not code, so like layout they are not compared.
      detail = { module: nameOf(node.type), params, repeat: node.repeat ?? null };
    } else if (node.kind === 'code') {
      detail = { code: node.code };
    } else {
      const spec = getSpec(node.type);
      const resolved = analysis.nodes[id]?.resolved;
      const params: Record<string, Literal | string | undefined> = {};
      if (spec) {
        const full = withDefaults(spec, node.params);
        for (const p of spec.params) {
          // Submodule inputs take their shape from the call site; only the entry declares one.
          if (node.kind === 'input' && !isEntry && p.name === 'shape') continue;
          const text = exprOf(full[p.name]);
          if (text !== undefined) {
            params[p.name] = `=${text}`;
            continue;
          }
          let value = canonicalValue(resolved?.[p.name] ?? literalOf(full[p.name]), p);
          if (p.type === 'shape' && typeof value === 'string') {
            const parsed = parseShape(value);
            if (parsed.ok) value = shapeText(parsed.shape);
          }
          params[p.name] = value;
        }
      }
      detail = { type: node.type, params, repeat: node.repeat ?? null };
    }
    lines.push(`${prefix} node ${label.get(id)} ${stable(detail)}`);
  }
  for (const e of mod.edges) {
    lines.push(
      `${prefix} edge ${label.get(e.source.node)}.${portName(e.source.node, e.source.port)} -> ${label.get(e.target.node)}.${portName(e.target.node, e.target.port)}`,
    );
  }
  return lines;
}

/** JSON with sorted object keys, so key order does not count as a difference. */
function stable(value: unknown): string {
  return JSON.stringify(value, (_key, v: unknown) =>
    v && typeof v === 'object' && !Array.isArray(v) ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => (a < b ? -1 : 1))) : v,
  );
}

export function diffSignatures(a: string[], b: string[]): { onlyInA: string[]; onlyInB: string[] } {
  const setA = new Set(a);
  const setB = new Set(b);
  return { onlyInA: a.filter((l) => !setB.has(l)), onlyInB: b.filter((l) => !setA.has(l)) };
}
