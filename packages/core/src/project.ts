// Project-level structure: node ports (which depend on the referenced module
// for module nodes), module dependencies and instance paths (PRD section 9).
import type { LoomProject, ModuleDef, ModuleId, NodeId, NodeInst } from './ir';
import { getSpec, outputsOf, withDefaults, type Params } from './specs';

export interface PortInfo {
  /** Port id used by edges. Module nodes use the definition's input / output node ids. */
  name: string;
  /** Shown on the canvas. */
  label: string;
  optional?: boolean;
  /** Accepts any number of edges, stored as name0, name1, … (torch.cat). */
  variadic?: boolean;
  /** Index path into a tuple result. */
  path?: number[];
}

export interface NodePorts {
  inputs: PortInfo[];
  outputs: PortInfo[];
}

/**
 * Ports of a node, or undefined when its type is unknown. `params` overrides the
 * node's own parameters (bound values, for outputs counted by a parameter).
 */
export function portsOf(project: LoomProject, node: NodeInst, params?: Params): NodePorts | undefined {
  if (node.kind === 'module') {
    const def = project.modules[node.type];
    if (!def) return undefined;
    const port = (id: NodeId) => ({ name: id, label: def.nodes[id]?.name ?? id });
    // A forward() argument with a default may stay unconnected.
    const input = (id: NodeId) => ({ ...port(id), ...(def.interface.defaults?.[id] !== undefined ? { optional: true } : {}) });
    return { inputs: def.interface.inputs.map(input), outputs: def.interface.outputs.map(port) };
  }
  if (node.kind === 'code') {
    return {
      inputs: (node.code?.inputs ?? []).map((name) => ({ name, label: name })),
      outputs: (node.code?.outputs ?? []).map((name) => ({ name, label: name })),
    };
  }
  const spec = getSpec(node.type);
  if (!spec) return undefined;
  return {
    inputs: spec.inputs.map((p) => ({ ...p, label: p.name })),
    outputs: outputsOf(spec, withDefaults(spec, params ?? node.params)).map((o) => ({ ...o, label: o.name })),
  };
}

/** Module definitions a module instantiates directly. */
export function moduleDeps(project: LoomProject, mod: ModuleDef): ModuleId[] {
  const deps = new Set<ModuleId>();
  for (const n of Object.values(mod.nodes)) if (n.kind === 'module' && project.modules[n.type]) deps.add(n.type);
  return [...deps];
}

/** True when `from` contains `target`, directly or through other modules (or is it). */
export function dependsOn(project: LoomProject, from: ModuleId, target: ModuleId): boolean {
  const seen = new Set<ModuleId>();
  const stack = [from];
  while (stack.length > 0) {
    const id = stack.pop()!;
    if (id === target) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    const mod = project.modules[id];
    if (mod) stack.push(...moduleDeps(project, mod));
  }
  return false;
}

/**
 * All modules with dependencies first and the entry last (PRD 10.2), so classes
 * are defined before they are used. Ties keep the project's module order.
 */
export function dependencyOrder(project: LoomProject): ModuleId[] {
  const order: ModuleId[] = [];
  const state = new Map<ModuleId, 'visiting' | 'done'>();
  const visit = (id: ModuleId) => {
    if (state.get(id)) return;
    state.set(id, 'visiting');
    const mod = project.modules[id];
    if (mod) for (const dep of moduleDeps(project, mod)) visit(dep);
    state.set(id, 'done');
    order.push(id);
  };
  for (const id of Object.keys(project.modules)) if (id !== project.entry) visit(id);
  visit(project.entry);
  // The entry goes last even when nothing depends on the modules before it.
  return [...order.filter((id) => id !== project.entry), project.entry].filter((id) => project.modules[id]);
}

export interface InstanceRef {
  moduleId: ModuleId;
  nodeId: NodeId;
}

export function instancesOf(project: LoomProject, defId: ModuleId): InstanceRef[] {
  const refs: InstanceRef[] = [];
  for (const mod of Object.values(project.modules)) {
    for (const n of Object.values(mod.nodes)) if (n.kind === 'module' && n.type === defId) refs.push({ moduleId: mod.id, nodeId: n.id });
  }
  return refs;
}

/**
 * Follows a drill-down path of module node ids from the entry. Returns the
 * module shown at the end and the part of the path that is still valid.
 */
export function resolvePath(project: LoomProject, path: NodeId[]): { moduleId: ModuleId; path: NodeId[] } {
  let moduleId = project.entry;
  const valid: NodeId[] = [];
  for (const id of path) {
    const node = project.modules[moduleId]?.nodes[id];
    if (node?.kind !== 'module' || !project.modules[node.type]) break;
    moduleId = node.type;
    valid.push(id);
  }
  return { moduleId, path: valid };
}

/** The shortest drill-down path from the entry to an instance of a module, if it is used. */
export function findPath(project: LoomProject, target: ModuleId): NodeId[] | undefined {
  if (target === project.entry) return [];
  const queue: { moduleId: ModuleId; path: NodeId[] }[] = [{ moduleId: project.entry, path: [] }];
  const seen = new Set<ModuleId>([project.entry]);
  while (queue.length > 0) {
    const { moduleId, path } = queue.shift()!;
    for (const n of Object.values(project.modules[moduleId]?.nodes ?? {})) {
      if (n.kind !== 'module') continue;
      if (n.type === target) return [...path, n.id];
      if (!seen.has(n.type) && project.modules[n.type]) {
        seen.add(n.type);
        queue.push({ moduleId: n.type, path: [...path, n.id] });
      }
    }
  }
  return undefined;
}

/** A class name not used by any module, based on `base` ("Block" -> "Block2"). */
export function uniqueClassName(project: LoomProject, base: string): string {
  const taken = new Set(Object.values(project.modules).map((m) => m.className));
  if (!taken.has(base)) return base;
  const stem = base.replace(/\d+$/, '');
  for (let i = 2; ; i++) if (!taken.has(`${stem}${i}`)) return `${stem}${i}`;
}

/** A module id not used yet, derived from a class name. */
export function uniqueModuleId(project: LoomProject, className: string): ModuleId {
  const base = className.replace(/[^A-Za-z0-9_]/g, '_') || 'Module';
  if (!project.modules[base]) return base;
  for (let i = 2; ; i++) if (!project.modules[`${base}_${i}`]) return `${base}_${i}`;
}

/** "ResBlock" -> "res_block": the base of instance names. */
export function snakeCase(className: string): string {
  const s = className
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/([A-Z])([A-Z][a-z])/g, '$1_$2')
    .replace(/[^A-Za-z0-9_]/g, '_')
    .toLowerCase();
  return /^[a-z_]/.test(s) ? s : `m_${s}`;
}
