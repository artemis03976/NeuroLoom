// Copy/paste of node selections (PRD F1.2). The clipboard holds an IR fragment
// plus the module definitions its module nodes use, so it works across projects
// and windows; names are made unique on paste.
import { newId, uniqueName } from './graph';
import { CODE_TYPE, SCHEMA_VERSION, type Edge, type LoomProject, type ModuleDef, type ModuleId, type NodeId, type NodeInst } from './ir';
import { snakeCase, uniqueClassName, uniqueModuleId } from './project';
import { getSpec } from './specs';

export const CLIP_KIND = 'neuroloom/clip';

export interface Clip {
  kind: typeof CLIP_KIND;
  schemaVersion: string;
  nodes: NodeInst[];
  /** Only edges between copied nodes. */
  edges: Edge[];
  layout: Record<NodeId, { x: number; y: number }>;
  /** Definitions used by copied module nodes, directly or indirectly. */
  modules: Record<ModuleId, ModuleDef>;
}

function usedModules(project: LoomProject, nodes: NodeInst[]): Record<ModuleId, ModuleDef> {
  const out: Record<ModuleId, ModuleDef> = {};
  const stack = nodes.filter((n) => n.kind === 'module').map((n) => n.type);
  while (stack.length > 0) {
    const id = stack.pop()!;
    const def = project.modules[id];
    if (!def || out[id]) continue;
    out[id] = structuredClone(def);
    for (const n of Object.values(def.nodes)) if (n.kind === 'module') stack.push(n.type);
  }
  return out;
}

export function copyNodes(project: LoomProject, mod: ModuleDef, ids: NodeId[]): Clip {
  const picked = new Set(ids.filter((id) => mod.nodes[id]));
  const nodes = [...picked].map((id) => structuredClone(mod.nodes[id]!));
  return {
    kind: CLIP_KIND,
    schemaVersion: SCHEMA_VERSION,
    nodes,
    edges: mod.edges.filter((e) => picked.has(e.source.node) && picked.has(e.target.node)).map((e) => structuredClone(e)),
    layout: Object.fromEntries([...picked].map((id) => [id, mod.layout[id] ?? { x: 0, y: 0 }])),
    modules: usedModules(project, nodes),
  };
}

const knownType = (n: NodeInst, modules: Record<ModuleId, ModuleDef>) =>
  n.kind === 'module' ? !!modules[n.type] : n.type === CODE_TYPE ? !!n.code : !!getSpec(n.type);

/** A clip from clipboard text, or null when the text is something else. */
export function parseClip(text: string): Clip | null {
  try {
    const value = JSON.parse(text);
    if (value?.kind !== CLIP_KIND || value.schemaVersion !== SCHEMA_VERSION || !Array.isArray(value.nodes)) return null;
    const modules = (value.modules ?? {}) as Record<ModuleId, ModuleDef>;
    if (!value.nodes.every((n: NodeInst) => typeof n.name === 'string' && knownType(n, modules))) return null;
    return { ...value, modules } as Clip;
  } catch {
    return null;
  }
}

/** Definitions equal in everything but layout are the same module. */
/** Same definition apart from where it is stored: its id, layout, and a numbered class name (Block2 for Block). */
function sameDef(existing: ModuleDef, def: ModuleDef): boolean {
  const content = (m: ModuleDef) => JSON.stringify({ ...m, id: '', className: '', layout: {} });
  const named = existing.className === def.className || (existing.className.startsWith(def.className) && /^\d+$/.test(existing.className.slice(def.className.length)));
  return named && content(existing) === content(def);
}

/** Clip definitions with the ones they use first. */
function usedFirst(defs: Record<ModuleId, ModuleDef>): ModuleDef[] {
  const out: ModuleDef[] = [];
  const seen = new Set<ModuleId>();
  const visit = (id: ModuleId) => {
    const def = defs[id];
    if (!def || seen.has(id)) return;
    seen.add(id);
    for (const n of Object.values(def.nodes)) if (n.kind === 'module') visit(n.type);
    out.push(def);
  };
  Object.keys(defs).forEach(visit);
  return out;
}

/**
 * Pastes into `moduleId` with fresh ids, moved by `offset`; clashing names get
 * the next free one. Copied definitions are reused when the project already has
 * them and added otherwise.
 */
export function pasteClip(project: LoomProject, moduleId: ModuleId, clip: Clip, offset: { x: number; y: number }): { project: LoomProject; nodeIds: NodeId[] } {
  const modules = { ...project.modules };
  const defMap = new Map<ModuleId, ModuleId>();
  for (const def of usedFirst(clip.modules)) {
    const nodes = Object.fromEntries(
      Object.entries(def.nodes).map(([k, n]) => [k, n.kind === 'module' ? { ...n, type: defMap.get(n.type) ?? n.type } : n]),
    );
    const mapped = { ...def, nodes };
    const existing = modules[def.id] && sameDef(modules[def.id]!, mapped) ? modules[def.id] : Object.values(modules).find((m) => sameDef(m, mapped));
    if (existing) {
      defMap.set(def.id, existing.id);
      continue;
    }
    const id = modules[def.id] ? uniqueModuleId({ ...project, modules }, def.className) : def.id;
    const className = uniqueClassName({ ...project, modules }, def.className);
    defMap.set(def.id, id);
    modules[id] = { ...mapped, id, className };
  }

  const mod = modules[moduleId]!;
  const idMap = new Map<NodeId, NodeId>();
  const next: ModuleDef = {
    ...mod,
    nodes: { ...mod.nodes },
    layout: { ...mod.layout },
    edges: [...mod.edges],
    interface: { ...mod.interface, inputs: [...mod.interface.inputs], outputs: [...mod.interface.outputs] },
  };

  for (const node of clip.nodes) {
    const id = newId('n');
    idMap.set(node.id, id);
    const type = node.kind === 'module' ? (defMap.get(node.type) ?? node.type) : node.type;
    const spec = getSpec(type);
    const kind = node.kind === 'module' || node.kind === 'code' ? node.kind : spec!.kind;
    const taken = Object.values(next.nodes).some((n) => n.name === node.name);
    const style = kind === 'op' || kind === 'code' ? 'fx' : kind === 'layer' || kind === 'module' ? 'numbered' : 'bare';
    const base = kind === 'layer' && spec ? spec.defaultName : node.name.replace(/_?\d+$/, '') || (spec?.defaultName ?? 'node');
    const name = taken ? uniqueName(next, base, style) : node.name;
    next.nodes[id] = { ...structuredClone(node), id, name, kind, type };
    const at = clip.layout[node.id] ?? { x: 0, y: 0 };
    next.layout[id] = { x: at.x + offset.x, y: at.y + offset.y };
    if (kind === 'input') next.interface.inputs.push(id);
    if (kind === 'output') next.interface.outputs.push(id);
  }
  for (const e of clip.edges) {
    const source = idMap.get(e.source.node);
    const target = idMap.get(e.target.node);
    if (source && target) next.edges.push({ id: newId('e'), source: { ...e.source, node: source }, target: { ...e.target, node: target } });
  }
  modules[moduleId] = next;
  return { project: { ...project, modules }, nodeIds: [...idMap.values()] };
}

/**
 * Adds an instance of another project's entry module to `moduleId` (a built-in
 * block, PRD 9.6), with the definitions it uses. Definitions the project already
 * has are reused, others are added under free names.
 */
export function insertModuleFrom(
  project: LoomProject,
  moduleId: ModuleId,
  source: LoomProject,
  position: { x: number; y: number },
): { project: LoomProject; nodeId: NodeId } {
  const entry = source.modules[source.entry]!;
  const node: NodeInst = { id: 'block', kind: 'module', type: entry.id, name: snakeCase(entry.className), params: {} };
  const clip: Clip = {
    kind: CLIP_KIND,
    schemaVersion: SCHEMA_VERSION,
    nodes: [node],
    edges: [],
    layout: { [node.id]: position },
    modules: usedModules(source, [node]),
  };
  const result = pasteClip(project, moduleId, clip, { x: 0, y: 0 });
  return { project: result.project, nodeId: result.nodeIds[0]! };
}
