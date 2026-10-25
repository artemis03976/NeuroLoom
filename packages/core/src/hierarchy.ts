// Editing operations that span modules (PRD section 9): instantiate, pack a
// selection into a new module, unpack an instance, make an independent copy,
// and manage hyperparameters. Every function returns a new project.
import type { Analysis } from './analyze';
import { evalExpr, freeNames, literalSource, renameInExpr, substituteExpr } from './expr';
import { isValidIdentifier, newId, portKey, topoOrder, uniqueName } from './graph';
import { msg } from './i18n';
import {
  CODE_TYPE,
  exprOf,
  type Edge,
  type HyperParam,
  type LoomProject,
  type ModuleDef,
  type ModuleId,
  type NodeId,
  type NodeInst,
  type ParamValue,
} from './ir';
import { dependsOn, instancesOf, portsOf, snakeCase, uniqueClassName, uniqueModuleId } from './project';
import { shapeText } from './shape';

export class EditError extends Error {}

export function withModule(project: LoomProject, moduleId: ModuleId, update: (mod: ModuleDef) => ModuleDef): LoomProject {
  const mod = project.modules[moduleId];
  if (!mod) return project;
  const next = update(mod);
  return next === mod ? project : { ...project, modules: { ...project.modules, [moduleId]: next } };
}

export function isValidClassName(name: string): boolean {
  return isValidIdentifier(name) && /^[A-Z]/.test(name);
}

/** Adds an instance of `defId` to `moduleId`. Refuses one that would make a module contain itself. */
export function addModuleNode(
  project: LoomProject,
  moduleId: ModuleId,
  defId: ModuleId,
  position: { x: number; y: number },
): { project: LoomProject; nodeId: NodeId } {
  const def = project.modules[defId];
  if (!def) throw new EditError(msg(`模块 ${defId} 不存在`, `module ${defId} does not exist`));
  if (dependsOn(project, defId, moduleId)) {
    throw new EditError(msg(`不能在 ${def.className} 自身（或它包含的模块）中使用它`, `${def.className} cannot be used inside itself or a module it contains`));
  }
  const id = newId('n');
  const project2 = withModule(project, moduleId, (mod) => ({
    ...mod,
    nodes: { ...mod.nodes, [id]: { id, kind: 'module', type: defId, name: uniqueName(mod, snakeCase(def.className)), params: {} } },
    layout: { ...mod.layout, [id]: position },
  }));
  return { project: project2, nodeId: id };
}

/** Adds an empty code block node. */
export function addCodeNode(mod: ModuleDef, position: { x: number; y: number }): { mod: ModuleDef; nodeId: NodeId } {
  const id = newId('n');
  const node: NodeInst = { id, kind: 'code', type: CODE_TYPE, name: uniqueName(mod, 'code', 'fx'), params: {}, code: { source: 'y = x', inputs: ['x'], outputs: ['y'] } };
  return { nodeId: id, mod: { ...mod, nodes: { ...mod.nodes, [id]: node }, layout: { ...mod.layout, [id]: position } } };
}

/** A new module with one input passed straight to one output. */
export function createModule(project: LoomProject, className: string): { project: LoomProject; moduleId: ModuleId } {
  const id = uniqueModuleId(project, className);
  const input: NodeInst = { id: 'in', kind: 'input', type: 'neuroloom.Input', name: 'x', params: { shape: { literal: '' }, dtype: { literal: 'float32' } } };
  const output: NodeInst = { id: 'out', kind: 'output', type: 'neuroloom.Output', name: 'output', params: {} };
  const mod: ModuleDef = {
    id,
    className: uniqueClassName(project, className),
    hyperparams: [],
    nodes: { in: input, out: output },
    edges: [{ id: 'e', source: { node: 'in', port: 'out' }, target: { node: 'out', port: 'in' } }],
    layout: { in: { x: 0, y: 0 }, out: { x: 360, y: 0 } },
    interface: { inputs: ['in'], outputs: ['out'] },
  };
  return { project: { ...project, modules: { ...project.modules, [id]: mod } }, moduleId: id };
}

export function deleteModule(project: LoomProject, defId: ModuleId): LoomProject {
  if (defId === project.entry) throw new EditError(msg('不能删除入口模型', 'the entry model cannot be deleted'));
  if (instancesOf(project, defId).length > 0) throw new EditError(msg('模块仍在使用中，请先删除它的实例', 'the module is still in use; delete its instances first'));
  const modules = { ...project.modules };
  delete modules[defId];
  return { ...project, modules };
}

export function renameModule(project: LoomProject, defId: ModuleId, className: string): LoomProject {
  return withModule(project, defId, (m) => ({ ...m, className }));
}

/** Changes an expression's names everywhere it can refer to `defId`'s scope. */
function renameNamesInModule(mod: ModuleDef, rename: (name: string) => string | undefined): ModuleDef {
  const value = (v: ParamValue): ParamValue => {
    const text = exprOf(v);
    return text === undefined ? v : { expr: renameInExpr(text, rename) };
  };
  const values = (params: Record<string, ParamValue>) => Object.fromEntries(Object.entries(params).map(([k, v]) => [k, value(v)]));
  const nodes = Object.fromEntries(
    Object.entries(mod.nodes).map(([id, n]) => [
      id,
      {
        ...n,
        params: values(n.params),
        ...(n.repeat ? { repeat: { count: value(n.repeat.count), ...(n.repeat.first ? { first: values(n.repeat.first) } : {}) } } : {}),
      },
    ]),
  );
  const constants = mod.constants?.map((c) => ({ ...c, expr: renameInExpr(c.expr, rename) }));
  return { ...mod, nodes, ...(constants ? { constants } : {}) };
}

/**
 * Replaces a module's hyperparameter list. A renamed hyperparameter (same
 * index, `renamed` maps old -> new) is renamed in the module's expressions and
 * in every instance; a removed one is dropped from every instance.
 */
export function setHyperparams(project: LoomProject, defId: ModuleId, hyperparams: HyperParam[], renamed: Record<string, string> = {}): LoomProject {
  const def = project.modules[defId];
  if (!def) return project;
  const keep = new Set(hyperparams.map((h) => h.name));
  let next = withModule(project, defId, (m) => ({ ...renameNamesInModule(m, (n) => renamed[n]), hyperparams }));
  for (const ref of instancesOf(next, defId)) {
    next = withModule(next, ref.moduleId, (m) => {
      const node = m.nodes[ref.nodeId]!;
      const fix = (params: Record<string, ParamValue>) =>
        Object.fromEntries(Object.entries(params).flatMap(([k, v]) => (keep.has(renamed[k] ?? k) ? [[renamed[k] ?? k, v]] : [])));
      const repeat = node.repeat && { ...node.repeat, ...(node.repeat.first ? { first: fix(node.repeat.first) } : {}) };
      return { ...m, nodes: { ...m.nodes, [node.id]: { ...node, params: fix(node.params), ...(repeat ? { repeat } : {}) } } };
    });
  }
  return next;
}

export function updateNode(mod: ModuleDef, nodeId: NodeId, patch: Partial<Pick<NodeInst, 'repeat' | 'code' | 'shapes' | 'params'>>): ModuleDef {
  const node = mod.nodes[nodeId];
  if (!node) return mod;
  const next: NodeInst = { ...node, ...patch };
  for (const key of ['repeat', 'shapes'] as const) if (key in patch && patch[key] === undefined) delete next[key];
  return { ...mod, nodes: { ...mod.nodes, [nodeId]: next } };
}

/** Names a node's expressions use. */
function namesUsed(node: NodeInst): string[] {
  const texts = [...Object.values(node.params), ...(node.repeat ? [node.repeat.count, ...Object.values(node.repeat.first ?? {})] : [])]
    .map(exprOf)
    .filter((t): t is string => t !== undefined);
  return texts.flatMap(freeNames);
}

/**
 * Packs selected nodes into a new module (PRD F4.1). Edges entering the
 * selection become its inputs, edges leaving it its outputs; hyperparameters
 * and constants the nodes refer to become hyperparameters of the new module,
 * passed through by the instance that replaces the selection.
 */
export function packAsModule(
  project: LoomProject,
  moduleId: ModuleId,
  nodeIds: NodeId[],
  className: string,
  analysis?: Analysis,
): { project: LoomProject; moduleId: ModuleId; nodeId: NodeId } {
  const mod = project.modules[moduleId];
  if (!mod) throw new EditError(msg('模块不存在', 'the module does not exist'));
  if (!isValidClassName(className)) throw new EditError(msg('类名必须是以大写字母开头的 Python 标识符', 'the class name must be a Python identifier starting with a capital letter'));
  if (Object.values(project.modules).some((m) => m.className === className)) throw new EditError(msg(`类名 ${className} 已被使用`, `class name ${className} is already used`));
  if (mod.sequential) throw new EditError(msg('不能在 nn.Sequential 内打包', 'cannot pack inside an nn.Sequential'));
  const picked = new Set(nodeIds.filter((id) => mod.nodes[id] && mod.nodes[id]!.kind !== 'input' && mod.nodes[id]!.kind !== 'output'));
  if (picked.size === 0) throw new EditError(msg('请选择至少一个层、算子或模块（输入输出节点不能打包）', 'select at least one layer, op or module (inputs and outputs cannot be packed)'));

  const defId = uniqueModuleId(project, className);
  const rank = new Map(topoOrder(mod).order.map((id, i) => [id, i]));
  const byRank = (a: string, b: string) => {
    const [na, pa] = a.split(':') as [string, string];
    const [nb, pb] = b.split(':') as [string, string];
    return (rank.get(na) ?? 0) - (rank.get(nb) ?? 0) || (pa < pb ? -1 : pa > pb ? 1 : 0);
  };

  // Boundary values: one input per distinct value entering, one output per distinct value leaving.
  const entering = [...new Set(mod.edges.filter((e) => picked.has(e.target.node) && !picked.has(e.source.node)).map((e) => portKey(e.source.node, e.source.port)))].sort(byRank);
  const leaving = [...new Set(mod.edges.filter((e) => picked.has(e.source.node) && !picked.has(e.target.node)).map((e) => portKey(e.source.node, e.source.port)))].sort(byRank);

  const inner: ModuleDef = { id: defId, className, hyperparams: [], nodes: {}, edges: [], layout: {}, interface: { inputs: [], outputs: [] } };
  for (const id of picked) {
    inner.nodes[id] = structuredClone(mod.nodes[id]!);
    inner.layout[id] = mod.layout[id] ?? { x: 0, y: 0 };
  }
  inner.edges = mod.edges.filter((e) => picked.has(e.source.node) && picked.has(e.target.node));
  const xs = [...picked].map((id) => inner.layout[id]!.x);
  const ys = [...picked].map((id) => inner.layout[id]!.y);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];

  const inputFor = new Map<string, NodeId>();
  entering.forEach((key, i) => {
    const [src, port] = key.split(':') as [NodeId, string];
    const source = mod.nodes[src]!;
    const t = analysis?.edgeTypes[mod.edges.find((e) => e.source.node === src && e.source.port === port)?.id ?? ''];
    const id = newId('n');
    const base = source.kind === 'input' ? source.name : port === 'out' || port === 'output' ? source.name : `${source.name}_${port}`;
    inner.nodes[id] = {
      id,
      kind: 'input',
      type: 'neuroloom.Input',
      name: isValidIdentifier(base) ? uniqueName(inner, base, 'bare') : uniqueName(inner, 'x', 'bare'),
      params: { shape: { literal: t ? shapeText(t.shape) : '' }, dtype: { literal: t?.dtype === 'int64' || t?.dtype === 'bool' ? t.dtype : 'float32' } },
    };
    inner.layout[id] = { x: minX - 240, y: minY + i * 120 };
    inner.interface.inputs.push(id);
    inputFor.set(key, id);
  });
  for (const e of mod.edges) {
    const key = portKey(e.source.node, e.source.port);
    if (picked.has(e.target.node) && inputFor.has(key)) {
      inner.edges.push({ id: newId('e'), source: { node: inputFor.get(key)!, port: 'out' }, target: e.target });
    }
  }
  const outputFor = new Map<string, NodeId>();
  leaving.forEach((key, i) => {
    const [src, port] = key.split(':') as [NodeId, string];
    const id = newId('n');
    inner.nodes[id] = { id, kind: 'output', type: 'neuroloom.Output', name: i === 0 ? 'output' : `output_${i}`, params: {} };
    inner.layout[id] = { x: maxX + 260, y: minY + i * 120 };
    inner.interface.outputs.push(id);
    inner.edges.push({ id: newId('e'), source: { node: src, port }, target: { node: id, port: 'in' } });
    outputFor.set(key, id);
  });

  // Names from the parent's scope become hyperparameters, passed straight through.
  const scope = new Map<string, HyperParam | undefined>([
    ...mod.hyperparams.map((h) => [h.name, h] as const),
    ...(mod.constants ?? []).map((c) => [c.name, undefined] as const),
  ]);
  const promoted = [...new Set([...picked].flatMap((id) => namesUsed(mod.nodes[id]!)))].filter((n) => scope.has(n)).sort();
  inner.hyperparams = promoted.map((name) => ({ name, ...(scope.get(name)?.annotation ? { annotation: scope.get(name)!.annotation } : {}) }));

  // The parent: selection out, one instance in.
  const instanceId = newId('n');
  const nodes = { ...mod.nodes };
  const layout = { ...mod.layout };
  for (const id of picked) {
    delete nodes[id];
    delete layout[id];
  }
  const parent: ModuleDef = { ...mod, nodes, layout, edges: mod.edges.filter((e) => !picked.has(e.source.node) && !picked.has(e.target.node)) };
  parent.nodes[instanceId] = {
    id: instanceId,
    kind: 'module',
    type: defId,
    name: uniqueName(parent, snakeCase(className)),
    params: Object.fromEntries(promoted.map((n) => [n, { expr: n }])),
  };
  // Where the selection starts: the first of its nodes the data reaches.
  const first = [...picked].sort((a, b) => (rank.get(a) ?? 0) - (rank.get(b) ?? 0))[0]!;
  parent.layout[instanceId] = { ...(mod.layout[first] ?? { x: minX, y: minY }) };
  for (const key of entering) {
    const [node, port] = key.split(':') as [NodeId, string];
    parent.edges.push({ id: newId('e'), source: { node, port }, target: { node: instanceId, port: inputFor.get(key)! } });
  }
  for (const e of mod.edges) {
    const key = portKey(e.source.node, e.source.port);
    if (outputFor.has(key) && !picked.has(e.target.node)) {
      parent.edges.push({ id: newId('e'), source: { node: instanceId, port: outputFor.get(key)! }, target: e.target });
    }
  }

  return {
    project: { ...project, modules: { ...project.modules, [moduleId]: parent, [defId]: inner } },
    moduleId: defId,
    nodeId: instanceId,
  };
}

/**
 * Inlines a module instance into its parent (PRD F4.1, Ctrl+Shift+G): the
 * definition's nodes are copied in and its hyperparameters replaced by the
 * instance's values.
 */
export function unpackModule(project: LoomProject, moduleId: ModuleId, nodeId: NodeId): { project: LoomProject; nodeIds: NodeId[] } {
  const mod = project.modules[moduleId];
  const node = mod?.nodes[nodeId];
  const def = node?.kind === 'module' ? project.modules[node.type] : undefined;
  if (!mod || !node || !def) throw new EditError(msg('请选择一个模块实例', 'select a module instance'));
  if (node.repeat) throw new EditError(msg('请先取消重复，再解包', 'turn off repeat before unpacking'));
  if (def.opaque || def.initCode?.trim() || def.members?.trim()) {
    throw new EditError(msg(`${def.className} 含有无法拆开的 Python 代码`, `${def.className} contains Python code that cannot be unpacked`));
  }

  // Hyperparameters, then constants, become the expressions they stand for.
  const values: Record<string, string> = {};
  for (const hp of def.hyperparams) {
    const v = node.params[hp.name] ?? hp.default;
    if (!v) continue;
    values[hp.name] = 'expr' in v ? v.expr : 'literal' in v ? literalSource(v.literal) : '';
  }
  for (const c of def.constants ?? []) values[c.name] = substituteExpr(c.expr, values);
  const inline = (v: ParamValue): ParamValue => {
    const text = exprOf(v);
    if (text === undefined || def.sequential) return v;
    const replaced = substituteExpr(text, values);
    if (freeNames(replaced).length === 0) {
      const r = evalExpr(replaced, {});
      if (r.ok) return { literal: r.value };
    }
    return { expr: replaced };
  };
  const inlineAll = (params: Record<string, ParamValue>) => Object.fromEntries(Object.entries(params).map(([k, v]) => [k, inline(v)]));

  const io = new Set([...def.interface.inputs, ...def.interface.outputs]);
  const innerIds = Object.keys(def.nodes).filter((id) => !io.has(id));
  const xs = innerIds.map((id) => def.layout[id]?.x ?? 0);
  const ys = innerIds.map((id) => def.layout[id]?.y ?? 0);
  const origin = { x: (Math.min(...xs) + Math.max(...xs)) / 2 || 0, y: (Math.min(...ys) + Math.max(...ys)) / 2 || 0 };
  const at = mod.layout[nodeId] ?? { x: 0, y: 0 };

  const nodes = { ...mod.nodes };
  delete nodes[nodeId];
  const layout = { ...mod.layout };
  delete layout[nodeId];
  const next: ModuleDef = { ...mod, nodes, layout, edges: mod.edges.filter((e) => e.source.node !== nodeId && e.target.node !== nodeId) };
  const idMap = new Map<NodeId, NodeId>();
  for (const id of innerIds) {
    const inner = def.nodes[id]!;
    const newNodeId = newId('n');
    idMap.set(id, newNodeId);
    const style = inner.kind === 'op' || inner.kind === 'code' ? 'fx' : 'numbered';
    const name = Object.values(next.nodes).some((n) => n.name === inner.name) ? uniqueName(next, inner.name.replace(/_?\d+$/, '') || 'node', style) : inner.name;
    next.nodes[newNodeId] = {
      ...structuredClone(inner),
      id: newNodeId,
      name,
      params: inlineAll(inner.params),
      ...(inner.repeat ? { repeat: { count: inline(inner.repeat.count), ...(inner.repeat.first ? { first: inlineAll(inner.repeat.first) } : {}) } } : {}),
    };
    const p = def.layout[id] ?? { x: 0, y: 0 };
    next.layout[newNodeId] = { x: Math.round(at.x + p.x - origin.x), y: Math.round(at.y + p.y - origin.y) };
  }

  const into = (port: string) => mod.edges.filter((e) => e.target.node === nodeId && e.target.port === port);
  const outOf = (port: string) => mod.edges.filter((e) => e.source.node === nodeId && e.source.port === port);
  const add = (source: Edge['source'], target: Edge['target']) => next.edges.push({ id: newId('e'), source, target });
  for (const e of def.edges) {
    const fromInput = def.interface.inputs.includes(e.source.node);
    const toOutput = def.interface.outputs.includes(e.target.node);
    const sources = fromInput ? into(e.source.node).map((p) => p.source) : [{ node: idMap.get(e.source.node)!, port: e.source.port }];
    const targets = toOutput ? outOf(e.target.node).map((p) => p.target) : [{ node: idMap.get(e.target.node)!, port: e.target.port }];
    for (const s of sources) for (const t of targets) if (s.node && t.node) add(s, t);
  }
  return { project: { ...project, modules: { ...project.modules, [moduleId]: next } }, nodeIds: [...idMap.values()] };
}

/** Points one instance at a copy of its module, so it can change on its own (PRD F4.3). */
export function makeUnique(project: LoomProject, moduleId: ModuleId, nodeId: NodeId): { project: LoomProject; moduleId: ModuleId } {
  const node = project.modules[moduleId]?.nodes[nodeId];
  const def = node?.kind === 'module' ? project.modules[node.type] : undefined;
  if (!node || !def) throw new EditError(msg('请选择一个模块实例', 'select a module instance'));
  const className = uniqueClassName(project, def.className);
  const id = uniqueModuleId(project, className);
  const copy: ModuleDef = { ...structuredClone(def), id, className };
  const next = { ...project, modules: { ...project.modules, [id]: copy } };
  return { project: withModule(next, moduleId, (m) => ({ ...m, nodes: { ...m.nodes, [nodeId]: { ...node, type: id } } })), moduleId: id };
}

/**
 * Drops edges into or out of ports that code blocks and module instances no
 * longer have (a renamed code variable, a deleted module input). Returns the
 * same project when nothing changes.
 */
export function pruneDanglingEdges(project: LoomProject): LoomProject {
  let changed = false;
  const modules = Object.fromEntries(
    Object.entries(project.modules).map(([id, mod]) => {
      const ports = new Map<NodeId, { inputs: Set<string>; outputs: Set<string> } | null>();
      const portSet = (nodeId: NodeId) => {
        if (!ports.has(nodeId)) {
          const node = mod.nodes[nodeId];
          const p = node && (node.kind === 'module' || node.kind === 'code') ? portsOf(project, node) : undefined;
          ports.set(nodeId, p ? { inputs: new Set(p.inputs.map((x) => x.name)), outputs: new Set(p.outputs.map((x) => x.name)) } : null);
        }
        return ports.get(nodeId);
      };
      const edges = mod.edges.filter((e) => {
        const s = portSet(e.source.node);
        const t = portSet(e.target.node);
        return (!s || s.outputs.has(e.source.port)) && (!t || t.inputs.has(e.target.port));
      });
      if (edges.length === mod.edges.length) return [id, mod];
      changed = true;
      return [id, { ...mod, edges }];
    }),
  );
  return changed ? { ...project, modules } : project;
}

/** Makes `defId` the entry model. */
export function setEntry(project: LoomProject, defId: ModuleId): LoomProject {
  return project.modules[defId] ? { ...project, entry: defId } : project;
}
