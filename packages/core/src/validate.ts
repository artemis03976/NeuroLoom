// Checks untrusted JSON (a .loom file, or the Python backend's import result)
// before it enters the editor, migrating older schema versions on the way.
import { SCHEMA_VERSION, type LoomProject } from './ir';
import { getSpec } from './specs';

type Json = Record<string, unknown>;
const isObject = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v);

/** 0.1 had no explicit interface; forward() order came from node insertion order. */
function migrate01(project: Json): void {
  for (const mod of Object.values(project.modules as Json)) {
    if (!isObject(mod) || !isObject(mod.nodes)) continue;
    const nodes = Object.values(mod.nodes).filter(isObject);
    mod.interface = {
      inputs: nodes.filter((n) => n.kind === 'input').map((n) => n.id),
      outputs: nodes.filter((n) => n.kind === 'output').map((n) => n.id),
    };
  }
  project.schemaVersion = '0.2';
}

export function parseProject(json: unknown): LoomProject {
  if (!isObject(json)) throw new Error('项目数据不是一个对象');
  if (typeof json.entry !== 'string' || !isObject(json.modules)) throw new Error('项目缺少 entry 或 modules');
  if (json.schemaVersion === '0.1') migrate01(json);
  if (json.schemaVersion !== SCHEMA_VERSION) {
    throw new Error(`不支持的 schemaVersion：${String(json.schemaVersion)}（这个文件可能来自更新版本的 NeuroLoom）`);
  }
  if (!isObject(json.modules[json.entry])) throw new Error(`入口模块 ${json.entry} 不存在`);

  for (const mod of Object.values(json.modules)) {
    if (!isObject(mod) || !isObject(mod.nodes) || !Array.isArray(mod.edges) || typeof mod.className !== 'string') {
      throw new Error('模块结构不完整');
    }
    for (const node of Object.values(mod.nodes)) {
      if (!isObject(node) || typeof node.id !== 'string' || typeof node.name !== 'string' || !isObject(node.params)) {
        throw new Error('节点结构不完整');
      }
      const spec = getSpec(node.type as string);
      if (!spec) throw new Error(`未知的节点类型 ${String(node.type)}`);
      node.kind = spec.kind;
    }
    for (const edge of mod.edges) {
      if (!isObject(edge) || !isObject(edge.source) || !isObject(edge.target)) throw new Error('边结构不完整');
      if (!((edge.source.node as string) in mod.nodes) || !((edge.target.node as string) in mod.nodes)) {
        throw new Error(`边 ${String(edge.id)} 指向不存在的节点`);
      }
    }
    if (!isObject(mod.layout)) mod.layout = {};

    // Keep the interface consistent with the nodes: drop stale ids, append unlisted inputs/outputs.
    const nodes = mod.nodes as Record<string, Json>;
    const declared = isObject(mod.interface) ? mod.interface : {};
    const fix = (kind: 'input' | 'output', listed: unknown) => {
      const ids = (Array.isArray(listed) ? listed : []).filter(
        (id): id is string => typeof id === 'string' && nodes[id]?.kind === kind,
      );
      const unique = [...new Set(ids)];
      for (const n of Object.values(nodes)) if (n.kind === kind && !unique.includes(n.id as string)) unique.push(n.id as string);
      return unique;
    };
    mod.interface = { inputs: fix('input', declared.inputs), outputs: fix('output', declared.outputs) };
  }
  return json as unknown as LoomProject;
}
