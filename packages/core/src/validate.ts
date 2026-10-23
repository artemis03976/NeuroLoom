// Checks untrusted JSON (a .loom file, or the Python backend's import result)
// before it enters the editor.
import { SCHEMA_VERSION, type LoomProject } from './ir';
import { getSpec } from './specs';

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

export function parseProject(json: unknown): LoomProject {
  if (!isObject(json)) throw new Error('项目数据不是一个对象');
  if (json.schemaVersion !== SCHEMA_VERSION) throw new Error(`不支持的 schemaVersion：${String(json.schemaVersion)}`);
  if (typeof json.entry !== 'string' || !isObject(json.modules)) throw new Error('项目缺少 entry 或 modules');
  const entry = json.modules[json.entry];
  if (!isObject(entry)) throw new Error(`入口模块 ${json.entry} 不存在`);

  for (const mod of Object.values(json.modules)) {
    if (!isObject(mod) || !isObject(mod.nodes) || !Array.isArray(mod.edges) || typeof mod.className !== 'string') {
      throw new Error('模块结构不完整');
    }
    for (const node of Object.values(mod.nodes)) {
      if (!isObject(node) || typeof node.id !== 'string' || typeof node.name !== 'string' || !isObject(node.params)) {
        throw new Error('节点结构不完整');
      }
      if (!getSpec(node.type as string)) throw new Error(`未知的节点类型 ${String(node.type)}`);
    }
    for (const edge of mod.edges) {
      if (!isObject(edge) || !isObject(edge.source) || !isObject(edge.target)) throw new Error('边结构不完整');
      if (!(edge.source.node as string in mod.nodes) || !(edge.target.node as string in mod.nodes)) {
        throw new Error(`边 ${String(edge.id)} 指向不存在的节点`);
      }
    }
    if (!isObject(mod.layout)) mod.layout = {};
  }
  return json as unknown as LoomProject;
}
