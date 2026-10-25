// Checks untrusted JSON (a .loom file, or the Python backend's import result)
// before it enters the editor, migrating older schema versions on the way.
import { isValidIdentifier } from './graph';
import { msg } from './i18n';
import { CODE_TYPE, SCHEMA_VERSION, type LoomProject } from './ir';
import { getSpec } from './specs';

type Json = Record<string, unknown>;
const isObject = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v);
const isStringList = (v: unknown): v is string[] => Array.isArray(v) && v.every((s) => typeof s === 'string');
const isParamValue = (v: unknown): boolean =>
  isObject(v) && ('literal' in v || v.auto === true || typeof v.expr === 'string');

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

/** 0.2 had a single flat module without hyperparameters. */
function migrate02(project: Json): void {
  for (const mod of Object.values(project.modules as Json)) {
    if (isObject(mod) && !Array.isArray(mod.hyperparams)) mod.hyperparams = [];
  }
  project.schemaVersion = '0.3';
}

const versionOf = (v: unknown): [number, number] | undefined =>
  typeof v === 'string' && /^\d+\.\d+$/.test(v) ? (v.split('.').map(Number) as [number, number]) : undefined;

/**
 * The project's schema version when it is newer than this build reads ("0.4"
 * against "0.3"). Such a file opens read-only and asks for an upgrade (PRD,
 * non-functional requirements: compatibility).
 */
export function newerSchema(project: { schemaVersion?: unknown }): string | undefined {
  const file = versionOf(project.schemaVersion);
  const ours = versionOf(SCHEMA_VERSION)!;
  return file && (file[0] > ours[0] || (file[0] === ours[0] && file[1] > ours[1])) ? (project.schemaVersion as string) : undefined;
}

/**
 * A node type this build does not know, in a newer file: shown as a code block
 * with the ports its edges use, so the rest of the graph can still be read.
 */
function placeholder(node: Json, edges: unknown[]): void {
  // Port names become variables in the code block: `in` is written `in_`.
  const ports = (side: 'source' | 'target') => {
    const names = new Set<string>();
    for (const end of edges.filter(isObject).map((e) => e[side])) {
      if (!isObject(end) || end.node !== node.id) continue;
      const name = String(end.port);
      end.port = isValidIdentifier(name) ? name : `${/^\d/.test(name) ? 'p' : ''}${name.replace(/\W/g, '_')}_`;
      names.add(end.port as string);
    }
    return [...names];
  };
  node.code = { source: `# ${String(node.type)}: ${msg('需要更新版本的 NeuroLoom', 'needs a newer NeuroLoom')}`, inputs: ports('target'), outputs: ports('source') };
  node.type = CODE_TYPE;
  node.params = {};
}

export function parseProject(json: unknown): LoomProject {
  const invalid = (zh: string, en: string) => new Error(msg(zh, en));
  if (!isObject(json)) throw invalid('项目数据不是一个对象', 'the project data is not an object');
  if (typeof json.entry !== 'string' || !isObject(json.modules)) throw invalid('项目缺少 entry 或 modules', 'the project has no entry or modules');
  if (json.schemaVersion === '0.1') migrate01(json);
  if (json.schemaVersion === '0.2') migrate02(json);
  const newer = newerSchema(json);
  if (json.schemaVersion !== SCHEMA_VERSION && !newer) {
    throw invalid(`不支持的 schemaVersion：${String(json.schemaVersion)}`, `unsupported schemaVersion ${String(json.schemaVersion)}`);
  }
  if (!newer) return check(json);
  // A newer file is read as this version's; what this build cannot show is reported with the upgrade.
  try {
    return check(json, newer);
  } catch (e) {
    throw invalid(
      `这个文件来自更新版本的 NeuroLoom（文件格式 ${newer}，当前版本支持 ${SCHEMA_VERSION}），当前版本无法显示：${(e as Error).message}。请升级 NeuroLoom 后打开。`,
      `this file comes from a newer NeuroLoom (file format ${newer}; this version reads ${SCHEMA_VERSION}) and cannot be shown here: ${(e as Error).message}. Upgrade NeuroLoom to open it.`,
    );
  }
}

/** Checks a project of this schema version, or of a newer one shown read-only. */
function check(json: Json, newer?: string): LoomProject {
  const invalid = (zh: string, en: string) => new Error(msg(zh, en));
  const modules = json.modules as Json;
  if (!isObject(modules[json.entry as string])) throw invalid(`入口模块 ${String(json.entry)} 不存在`, `entry module ${String(json.entry)} does not exist`);
  if (json.preamble !== undefined && typeof json.preamble !== 'string') throw invalid('preamble 必须是文本', 'preamble must be text');
  if (json.epilogue !== undefined && typeof json.epilogue !== 'string') throw invalid('epilogue 必须是文本', 'epilogue must be text');
  if (json.exportOptions !== undefined) {
    const o = json.exportOptions;
    const ok = isObject(o) && Object.entries(o).every(([k, v]) => (k === 'nesting' ? Number.isInteger(v) && (v as number) >= 0 : typeof v === 'boolean'));
    if (!ok) throw invalid('exportOptions 结构不完整', 'incomplete exportOptions');
  }

  for (const [moduleId, mod] of Object.entries(modules)) {
    if (!isObject(mod) || !isObject(mod.nodes) || !Array.isArray(mod.edges) || typeof mod.className !== 'string') {
      throw invalid('模块结构不完整', 'incomplete module');
    }
    mod.id = moduleId;
    if (!Array.isArray(mod.hyperparams)) mod.hyperparams = [];
    for (const hp of mod.hyperparams as unknown[]) {
      if (
        !isObject(hp) ||
        typeof hp.name !== 'string' ||
        (hp.default !== undefined && !isParamValue(hp.default)) ||
        (hp.keywordOnly !== undefined && typeof hp.keywordOnly !== 'boolean')
      ) {
        throw invalid(`模块 ${mod.className} 的超参数结构不完整`, `incomplete hyperparameter in ${mod.className}`);
      }
    }
    for (const c of (mod.constants ?? []) as unknown[]) {
      if (!isObject(c) || typeof c.name !== 'string' || typeof c.expr !== 'string') throw invalid('常量结构不完整', 'incomplete constant');
    }
    for (const key of ['initCode', 'members', 'kwargs'] as const) {
      if (mod[key] !== undefined && typeof mod[key] !== 'string') throw invalid(`${key} 必须是文本`, `${key} must be text`);
    }
    if (mod.opaque !== undefined && (!isObject(mod.opaque) || (mod.opaque.source !== undefined && typeof mod.opaque.source !== 'string'))) {
      throw invalid('opaque 结构不完整', 'incomplete opaque module');
    }

    for (const node of Object.values(mod.nodes)) {
      if (!isObject(node) || typeof node.id !== 'string' || typeof node.name !== 'string' || !isObject(node.params)) {
        throw invalid('节点结构不完整', 'incomplete node');
      }
      if (newer && node.type !== CODE_TYPE && node.kind !== 'module' && !isObject(modules[node.type as string]) && !getSpec(node.type as string)) {
        placeholder(node, mod.edges);
      }
      if (!Object.values(node.params).every(isParamValue)) throw invalid(`节点 ${node.name} 的参数结构不完整`, `incomplete parameters on ${node.name}`);
      const type = node.type as string;
      if (type === CODE_TYPE) {
        const code = node.code;
        if (!isObject(code) || typeof code.source !== 'string' || !isStringList(code.inputs) || !isStringList(code.outputs)) {
          throw invalid(`代码块 ${node.name} 结构不完整`, `incomplete code block ${node.name}`);
        }
        node.kind = 'code';
      } else if (node.kind === 'module' || (isObject(modules[type]) && !getSpec(type))) {
        if (!isObject(modules[type])) throw invalid(`节点 ${node.name} 引用的模块 ${type} 不存在`, `module ${type} used by ${node.name} does not exist`);
        node.kind = 'module';
      } else {
        const spec = getSpec(type);
        if (!spec) throw invalid(`未知的节点类型 ${String(type)}`, `unknown node type ${String(type)}`);
        node.kind = spec.kind;
      }
      if (node.repeat !== undefined && (!isObject(node.repeat) || !isParamValue(node.repeat.count))) {
        throw invalid(`节点 ${node.name} 的重复设置不完整`, `incomplete repeat on ${node.name}`);
      }
    }
    for (const edge of mod.edges) {
      if (!isObject(edge) || !isObject(edge.source) || !isObject(edge.target)) throw invalid('边结构不完整', 'incomplete edge');
      if (!((edge.source.node as string) in mod.nodes) || !((edge.target.node as string) in mod.nodes)) {
        throw invalid(`边 ${String(edge.id)} 指向不存在的节点`, `edge ${String(edge.id)} points to a missing node`);
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
    const inputs = fix('input', declared.inputs);
    // Defaults of optional forward() arguments: kept for inputs that still exist.
    const defaults = isObject(declared.defaults)
      ? Object.fromEntries(Object.entries(declared.defaults).filter(([id, v]) => inputs.includes(id) && typeof v === 'string'))
      : {};
    mod.interface = {
      inputs,
      outputs: fix('output', declared.outputs),
      ...(Object.keys(defaults).length > 0 ? { defaults } : {}),
    };
  }
  return json as unknown as LoomProject;
}
