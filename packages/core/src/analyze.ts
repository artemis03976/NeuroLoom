// Shape inference and validation (PRD section 8.4–8.5), per module and per
// instance: a module is analyzed in the context of each place it is used, with
// that instance's hyperparameter values and input shapes (PRD 9.3).
import { evalExpr, freeNames, type Env } from './expr';
import { incomingByPort, isValidIdentifier, portKey, topoOrder } from './graph';
import { msg } from './i18n';
import {
  exprOf,
  isAuto,
  literalOf,
  type Edge,
  type Literal,
  type LoomProject,
  type ModuleDef,
  type ModuleId,
  type NodeId,
  type NodeInst,
  type ParamValue,
} from './ir';
import { portsOf } from './project';
import { parseShape, type TensorType } from './shape';
import { getSpec, variadicIndex, withDefaults, type Inputs, type NodeSpec, type ParamSpec, type Params, type Resolved } from './specs';

export { withDefaults };

export interface Diagnostic {
  severity: 'error' | 'warning';
  message: string;
  nodeId?: NodeId;
}

export interface NodeAnalysis {
  inputs: Inputs;
  /** Per output port; undefined where the type is unknown. Absent when inference failed here or upstream. */
  outputs?: (TensorType | undefined)[];
  /** Layers and ops: parameter values after inference; module nodes: hyperparameter values. */
  resolved?: Resolved;
  /** Repeated layers: the first copy's values, which may differ (auto in_features). `resolved` holds the rest. */
  resolvedFirst?: Resolved;
  paramCount: number;
  /** Module nodes: the analysis of each copy (one, or one per repeat). */
  children?: Analysis[];
}

export interface Analysis {
  moduleId: ModuleId;
  order: NodeId[];
  nodes: Record<NodeId, NodeAnalysis>;
  edgeTypes: Record<string, TensorType>;
  diagnostics: Diagnostic[];
  totalParams: number;
  /** Some parameter counts are unknown (opaque or third-party modules). */
  partial: boolean;
  /** forward() return types, in interface order. */
  outputs: (TensorType | undefined)[];
  /** Hyperparameters and constants in scope. */
  env: Env;
}

export interface ModuleContext {
  env: Env;
  /** Types arriving from the call site, in interface order; absent: the declared input shapes. */
  inputs?: (TensorType | undefined)[];
}

export interface ProjectAnalysis {
  entry: Analysis;
  /** Every distinct analysis of each module, in the order first reached; unused modules get a standalone one. */
  byModule: Record<ModuleId, Analysis[]>;
}

const isInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v);

/** Validates one literal against its ParamSpec; returns an error message or undefined. */
export function checkParamValue(p: ParamSpec, v: Literal | undefined): string | undefined {
  if (v === null) return p.nullable ? undefined : msg(`${p.name} 不能为空`, `${p.name} cannot be None`);
  if (v === undefined) return msg(`缺少参数 ${p.name}`, `missing parameter ${p.name}`);
  const atLeast = (n: number) => p.min === undefined || n >= p.min;
  switch (p.type) {
    case 'int':
      if (!isInt(v)) return msg(`${p.name} 必须是整数`, `${p.name} must be an integer`);
      return atLeast(v) ? undefined : msg(`${p.name} 必须 ≥ ${p.min}`, `${p.name} must be ≥ ${p.min}`);
    case 'float':
      return typeof v === 'number' && Number.isFinite(v) ? undefined : msg(`${p.name} 必须是数字`, `${p.name} must be a number`);
    case 'bool':
      return typeof v === 'boolean' ? undefined : msg(`${p.name} 必须是布尔值`, `${p.name} must be True or False`);
    case 'enum':
      return typeof v === 'string' && p.options?.includes(v)
        ? undefined
        : msg(`${p.name} 必须是 ${p.options?.join(' / ')} 之一`, `${p.name} must be one of ${p.options?.join(' / ')}`);
    case 'shape':
      return typeof v === 'string' ? undefined : msg(`${p.name} 必须是形状文本`, `${p.name} must be shape text`);
    case 'ints': {
      if (typeof v === 'string') {
        return p.options?.includes(v)
          ? undefined
          : msg(`${p.name} 必须是整数或 ${p.options?.join(' / ') ?? '整数元组'}`, `${p.name} must be an integer or ${p.options?.join(' / ') ?? 'a tuple of integers'}`);
      }
      const values = Array.isArray(v) ? v : [v];
      if (values.length === 0 || !values.every(isInt)) return msg(`${p.name} 必须是整数或整数元组`, `${p.name} must be an integer or a tuple of integers`);
      if (Array.isArray(v) && p.arity !== undefined && v.length !== p.arity) return msg(`${p.name} 需要 ${p.arity} 个值`, `${p.name} needs ${p.arity} values`);
      return values.every((n) => atLeast(n as number)) ? undefined : msg(`${p.name} 的每个值都必须 ≥ ${p.min}`, `every value of ${p.name} must be ≥ ${p.min}`);
    }
    case 'dims':
      return Array.isArray(v) && v.length > 0 && v.every(isInt) ? undefined : msg(`${p.name} 必须是整数列表`, `${p.name} must be a list of integers`);
  }
}

function checkParams(spec: NodeSpec, params: Params): string | undefined {
  for (const p of spec.params) {
    const value = params[p.name];
    if (value === undefined) return msg(`缺少参数 ${p.name}`, `missing parameter ${p.name}`);
    if (isAuto(value)) {
      if (!p.allowAuto) return msg(`参数 ${p.name} 不支持自动推断`, `${p.name} cannot be inferred automatically`);
      continue;
    }
    const error = checkParamValue(p, literalOf(value));
    if (error) return error;
  }
  return undefined;
}

/**
 * Replaces expressions with their values in `env`. `unknown` names a parameter
 * whose value is only known when the model runs (it uses a preamble function).
 */
export function bindParams(params: Record<string, ParamValue>, env: Env): { params: Params } | { error: string } | { unknown: string } {
  const bound: Params = {};
  for (const [name, value] of Object.entries(params)) {
    const text = exprOf(value);
    if (text === undefined) {
      bound[name] = value;
      continue;
    }
    const result = evalExpr(text, env);
    if (!result.ok && result.unknown) return { unknown: `${name} = ${text}` };
    if (!result.ok) return { error: msg(`参数 ${name} = ${text}：${result.error}`, `${name} = ${text}: ${result.error}`) };
    bound[name] = { literal: result.value };
  }
  return { params: bound };
}

/** Edges into a variadic input, in port order. */
export function variadicEdges(mod: ModuleDef, nodeId: NodeId, base: string): Edge[] {
  return mod.edges
    .filter((e) => e.target.node === nodeId && variadicIndex(e.target.port, base) !== undefined)
    .sort((a, b) => variadicIndex(a.target.port, base)! - variadicIndex(b.target.port, base)!);
}

/** The value of a hyperparameter default or constant; defaults see no other names. */
function evalDefault(value: { literal: Literal } | { expr: string }, env: Env): Literal | undefined {
  if ('literal' in value) return value.literal;
  const r = evalExpr(value.expr, env);
  return r.ok ? r.value : undefined;
}

/**
 * Module-level constants from the preamble (`HIDDEN = 128`), which every
 * expression can use. Other preamble names are in scope with unknown values,
 * and so are the epilogue's, which code can use once the file has run.
 */
export function preambleEnv(project: LoomProject): Env {
  const env: Env = {};
  for (const line of project.preamble?.split('\n') ?? []) {
    const m = /^([A-Za-z_]\w*)\s*(?::[^=]+)?=(?!=)\s*(.+)$/.exec(line);
    if (m) {
      const r = evalExpr(m[2]!, env);
      env[m[1]!] = r.ok ? r.value : undefined;
    } else {
      const def = /^(?:def|class)\s+([A-Za-z_]\w*)/.exec(line);
      if (def) env[def[1]!] = undefined;
    }
  }
  for (const line of project.epilogue?.split('\n') ?? []) {
    const m = /^(?:(?:async\s+)?def|class)\s+([A-Za-z_]\w*)/.exec(line) ?? /^([A-Za-z_]\w*)\s*(?::[^=]+)?=(?!=)/.exec(line);
    if (m && !(m[1]! in env)) env[m[1]!] = undefined;
  }
  return env;
}

/** Values of a module's hyperparameters when it is used with its defaults (standalone, or as the entry). */
export function defaultEnv(mod: ModuleDef, globals: Env = {}): Env {
  const env: Env = { ...globals };
  for (const hp of mod.hyperparams) env[hp.name] = hp.default ? evalDefault(hp.default, globals) : undefined;
  return env;
}

/**
 * Layers that Python kept verbatim uses directly (`self.attn(...)` in a code
 * block, or in a helper method that forward() calls), so they need no edges.
 */
export function usedByCode(mod: ModuleDef): Set<string> {
  const names = new Set<string>();
  const texts = [mod.members ?? '', mod.initCode ?? '', ...Object.values(mod.nodes).map((n) => n.code?.source ?? '')];
  for (const text of texts) for (const m of text.matchAll(/\bself\.([A-Za-z_][A-Za-z0-9_]*)/g)) names.add(m[1]!);
  return names;
}

const errorText = (d: Diagnostic | undefined, mod: ModuleDef) =>
  d ? (d.nodeId && mod.nodes[d.nodeId] ? `${mod.nodes[d.nodeId]!.name}: ${d.message}` : d.message) : '';

class Analyzer {
  readonly byModule: Record<ModuleId, Analysis[]> = {};
  private readonly memo = new Map<string, Analysis>();
  private readonly stack: ModuleId[] = [];
  private readonly globals: Env;

  constructor(private readonly project: LoomProject) {
    this.globals = preambleEnv(project);
  }

  module(moduleId: ModuleId, ctx: ModuleContext | undefined, isEntry: boolean): Analysis {
    const mod = this.project.modules[moduleId]!;
    const context = ctx ?? { env: defaultEnv(mod, this.globals) };
    const key = `${moduleId}|${isEntry}|${JSON.stringify(context.env)}|${JSON.stringify(context.inputs ?? null)}`;
    const cached = this.memo.get(key);
    if (cached) return cached;
    this.stack.push(moduleId);
    const analysis = this.run(mod, context, isEntry);
    this.stack.pop();
    this.memo.set(key, analysis);
    (this.byModule[moduleId] ??= []).push(analysis);
    return analysis;
  }

  private run(mod: ModuleDef, ctx: ModuleContext, isEntry: boolean): Analysis {
    const project = this.project;
    const diagnostics: Diagnostic[] = [];
    const nodes: Record<NodeId, NodeAnalysis> = {};
    let partial = false;

    const env: Env = { ...ctx.env };
    for (const c of mod.constants ?? []) {
      const r = evalExpr(c.expr, env);
      env[c.name] = r.ok ? r.value : undefined;
    }

    if (mod.opaque) {
      return {
        moduleId: mod.id,
        order: [],
        nodes,
        edgeTypes: {},
        diagnostics,
        totalParams: 0,
        partial: true,
        outputs: mod.interface.outputs.map(() => undefined),
        env,
      };
    }

    const { order, cyclic } = topoOrder(mod);
    for (const nodeId of cyclic) {
      diagnostics.push({ severity: 'error', nodeId, message: msg('该节点处在环中；forward 中不允许环', 'this node is part of a cycle; forward() cannot have cycles') });
    }

    if (!mod.sequential) {
      let defaulted = false;
      const hpNames = new Set<string>();
      for (const hp of mod.hyperparams) {
        if (!isValidIdentifier(hp.name) || hpNames.has(hp.name)) {
          diagnostics.push({ severity: 'error', message: msg(`超参数名“${hp.name}”无效或重复`, `hyperparameter name "${hp.name}" is invalid or used twice`) });
        } else if (hp.default) {
          defaulted = true;
        } else if (defaulted) {
          diagnostics.push({ severity: 'error', message: msg(`没有默认值的超参数 ${hp.name} 不能排在有默认值的超参数之后`, `hyperparameter ${hp.name} has no default, so it cannot follow ones that do`) });
        }
        hpNames.add(hp.name);
      }
      const seen = new Set<string>();
      for (const node of Object.values(mod.nodes)) {
        if (node.kind === 'code') continue;
        if (!isValidIdentifier(node.name)) {
          diagnostics.push({ severity: 'error', nodeId: node.id, message: msg(`“${node.name}”不是合法的 Python 变量名`, `"${node.name}" is not a valid Python name`) });
        } else if (seen.has(node.name) || (mod.constants ?? []).some((c) => c.name === node.name)) {
          diagnostics.push({ severity: 'error', nodeId: node.id, message: msg(`变量名“${node.name}”重复`, `the name "${node.name}" is used twice`) });
        }
        seen.add(node.name);
      }
    }

    const incoming = incomingByPort(mod);
    const codeUses = usedByCode(mod);
    /** Output port index of a source node, or -1 if the port does not exist (any more). */
    const outputIndex = (nodeId: NodeId, port: string): number => {
      const node = mod.nodes[nodeId];
      const ports = node && portsOf(project, node, boundCache.get(nodeId));
      return ports ? ports.outputs.findIndex((o) => o.name === port) : -1;
    };
    const boundCache = new Map<NodeId, Params>();
    const typeOf = (e: Edge): TensorType | undefined => nodes[e.source.node]?.outputs?.[outputIndex(e.source.node, e.source.port)];
    const fail = (nodeId: NodeId, message: string) => diagnostics.push({ severity: 'error', nodeId, message });

    for (const nodeId of order) {
      const node = mod.nodes[nodeId]!;
      nodes[nodeId] = { inputs: [], paramCount: 0 };

      // Parameters first: a chunk's outputs depend on its bound parameters.
      let bound: Params = node.params;
      if (node.kind === 'layer' || node.kind === 'op') {
        const result = bindParams(node.params, env);
        if ('error' in result) {
          fail(nodeId, result.error);
          continue;
        }
        if ('unknown' in result) {
          diagnostics.push({
            severity: 'warning',
            nodeId,
            message: msg(`参数 ${result.unknown} 要运行时才知道，无法推断形状`, `${result.unknown} is only known when the model runs, so its shape cannot be inferred`),
          });
          continue;
        }
        bound = result.params;
        boundCache.set(nodeId, bound);
      }

      const ports = portsOf(project, node, bound);
      if (!ports) {
        fail(nodeId, node.kind === 'module' ? msg(`模块定义 ${node.type} 不存在`, `module ${node.type} does not exist`) : msg(`未知的节点类型 ${node.type}`, `unknown node type ${node.type}`));
        continue;
      }
      if (node.kind === 'module' && this.stack.includes(node.type)) {
        const name = project.modules[node.type]!.className;
        fail(nodeId, msg(`模块 ${name} 递归地包含了自己`, `module ${name} contains itself`));
        continue;
      }

      // Input node: the call site's type, or the declared example shape.
      if (node.kind === 'input') {
        const index = mod.interface.inputs.indexOf(nodeId);
        if (ctx.inputs) {
          const t = ctx.inputs[index];
          nodes[nodeId] = { inputs: [], outputs: [t], paramCount: 0 };
          continue;
        }
        const text = String(literalOf(node.params.shape) ?? '').trim();
        // No example shape: a module's input typed by its callers, or an optional argument left at its default.
        if (text === '' && (!isEntry || mod.interface.defaults?.[nodeId] !== undefined)) {
          nodes[nodeId] = { inputs: [], outputs: [undefined], paramCount: 0 };
          continue;
        }
      }

      const inputs: Inputs = [];
      const unconnected: string[] = [];
      let upstreamFailed = false;
      for (const port of ports.inputs) {
        const edges = port.variadic ? variadicEdges(mod, nodeId, port.name) : [incoming.get(portKey(nodeId, port.name))].filter((e): e is Edge => !!e);
        if (edges.length === 0) {
          if (!port.optional) unconnected.push(port.label);
          if (!port.variadic) inputs.push(undefined);
          continue;
        }
        for (const e of edges) {
          const t = typeOf(e);
          if (!t) upstreamFailed = true;
          inputs.push(t);
        }
      }
      nodes[nodeId]!.inputs = inputs;

      // Layers that only code blocks call have no edges, by design.
      const calledFromCode = (node.kind === 'layer' || node.kind === 'module') && codeUses.has(node.name) && !mod.edges.some((e) => e.target.node === nodeId);
      // A submodule forward() never calls (GoogLeNet's training-only aux heads, or one not wired up yet):
      // built in __init__ and holding weights, but not run.
      const unused = (node.kind === 'layer' || node.kind === 'module') && !mod.edges.some((e) => e.target.node === nodeId || e.source.node === nodeId);
      if (unused && !calledFromCode) diagnostics.push({ severity: 'warning', nodeId, message: msg('没有在 forward() 中使用', 'not used in forward()') });
      if (unconnected.length > 0 && !calledFromCode && !unused) {
        fail(nodeId, msg(`输入端口 ${unconnected.join('、')} 未连接`, `input ${unconnected.join(', ')} is not connected`));
        continue;
      }

      if (node.kind === 'module') {
        const result = this.moduleNode(mod, node, env, calledFromCode || unused ? undefined : inputs);
        nodes[nodeId] = { inputs, ...result.analysis };
        if (result.error) fail(nodeId, result.error);
        partial ||= result.partial;
        continue;
      }

      if (node.kind === 'code') {
        nodes[nodeId] = { inputs, outputs: ports.outputs.map((o) => declaredType(node, o.name)), paramCount: 0 };
        continue;
      }

      const spec = getSpec(node.type)!;
      const params = withDefaults(spec, bound);
      const paramError = checkParams(spec, params);
      if (paramError) {
        fail(nodeId, paramError);
        continue;
      }
      // An upstream failure is already reported there; don't cascade it downstream. Without
      // input types nothing is inferred, but the weights are known when every parameter is given.
      if (upstreamFailed || calledFromCode || unused) {
        if (!Object.values(params).some(isAuto) && !node.repeat) {
          nodes[nodeId]!.paramCount = spec.paramCount?.(Object.fromEntries(Object.entries(params).map(([k, v]) => [k, literalOf(v) ?? null]))) ?? 0;
        }
        continue;
      }

      if (node.repeat && node.kind === 'layer') {
        const result = this.repeatLayer(spec, node, params, inputs, env);
        if ('error' in result) fail(nodeId, result.error);
        else nodes[nodeId] = { inputs, ...result };
        continue;
      }

      const result = spec.infer(inputs, params);
      if (!result.ok) {
        fail(nodeId, result.error);
        continue;
      }
      nodes[nodeId] = { inputs, outputs: result.outputs, resolved: result.resolved, paramCount: spec.paramCount?.(result.resolved) ?? 0 };
    }
    for (const nodeId of cyclic) nodes[nodeId] = { inputs: [], paramCount: 0 };

    const edgeTypes: Record<string, TensorType> = {};
    for (const e of mod.edges) {
      if (outputIndex(e.source.node, e.source.port) < 0) {
        fail(e.source.node, msg(`输出端口 ${e.source.port} 已不存在，请重新连线`, `output ${e.source.port} no longer exists; reconnect it`));
        continue;
      }
      const t = typeOf(e);
      if (t) edgeTypes[e.id] = t;
    }

    const used = new Set(mod.edges.map((e) => e.source.node));
    for (const node of Object.values(mod.nodes)) {
      if (node.kind === 'output' || used.has(node.id) || codeUses.has(node.name)) continue;
      if (nodes[node.id]?.outputs?.some((t) => t)) {
        diagnostics.push({ severity: 'warning', nodeId: node.id, message: msg('输出未被使用', 'output is not used') });
      }
    }

    if (mod.sequential) this.checkSequential(mod, diagnostics);
    const kinds = new Set(Object.values(mod.nodes).map((n) => n.kind));
    if (!kinds.has('input')) diagnostics.push({ severity: 'warning', message: isEntry ? msg('模型还没有输入节点', 'the model has no input yet') : msg('模块还没有输入节点', 'the module has no input yet') });
    if (!kinds.has('output')) diagnostics.push({ severity: 'warning', message: isEntry ? msg('模型还没有输出节点', 'the model has no output yet') : msg('模块还没有输出节点', 'the module has no output yet') });
    if (isEntry) {
      for (const hp of mod.hyperparams) {
        if (!hp.default) diagnostics.push({ severity: 'error', message: msg(`入口模型的超参数 ${hp.name} 没有默认值，无法直接实例化`, `hyperparameter ${hp.name} of the entry model has no default, so it cannot be instantiated as is`) });
      }
    }

    const totalParams = Object.values(nodes).reduce((sum, n) => sum + n.paramCount, 0);
    const outputs = mod.interface.outputs.map((id) => nodes[id]?.inputs[0]);
    return { moduleId: mod.id, order, nodes, edgeTypes, diagnostics, totalParams, partial, outputs, env };
  }

  /** Binds an instance's hyperparameters, then analyzes its module once per copy. */
  private moduleNode(
    parent: ModuleDef,
    node: NodeInst,
    env: Env,
    inputs: Inputs | undefined,
  ): { analysis: Omit<NodeAnalysis, 'inputs'>; error?: string; partial: boolean } {
    const def = this.project.modules[node.type]!;
    const none = { analysis: { paramCount: 0 }, partial: false };
    const hpEnv = (overrides: Record<string, ParamValue> = {}): { env: Env } | { error: string } => {
      if (def.sequential) return { env };
      const childEnv: Env = { ...this.globals };
      const given = { ...node.params, ...overrides };
      for (const key of Object.keys(given)) {
        if (!def.hyperparams.some((hp) => hp.name === key)) return { error: msg(`${def.className} 没有超参数 ${key}`, `${def.className} has no hyperparameter ${key}`) };
      }
      for (const hp of def.hyperparams) {
        const value = given[hp.name];
        if (value === undefined) {
          if (!hp.default) {
            if (def.opaque) continue;
            return { error: msg(`缺少超参数 ${hp.name}`, `missing hyperparameter ${hp.name}`) };
          }
          childEnv[hp.name] = evalDefault(hp.default, this.globals);
        } else if ('auto' in value) {
          return { error: msg(`超参数 ${hp.name} 不能自动推断`, `hyperparameter ${hp.name} cannot be inferred`) };
        } else if ('literal' in value) {
          childEnv[hp.name] = value.literal;
        } else {
          const r = evalExpr(value.expr, env);
          if (!r.ok && !r.unknown) return { error: msg(`超参数 ${hp.name} = ${value.expr}：${r.error}`, `${hp.name} = ${value.expr}: ${r.error}`) };
          childEnv[hp.name] = r.ok ? r.value : undefined;
        }
      }
      return { env: childEnv };
    };

    const base = hpEnv();
    if ('error' in base) return { ...none, error: base.error };
    const resolved = Object.fromEntries(def.hyperparams.map((hp) => [hp.name, base.env[hp.name] ?? null])) as Resolved;

    if (def.opaque) {
      const outputs = def.interface.outputs.map((id) => declaredType(node, id));
      return { analysis: { outputs, resolved, paramCount: 0 }, partial: true };
    }
    if (!inputs) {
      // Not called with tensors here (only from code, or not at all): its weights still count.
      const child = this.module(def.id, { env: base.env }, false);
      return { analysis: { resolved, paramCount: child.totalParams, children: [child] }, partial: child.partial };
    }

    let copies = 1;
    let first: { env: Env } | { error: string } = base;
    if (node.repeat) {
      const count = paramNumber(node.repeat.count, env);
      if (typeof count === 'string') return { ...none, error: msg(`重复次数：${count}`, `repeat count: ${count}`) };
      if (count < 1) return { ...none, error: msg('重复次数必须 ≥ 1', 'the repeat count must be ≥ 1') };
      if (def.interface.outputs.length !== 1 || def.interface.inputs.length < 1) {
        return { ...none, error: msg('只有单输出、至少一个输入的模块才能重复', 'only modules with one output and at least one input can repeat') };
      }
      copies = count;
      if (node.repeat.first) first = hpEnv(node.repeat.first);
      if ('error' in first) return { ...none, error: first.error };
    }

    const children: Analysis[] = [];
    let current = inputs;
    for (let i = 0; i < copies; i++) {
      const childEnv = i === 0 ? (first as { env: Env }).env : base.env;
      const child = this.module(def.id, { env: childEnv, inputs: current }, false);
      children.push(child);
      current = [child.outputs[0], ...current.slice(1)];
    }
    const last = children.at(-1)!;
    const firstError = children.map((c) => c.diagnostics.find((d) => d.severity === 'error')).find((d) => d);
    const analysis: Omit<NodeAnalysis, 'inputs'> = {
      outputs: firstError ? undefined : last.outputs,
      resolved,
      paramCount: children.reduce((s, c) => s + c.totalParams, 0),
      children,
    };
    return {
      analysis,
      partial: children.some((c) => c.partial),
      error: firstError ? msg(`${def.className} 中有错误：${errorText(firstError, def)}`, `error inside ${def.className}: ${errorText(firstError, def)}`) : undefined,
    };
  }

  /** A layer repeated in series: each copy infers from the previous copy's output. */
  private repeatLayer(spec: NodeSpec, node: NodeInst, params: Params, inputs: Inputs, env: Env): Omit<NodeAnalysis, 'inputs'> | { error: string } {
    const count = paramNumber(node.repeat!.count, env);
    if (typeof count === 'string') return { error: msg(`重复次数：${count}`, `repeat count: ${count}`) };
    if (count < 1) return { error: msg('重复次数必须 ≥ 1', 'the repeat count must be ≥ 1') };
    if (spec.outputs.length !== 1 || spec.dynamicOutputs) return { error: msg('只有单输出的层才能重复', 'only single-output layers can repeat') };
    let firstParams = params;
    if (node.repeat!.first) {
      const result = bindParams(node.repeat!.first, env);
      if ('error' in result) return result;
      if ('unknown' in result) return { error: msg(`参数 ${result.unknown} 要运行时才知道`, `${result.unknown} is only known when the model runs`) };
      firstParams = { ...params, ...result.params };
      const error = checkParams(spec, firstParams);
      if (error) return { error };
    }
    let current = inputs;
    let paramCount = 0;
    let resolved: Resolved = {};
    let resolvedFirst: Resolved = {};
    let outputs: TensorType[] = [];
    for (let i = 0; i < count; i++) {
      const result = spec.infer(current, i === 0 ? firstParams : params);
      if (!result.ok) return { error: count > 1 ? msg(`第 ${i + 1} 次重复：${result.error}`, `copy ${i + 1}: ${result.error}`) : result.error };
      paramCount += spec.paramCount?.(result.resolved) ?? 0;
      if (i === 0) resolvedFirst = result.resolved;
      if (i === Math.min(1, count - 1)) resolved = result.resolved;
      outputs = result.outputs;
      current = [outputs[0], ...current.slice(1)];
    }
    return { outputs, resolved, resolvedFirst, paramCount };
  }

  private checkSequential(mod: ModuleDef, diagnostics: Diagnostic[]): void {
    const inner = Object.values(mod.nodes).filter((n) => n.kind !== 'input' && n.kind !== 'output');
    const bad = inner.find((n) => (n.kind !== 'layer' && n.kind !== 'module') || n.repeat);
    const chain = mod.interface.inputs.length === 1 && mod.interface.outputs.length === 1 && sequentialChain(mod).length === inner.length;
    if (bad || !chain) {
      diagnostics.push({
        severity: 'error',
        nodeId: bad?.id,
        message: msg('nn.Sequential 只能是由单输入单输出的层或模块组成的一条链', 'an nn.Sequential must be a single chain of one-input, one-output layers or modules'),
      });
    }
  }
}

/** The nodes of a sequential module in chain order (input to output), excluding both ends. */
export function sequentialChain(mod: ModuleDef): NodeInst[] {
  const chain: NodeInst[] = [];
  let current = mod.interface.inputs[0];
  const seen = new Set<NodeId>();
  while (current && !seen.has(current)) {
    seen.add(current);
    const out = mod.edges.filter((e) => e.source.node === current);
    if (out.length !== 1) break;
    const next = mod.nodes[out[0]!.target.node];
    if (!next || next.kind === 'output') break;
    if (mod.edges.filter((e) => e.target.node === next.id).length !== 1) break;
    chain.push(next);
    current = next.id;
  }
  return chain;
}

function declaredType(node: NodeInst, port: string): TensorType | undefined {
  const text = node.shapes?.[port];
  if (!text) return undefined;
  const [shape, dtype] = text.split(':').map((s) => s.trim());
  const parsed = parseShape(shape ?? '');
  return parsed.ok ? { shape: parsed.shape, dtype: dtype || 'float32' } : undefined;
}

/** An integer parameter that may be an expression; returns an error message when it is not one. */
function paramNumber(value: ParamValue, env: Env): number | string {
  let v: Literal | undefined;
  if ('literal' in value) v = value.literal;
  else if ('expr' in value) {
    const r = evalExpr(value.expr, env);
    if (!r.ok) return r.error;
    v = r.value;
  }
  return isInt(v) ? v : msg('需要整数', 'expected an integer');
}

export function analyzeProject(project: LoomProject): ProjectAnalysis {
  const analyzer = new Analyzer(project);
  const entry = analyzer.module(project.entry, undefined, true);
  for (const id of Object.keys(project.modules)) {
    if (!analyzer.byModule[id]) analyzer.module(id, undefined, false);
  }
  return { entry, byModule: analyzer.byModule };
}

/** One module analyzed on its own: the entry as the model, any other module with its defaults. */
export function analyzeModule(project: LoomProject, moduleId: ModuleId = project.entry, ctx?: ModuleContext): Analysis {
  return new Analyzer(project).module(moduleId, ctx, moduleId === project.entry && !ctx);
}

/** The analysis to show for a drill-down path: the first copy at each level. */
export function analysisAt(pa: ProjectAnalysis, project: LoomProject, path: NodeId[], moduleId: ModuleId): Analysis {
  let current: Analysis | undefined = pa.entry;
  for (const id of path) current = current?.nodes[id]?.children?.[0];
  if (current?.moduleId === moduleId) return current;
  return pa.byModule[moduleId]?.[0] ?? analyzeModule(project, moduleId);
}

/** Free names in a node's expressions that are not hyperparameters or constants of its module. */
export function undefinedNames(mod: ModuleDef, node: NodeInst): string[] {
  const known = new Set([...mod.hyperparams.map((h) => h.name), ...(mod.constants ?? []).map((c) => c.name)]);
  const texts = Object.values(node.params).flatMap((v) => (exprOf(v) ? [exprOf(v)!] : []));
  return [...new Set(texts.flatMap(freeNames))].filter((n) => !known.has(n));
}

export { errorText as diagnosticText };
