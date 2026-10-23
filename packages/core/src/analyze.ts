// Shape inference and validation for one module (PRD section 8.4–8.5).
import { incomingByPort, isValidIdentifier, portKey, topoOrder } from './graph';
import { isAuto, literalOf, type Edge, type Literal, type ModuleDef, type NodeId, type ParamValue } from './ir';
import type { TensorType } from './shape';
import { getSpec, outputsOf, variadicIndex, type Inputs, type NodeSpec, type ParamSpec, type Resolved } from './specs';

export interface Diagnostic {
  severity: 'error' | 'warning';
  message: string;
  nodeId?: NodeId;
}

export interface NodeAnalysis {
  inputs: Inputs;
  /** Undefined when inference failed here or upstream. */
  outputs?: TensorType[];
  resolved?: Resolved;
  paramCount: number;
}

export interface Analysis {
  order: NodeId[];
  nodes: Record<NodeId, NodeAnalysis>;
  edgeTypes: Record<string, TensorType>;
  diagnostics: Diagnostic[];
  totalParams: number;
}

/** Fills optional parameters with their defaults. */
export function withDefaults(spec: NodeSpec, params: Record<string, ParamValue>): Record<string, ParamValue> {
  const full: Record<string, ParamValue> = { ...params };
  for (const p of spec.params) {
    if (full[p.name] === undefined && p.default !== undefined) full[p.name] = { literal: p.default };
  }
  return full;
}

const isInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v);

/** Validates one literal against its ParamSpec; returns an error message or undefined. */
export function checkParamValue(p: ParamSpec, v: Literal | undefined): string | undefined {
  if (v === null) return p.nullable ? undefined : `${p.name} 不能为空`;
  if (v === undefined) return `缺少参数 ${p.name}`;
  const atLeast = (n: number) => p.min === undefined || n >= p.min;
  switch (p.type) {
    case 'int':
      if (!isInt(v)) return `${p.name} 必须是整数`;
      return atLeast(v) ? undefined : `${p.name} 必须 ≥ ${p.min}`;
    case 'float':
      return typeof v === 'number' && Number.isFinite(v) ? undefined : `${p.name} 必须是数字`;
    case 'bool':
      return typeof v === 'boolean' ? undefined : `${p.name} 必须是布尔值`;
    case 'enum':
      return typeof v === 'string' && p.options?.includes(v) ? undefined : `${p.name} 必须是 ${p.options?.join(' / ')} 之一`;
    case 'shape':
      return typeof v === 'string' ? undefined : `${p.name} 必须是形状文本`;
    case 'ints': {
      if (typeof v === 'string') return p.options?.includes(v) ? undefined : `${p.name} 必须是整数或 ${p.options?.join(' / ') ?? '整数元组'}`;
      const values = Array.isArray(v) ? v : [v];
      if (values.length === 0 || !values.every(isInt)) return `${p.name} 必须是整数或整数元组`;
      if (Array.isArray(v) && p.arity !== undefined && v.length !== p.arity) return `${p.name} 需要 ${p.arity} 个值`;
      return values.every((n) => atLeast(n as number)) ? undefined : `${p.name} 的每个值都必须 ≥ ${p.min}`;
    }
    case 'dims':
      return Array.isArray(v) && v.length > 0 && v.every(isInt) ? undefined : `${p.name} 必须是整数列表`;
  }
}

function checkParams(spec: NodeSpec, params: Record<string, ParamValue>): string | undefined {
  for (const p of spec.params) {
    const value = params[p.name];
    if (value === undefined) return `缺少参数 ${p.name}`;
    if (isAuto(value)) {
      if (!p.allowAuto) return `参数 ${p.name} 不支持自动推断`;
      continue;
    }
    const error = checkParamValue(p, literalOf(value));
    if (error) return error;
  }
  return undefined;
}

/** Edges into a variadic input, in port order. */
export function variadicEdges(mod: ModuleDef, nodeId: NodeId, base: string): Edge[] {
  return mod.edges
    .filter((e) => e.target.node === nodeId && variadicIndex(e.target.port, base) !== undefined)
    .sort((a, b) => variadicIndex(a.target.port, base)! - variadicIndex(b.target.port, base)!);
}

export function analyzeModule(mod: ModuleDef): Analysis {
  const diagnostics: Diagnostic[] = [];
  const { order, cyclic } = topoOrder(mod);
  for (const nodeId of cyclic) {
    diagnostics.push({ severity: 'error', nodeId, message: '该节点处在环中；forward 中不允许环' });
  }

  const seen = new Set<string>();
  for (const node of Object.values(mod.nodes)) {
    if (!isValidIdentifier(node.name)) {
      diagnostics.push({ severity: 'error', nodeId: node.id, message: `“${node.name}”不是合法的 Python 变量名` });
    } else if (seen.has(node.name)) {
      diagnostics.push({ severity: 'error', nodeId: node.id, message: `变量名“${node.name}”重复` });
    }
    seen.add(node.name);
  }

  const incoming = incomingByPort(mod);
  const nodes: Record<NodeId, NodeAnalysis> = {};
  /** Output port index of a source node, or -1 if the port does not exist (any more). */
  const outputIndex = (nodeId: NodeId, port: string): number => {
    const node = mod.nodes[nodeId];
    const spec = getSpec(node?.type);
    if (!node || !spec) return -1;
    return outputsOf(spec, withDefaults(spec, node.params)).findIndex((o) => o.name === port);
  };
  const typeOf = (e: Edge): TensorType | undefined => nodes[e.source.node]?.outputs?.[outputIndex(e.source.node, e.source.port)];

  for (const nodeId of order) {
    const node = mod.nodes[nodeId]!;
    const spec = getSpec(node.type);
    if (!spec) {
      diagnostics.push({ severity: 'error', nodeId, message: `未知的节点类型 ${node.type}` });
      nodes[nodeId] = { inputs: [], paramCount: 0 };
      continue;
    }

    const inputs: Inputs = [];
    const unconnected: string[] = [];
    let upstreamFailed = false;
    for (const port of spec.inputs) {
      const edges = port.variadic ? variadicEdges(mod, nodeId, port.name) : [incoming.get(portKey(nodeId, port.name))].filter((e): e is Edge => !!e);
      if (edges.length === 0) {
        if (!port.optional) unconnected.push(port.name);
        if (!port.variadic) inputs.push(undefined);
        continue;
      }
      for (const e of edges) {
        const t = typeOf(e);
        if (!t) upstreamFailed = true;
        inputs.push(t);
      }
    }
    nodes[nodeId] = { inputs, paramCount: 0 };

    if (unconnected.length > 0) {
      diagnostics.push({ severity: 'error', nodeId, message: `输入端口 ${unconnected.join('、')} 未连接` });
      continue;
    }
    const params = withDefaults(spec, node.params);
    const paramError = checkParams(spec, params);
    if (paramError) {
      diagnostics.push({ severity: 'error', nodeId, message: paramError });
      continue;
    }
    // An upstream failure is already reported there; don't cascade it downstream.
    if (upstreamFailed) continue;

    const result = spec.infer(inputs, params);
    if (!result.ok) {
      diagnostics.push({ severity: 'error', nodeId, message: result.error });
      continue;
    }
    nodes[nodeId] = {
      inputs,
      outputs: result.outputs,
      resolved: result.resolved,
      paramCount: spec.paramCount?.(result.resolved) ?? 0,
    };
  }
  for (const nodeId of cyclic) nodes[nodeId] = { inputs: [], paramCount: 0 };

  const edgeTypes: Record<string, TensorType> = {};
  for (const e of mod.edges) {
    if (outputIndex(e.source.node, e.source.port) < 0) {
      diagnostics.push({ severity: 'error', nodeId: e.source.node, message: `输出端口 ${e.source.port} 已不存在，请重新连线` });
      continue;
    }
    const t = typeOf(e);
    if (t) edgeTypes[e.id] = t;
  }

  const used = new Set(mod.edges.map((e) => e.source.node));
  for (const node of Object.values(mod.nodes)) {
    if (node.kind !== 'output' && !used.has(node.id) && nodes[node.id]?.outputs) {
      diagnostics.push({ severity: 'warning', nodeId: node.id, message: '输出未被使用' });
    }
  }

  const kinds = new Set(Object.values(mod.nodes).map((n) => n.kind));
  if (!kinds.has('input')) diagnostics.push({ severity: 'warning', message: '模型还没有输入节点' });
  if (!kinds.has('output')) diagnostics.push({ severity: 'warning', message: '模型还没有输出节点' });

  const totalParams = Object.values(nodes).reduce((sum, n) => sum + n.paramCount, 0);
  return { order, nodes, edgeTypes, diagnostics, totalParams };
}
