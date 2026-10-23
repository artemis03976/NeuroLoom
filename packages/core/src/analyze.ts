// Shape inference and validation for one module (PRD section 8.4–8.5).
import { incomingByPort, isValidIdentifier, portKey, topoOrder } from './graph';
import { isAuto, literalOf, type ModuleDef, type NodeId, type ParamValue } from './ir';
import type { TensorType } from './shape';
import { getSpec, type NodeSpec, type Resolved } from './specs';

export interface Diagnostic {
  severity: 'error' | 'warning';
  message: string;
  nodeId?: NodeId;
}

export interface NodeAnalysis {
  inputs: (TensorType | undefined)[];
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

function checkParams(spec: NodeSpec, params: Record<string, ParamValue>): string | undefined {
  for (const p of spec.params) {
    const value = params[p.name];
    if (value === undefined) return `缺少参数 ${p.name}`;
    if (isAuto(value)) {
      if (!p.allowAuto) return `参数 ${p.name} 不支持自动推断`;
      continue;
    }
    const v = literalOf(value);
    switch (p.type) {
      case 'int':
        if (typeof v !== 'number' || !Number.isInteger(v)) return `${p.name} 必须是整数`;
        if (p.min !== undefined && v < p.min) return `${p.name} 必须 ≥ ${p.min}`;
        break;
      case 'float':
        if (typeof v !== 'number' || !Number.isFinite(v)) return `${p.name} 必须是数字`;
        break;
      case 'bool':
        if (typeof v !== 'boolean') return `${p.name} 必须是布尔值`;
        break;
      case 'shape':
        if (typeof v !== 'string') return `${p.name} 必须是形状文本`;
        break;
    }
  }
  return undefined;
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
  const outputIndex = (nodeId: NodeId, port: string): number => {
    const spec = getSpec(mod.nodes[nodeId]?.type);
    return Math.max(0, spec?.outputs.findIndex((o) => o.name === port) ?? 0);
  };

  for (const nodeId of order) {
    const node = mod.nodes[nodeId]!;
    const spec = getSpec(node.type);
    if (!spec) {
      diagnostics.push({ severity: 'error', nodeId, message: `未知的节点类型 ${node.type}` });
      nodes[nodeId] = { inputs: [], paramCount: 0 };
      continue;
    }

    const inputs = spec.inputs.map((p) => {
      const e = incoming.get(portKey(nodeId, p.name));
      return e ? nodes[e.source.node]?.outputs?.[outputIndex(e.source.node, e.source.port)] : undefined;
    });
    nodes[nodeId] = { inputs, paramCount: 0 };

    const unconnected = spec.inputs.filter((p) => !incoming.has(portKey(nodeId, p.name)));
    if (unconnected.length > 0) {
      diagnostics.push({ severity: 'error', nodeId, message: `输入端口 ${unconnected.map((p) => p.name).join('、')} 未连接` });
      continue;
    }
    const params = withDefaults(spec, node.params);
    const paramError = checkParams(spec, params);
    if (paramError) {
      diagnostics.push({ severity: 'error', nodeId, message: paramError });
      continue;
    }
    // An upstream failure is already reported there; don't cascade it downstream.
    if (inputs.some((t) => t === undefined)) continue;

    const result = spec.infer(inputs as TensorType[], params);
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
    const t = nodes[e.source.node]?.outputs?.[outputIndex(e.source.node, e.source.port)];
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
