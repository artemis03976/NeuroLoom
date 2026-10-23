// IR -> PyTorch source (PRD section 10). Output is deterministic: the same graph
// always produces byte-identical code.
import { analyzeModule, withDefaults, type Analysis } from './analyze';
import { consumers, incomingByPort, interfaceOrder, portKey } from './graph';
import { entryModule, literalOf, type Literal, type LoomProject, type ModuleId, type NodeId, type NodeInst } from './ir';
import { formatShape } from './shape';
import { getSpec, type NodeSpec, type ParamSpec } from './specs';

export interface CodegenResult {
  code: string;
  /** 1-based line numbers per node: its constructor line and its forward line. */
  lineMap: Record<NodeId, number[]>;
  /** Number of TODO(neuroloom) markers written for unfinished parts of the graph. */
  todos: number;
}

const I1 = '    ';
const I2 = I1 + I1;

function pyLiteral(value: Literal | undefined, type?: ParamSpec['type']): string {
  if (value === undefined || value === null) return 'None';
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value === 'number') return type === 'float' && Number.isInteger(value) ? `${value}.0` : String(value);
  return JSON.stringify(value);
}

/** Required params are positional; optional ones are keywords, written only when not default. */
function callArgs(spec: NodeSpec, node: NodeInst, resolved: Record<string, Literal> | undefined): { args: string[]; missing: string[] } {
  const params = withDefaults(spec, node.params);
  const args: string[] = [];
  const missing: string[] = [];
  for (const p of spec.params) {
    const value = resolved?.[p.name] ?? literalOf(params[p.name]);
    if (p.default === undefined) {
      if (value === undefined) missing.push(p.name);
      args.push(pyLiteral(value, p.type));
    } else if (value !== undefined && value !== p.default) {
      args.push(`${p.name}=${pyLiteral(value, p.type)}`);
    }
  }
  return { args, missing };
}

export function generateCode(project: LoomProject, analyses: Record<ModuleId, Analysis> = {}): CodegenResult {
  const mod = entryModule(project);
  const analysis = analyses[mod.id] ?? analyzeModule(mod);
  const incoming = incomingByPort(mod);
  const ordered = analysis.order.map((id) => mod.nodes[id]!);
  const specOf = (n: NodeInst) => getSpec(n.type);

  const lines: string[] = [];
  const lineMap: Record<NodeId, number[]> = {};
  let todos = 0;
  const emit = (line: string, ...nodeIds: NodeId[]) => {
    lines.push(line);
    for (const id of nodeIds) (lineMap[id] ??= []).push(lines.length);
  };

  // isort order: plain imports first, then from-imports.
  emit('import torch');
  if (ordered.some((n) => specOf(n)?.codegen.fn?.startsWith('F.'))) emit('import torch.nn.functional as F');
  emit('from torch import nn');
  emit('');
  emit('');
  emit(`class ${mod.className}(nn.Module):`);
  emit(`${I1}def __init__(self):`);
  emit(`${I2}super().__init__()`);
  for (const node of ordered) {
    const spec = specOf(node);
    if (spec?.kind !== 'layer' || !spec.codegen.ctor) continue;
    const { args, missing } = callArgs(spec, node, analysis.nodes[node.id]?.resolved);
    let line = `${I2}self.${node.name} = ${spec.codegen.ctor}(${args.join(', ')})`;
    if (missing.length > 0) {
      line += `  # TODO(neuroloom): 无法确定 ${missing.join(', ')}`;
      todos++;
    }
    emit(line, node.id);
  }
  emit('');

  // Signature and return order are part of the model's interface, so they follow
  // creation order and never canvas position (moving a node must not reorder them).
  const inputs = interfaceOrder(mod, 'input');
  const outputs = interfaceOrder(mod, 'output');
  const returnType =
    outputs.length === 0 ? 'None' : outputs.length === 1 ? 'torch.Tensor' : `tuple[${outputs.map(() => 'torch.Tensor').join(', ')}]`;
  const signature = ['self', ...inputs.map((n) => `${n.name}: torch.Tensor`)].join(', ');
  emit(`${I1}def forward(${signature}) -> ${returnType}:`);
  const bodyStart = lines.length;

  // Variable naming: a value consumed exactly once hands its variable name on
  // (x = self.fc1(x)); a value used several times keeps its own name.
  const varOf = new Map<string, string>();
  const sourceVar = (nodeId: NodeId, port: string): string | undefined => {
    const e = incoming.get(portKey(nodeId, port));
    return e ? varOf.get(portKey(e.source.node, e.source.port)) : undefined;
  };
  let returned = false;
  // Errors are written next to the offending call so an exported file still points at them.
  const errorsOf = new Map<NodeId, string>();
  for (const d of analysis.diagnostics) {
    if (d.severity === 'error' && d.nodeId && !errorsOf.has(d.nodeId)) errorsOf.set(d.nodeId, d.message);
  }

  for (const node of ordered) {
    const spec = specOf(node);
    if (!spec) continue;
    if (node.kind === 'input') {
      varOf.set(portKey(node.id, 'out'), node.name);
      continue;
    }
    if (node.kind === 'output') continue;

    const args = spec.inputs.map((p) => sourceVar(node.id, p.name));
    if (args.some((a) => a === undefined)) {
      emit(`${I2}# TODO(neuroloom): ${node.name} 的输入未连接`, node.id);
      todos++;
      continue;
    }
    let expr: string;
    if (spec.kind === 'layer') {
      expr = `self.${node.name}(${args.join(', ')})`;
    } else {
      const { args: kwargs } = callArgs(spec, node, analysis.nodes[node.id]?.resolved);
      expr = `${spec.codegen.fn}(${[...(args as string[]), ...kwargs].join(', ')})`;
    }
    const shape = analysis.nodes[node.id]?.outputs?.[0]?.shape;
    const problem = errorsOf.get(node.id);
    let comment = shape ? `  # ${formatShape(shape)}` : '';
    if (problem) {
      comment = `  # TODO(neuroloom): ${problem}`;
      todos++;
    }
    const outPort = spec.outputs[0]?.name ?? 'out';
    const uses = consumers(mod, node.id, outPort);

    if (outputs.length === 1 && uses.length === 1 && uses[0]!.target.node === outputs[0]!.id) {
      emit(`${I2}return ${expr}${comment}`, node.id, outputs[0]!.id);
      returned = true;
      continue;
    }
    let target = node.name;
    const only = spec.inputs.length === 1 ? incoming.get(portKey(node.id, spec.inputs[0]!.name)) : undefined;
    if (only && consumers(mod, only.source.node, only.source.port).length === 1) {
      target = varOf.get(portKey(only.source.node, only.source.port))!;
    }
    emit(`${I2}${target} = ${expr}${comment}`, node.id);
    varOf.set(portKey(node.id, outPort), target);
  }

  if (!returned && outputs.length > 0) {
    const values = outputs.map((o) => sourceVar(o.id, 'in'));
    if (values.some((v) => v === undefined)) {
      emit(`${I2}return None  # TODO(neuroloom): 有输出节点未连接`, ...outputs.map((o) => o.id));
      todos++;
    } else {
      emit(`${I2}return ${values.join(', ')}`, ...outputs.map((o) => o.id));
    }
  } else if (lines.length === bodyStart) {
    emit(`${I2}pass`);
  }

  return { code: `${lines.join('\n')}\n`, lineMap, todos };
}
