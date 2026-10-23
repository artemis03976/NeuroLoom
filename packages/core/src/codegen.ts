// IR -> PyTorch source (PRD section 10). Output is deterministic: the same graph
// always produces byte-identical code, and every value has one canonical spelling
// so that code -> graph -> code is a fixed point.
import { analyzeModule, variadicEdges, withDefaults, type Analysis } from './analyze';
import { consumers, incomingByPort, interfaceOrder, portKey } from './graph';
import { entryModule, literalOf, type Literal, type LoomProject, type ModuleId, type NodeId, type NodeInst } from './ir';
import { formatShape } from './shape';
import { getSpec, outputsOf, type NodeSpec, type ParamSpec, type Resolved } from './specs';

export interface CodegenResult {
  code: string;
  /** 1-based line numbers per node: its constructor line and its forward line. */
  lineMap: Record<NodeId, number[]>;
  /** Number of TODO(neuroloom) markers written for unfinished parts of the graph. */
  todos: number;
}

const I1 = '    ';
const I2 = I1 + I1;

/** Canonical form of a parameter value: uniform tuples collapse to a scalar (kernel_size (3, 3) -> 3). */
export function canonicalValue(value: Literal | undefined, p: ParamSpec): Literal | undefined {
  if (p.type === 'ints' && Array.isArray(value) && value.length > 0) {
    if (value.length === 1 && p.arity === undefined) return value[0]!;
    if (p.arity !== undefined && value.every((v) => v === value[0])) return value[0]!;
  }
  return value;
}

export function sameValue(a: Literal | undefined, b: Literal | undefined, p: ParamSpec): boolean {
  return JSON.stringify(canonicalValue(a, p)) === JSON.stringify(canonicalValue(b, p));
}

function pyNumber(value: number, float: boolean): string {
  if (float && Number.isInteger(value)) return `${value}.0`;
  return String(value);
}

export function pyLiteral(value: Literal | undefined, p?: ParamSpec): string {
  const v = p ? canonicalValue(value, p) : value;
  if (v === undefined || v === null) return 'None';
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (typeof v === 'number') return pyNumber(v, p?.type === 'float');
  if (typeof v === 'string') return JSON.stringify(v);
  const items = v.map((item) => pyLiteral(item));
  return items.length === 1 ? `(${items[0]},)` : `(${items.join(', ')})`;
}

function paramValue(spec: NodeSpec, node: NodeInst, resolved: Resolved | undefined, p: ParamSpec): Literal | undefined {
  return resolved?.[p.name] ?? literalOf(withDefaults(spec, node.params)[p.name]);
}

/** Constructor arguments: required params positionally, optional ones as keywords when not default. */
function ctorArgs(spec: NodeSpec, node: NodeInst, resolved: Resolved | undefined): { args: string[]; missing: string[] } {
  const args: string[] = [];
  const missing: string[] = [];
  for (const p of spec.params) {
    const value = paramValue(spec, node, resolved, p);
    if (p.default === undefined) {
      if (value === undefined) missing.push(p.name);
      args.push(pyLiteral(value, p));
    } else if (value !== undefined && !sameValue(value, p.default, p)) {
      args.push(`${p.name}=${pyLiteral(value, p)}`);
    }
  }
  return { args, missing };
}

/**
 * Builds an op call from the spec's Python signature. Arguments are positional
 * until one is skipped or written as a keyword; after that, keywords only.
 */
function opCall(
  spec: NodeSpec,
  node: NodeInst,
  resolved: Resolved | undefined,
  portExpr: (port: string) => string | undefined,
  variadicExprs: (base: string) => string[],
): string {
  const py = spec.python!;
  const paramOf = (name: string) => spec.params.find((p) => p.name === name)!;
  const parts: string[] = [];
  let receiver: string | undefined;
  let keywordsOnly = false;
  const push = (name: string, expr: string, keyword: boolean) => {
    if (keyword || keywordsOnly) {
      parts.push(`${name}=${expr}`);
      keywordsOnly = true;
    } else parts.push(expr);
  };

  py.args.forEach((arg, i) => {
    if ('ports' in arg) {
      push(arg.name, `[${variadicExprs(arg.ports).join(', ')}]`, false);
    } else if ('port' in arg) {
      const expr = portExpr(arg.port);
      if (expr !== undefined) {
        if (spec.codegen.method && i === 0) receiver = expr;
        else push(arg.name, expr, false);
      } else if (arg.scalarParam) {
        push(arg.name, pyLiteral(paramValue(spec, node, resolved, paramOf(arg.scalarParam)), paramOf(arg.scalarParam)), false);
      } else {
        keywordsOnly = true;
      }
    } else {
      const p = paramOf(arg.param);
      const value = paramValue(spec, node, resolved, p);
      if (p.default !== undefined && sameValue(value, p.default, p)) {
        keywordsOnly = true;
        return;
      }
      if (p.spread && Array.isArray(value) && !keywordsOnly) {
        parts.push(...value.map((v) => pyLiteral(v)));
        return;
      }
      push(arg.name, pyLiteral(value, p), p.keyword === true || (p.default !== undefined && !p.positional));
    }
  });

  // Operators take exactly two operands: a tensor and a tensor or constant.
  if (spec.codegen.operator) return `${parts[0]} ${spec.codegen.operator} ${parts[1]}`;
  if (spec.codegen.method) return `${receiver}.${spec.codegen.method}(${parts.join(', ')})`;
  return `${spec.codegen.fn}(${parts.join(', ')})`;
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
    const { args, missing } = ctorArgs(spec, node, analysis.nodes[node.id]?.resolved);
    let line = `${I2}self.${node.name} = ${spec.codegen.ctor}(${args.join(', ')})`;
    if (missing.length > 0) {
      line += `  # TODO(neuroloom): 无法确定 ${missing.join(', ')}`;
      todos++;
    }
    emit(line, node.id);
  }
  emit('');

  // Signature and return order are part of the model's interface, so they follow
  // the declared interface and never canvas position.
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
  const errorsOf = new Map<NodeId, string>();
  for (const d of analysis.diagnostics) {
    if (d.severity === 'error' && d.nodeId && !errorsOf.has(d.nodeId)) errorsOf.set(d.nodeId, d.message);
  }
  let returned = false;

  for (const node of ordered) {
    const spec = specOf(node);
    if (!spec) continue;
    if (node.kind === 'input') {
      varOf.set(portKey(node.id, 'out'), node.name);
      continue;
    }
    if (node.kind === 'output') continue;

    const variadicExprs = (base: string) =>
      variadicEdges(mod, node.id, base).map((e) => varOf.get(portKey(e.source.node, e.source.port)) ?? 'None');
    const connected = spec.inputs.filter((p) => !p.optional);
    const missingInput = connected.some((p) => (p.variadic ? variadicExprs(p.name).length === 0 : sourceVar(node.id, p.name) === undefined));
    if (missingInput) {
      emit(`${I2}# TODO(neuroloom): ${node.name} 的输入未连接`, node.id);
      todos++;
      continue;
    }

    const resolved = analysis.nodes[node.id]?.resolved;
    let expr: string;
    if (spec.kind === 'layer') {
      const args: string[] = [];
      let keywordsOnly = false;
      for (const p of spec.inputs) {
        const v = sourceVar(node.id, p.name);
        if (v === undefined) keywordsOnly = true;
        else args.push(keywordsOnly ? `${p.name}=${v}` : v);
      }
      expr = `self.${node.name}(${args.join(', ')})`;
    } else {
      expr = opCall(spec, node, resolved, (port) => sourceVar(node.id, port), variadicExprs);
    }

    const shape = analysis.nodes[node.id]?.outputs?.[0]?.shape;
    const problem = errorsOf.get(node.id);
    let comment = shape ? `  # ${formatShape(shape)}` : '';
    if (problem) {
      comment = `  # TODO(neuroloom): ${problem}`;
      todos++;
    }

    const outs = outputsOf(spec, withDefaults(spec, node.params));
    if (outs.length !== 1) {
      // Tuple results: one name per used output, "_" for the rest.
      const names = outs.map((o) => (consumers(mod, node.id, o.name).length > 0 ? `${node.name}_${o.name}` : '_'));
      outs.forEach((o, i) => names[i] !== '_' && varOf.set(portKey(node.id, o.name), names[i]!));
      const lhs = spec.codegen.unpack
        ? outs.reduce((text, o, i) => text.replace(`{${o.name}}`, names[i]!), spec.codegen.unpack)
        : names.join(', ');
      emit(names.every((n) => n === '_') ? `${I2}${expr}${comment}` : `${I2}${lhs} = ${expr}${comment}`, node.id);
      continue;
    }

    const outPort = outs[0]!.name;
    const uses = consumers(mod, node.id, outPort);
    if (outputs.length === 1 && uses.length === 1 && uses[0]!.target.node === outputs[0]!.id) {
      emit(`${I2}return ${expr}${comment}`, node.id, outputs[0]!.id);
      returned = true;
      continue;
    }
    let target = node.name;
    const tensorInputs = spec.inputs.flatMap((p) => (p.variadic ? variadicEdges(mod, node.id, p.name) : [incoming.get(portKey(node.id, p.name))])).filter((e) => e);
    const only = tensorInputs.length === 1 ? tensorInputs[0] : undefined;
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
