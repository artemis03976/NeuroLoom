// Model cost (PRD F8.2): parameters, forward FLOPs and memory, per node and per
// module instance. FLOPs follow torch.utils.flop_counter.FlopCounterMode: a
// multiply-add counts 2, and only matrix products, convolutions and attention
// count at all (elementwise ops, norms and pooling are 0). Memory is parameters
// plus every tensor the forward pass produces, at the sizes the symbolic
// dimensions are given here (B = 1, T = 128, …).
import { bindParams, withDefaults, type Analysis, type NodeAnalysis } from './analyze';
import { evalExpr, type Env } from './expr';
import type { LoomProject, ModuleId, NodeId, NodeInst, ParamValue } from './ir';
import type { Dim, TensorType } from './shape';
import { getSpec, type Inputs, type NodeSpec, type Resolved } from './specs';

/** A value for each symbolic dimension. */
export type SymbolValues = Record<string, number>;

const DTYPE_BYTES: Record<string, number> = {
  bool: 1,
  int8: 1,
  uint8: 1,
  int16: 2,
  float16: 2,
  bfloat16: 2,
  int32: 4,
  float32: 4,
  int64: 8,
  float64: 8,
  complex64: 8,
  complex128: 16,
};

export const dtypeBytes = (dtype: string): number => DTYPE_BYTES[dtype] ?? 4;

const SYMBOL = /^[A-Za-z_][A-Za-z0-9_]*$/;

/** A dimension at the given symbol values; NaN when it is unknown or a symbol has no value. */
export function dimValue(d: Dim, values: SymbolValues): number {
  if (typeof d === 'number') return d;
  return d.split('+').reduce(
    (sum, term) =>
      sum +
      term.split('*').reduce((p, f) => {
        const t = f.trim();
        if (/^\d+$/.test(t)) return p * Number(t);
        return SYMBOL.test(t) && values[t] !== undefined ? p * values[t]! : NaN;
      }, 1),
    0,
  );
}

/** Concrete sizes of a shape, or undefined when any dimension is unknown. */
export function shapeValue(shape: Dim[], values: SymbolValues): number[] | undefined {
  const sizes = shape.map((d) => dimValue(d, values));
  return sizes.every(Number.isFinite) ? sizes : undefined;
}

/** Bytes of a tensor, or NaN when its size is unknown. */
export function tensorBytes(t: TensorType | undefined, values: SymbolValues): number {
  const sizes = t && shapeValue(t.shape, values);
  return sizes ? sizes.reduce((a, b) => a * b, 1) * dtypeBytes(t!.dtype) : NaN;
}

/** Every symbolic dimension used anywhere in the analysis (and the instances inside it), sorted with B first. */
export function symbolsOf(analysis: Analysis): string[] {
  const found = new Set<string>();
  const addShape = (t: TensorType | undefined) => {
    for (const d of t?.shape ?? []) {
      if (typeof d !== 'string') continue;
      for (const f of d.split(/[+*]/)) if (SYMBOL.test(f.trim())) found.add(f.trim());
    }
  };
  const walk = (a: Analysis) => {
    for (const n of Object.values(a.nodes)) {
      n.inputs.forEach(addShape);
      n.outputs?.forEach(addShape);
      n.children?.forEach(walk);
    }
  };
  walk(analysis);
  return [...found].sort((a, b) => (a === 'B' ? -1 : b === 'B' ? 1 : a.localeCompare(b)));
}

/** Defaults for the symbols: a batch of 1, 128 for everything else (sequence length, image size). */
export const defaultSymbolValue = (symbol: string): number => (symbol === 'B' ? 1 : 128);

export interface CostRow {
  /** Module node ids from the entry down to the module holding this node (the canvas view path). */
  path: NodeId[];
  moduleId: ModuleId;
  nodeId: NodeId;
  /** Dotted name as in `named_modules()`: layer1.0.conv1. */
  name: string;
  /** Layer type or class name. */
  type: string;
  params: number;
  /** Forward FLOPs at the symbol values. */
  flops: number;
  /** Bytes of the tensors this node (or everything inside the module) produces. */
  activationBytes: number;
  /** Something inside is unknown: a code block, an opaque module, or a shape inference could not tell. */
  partial: boolean;
  /** Module nodes: their nodes; repeated modules: one row per copy. */
  children?: CostRow[];
}

export interface Cost {
  rows: CostRow[];
  params: number;
  /** Parameters at 4 bytes each (float32). */
  paramBytes: number;
  flops: number;
  activationBytes: number;
  partial: boolean;
}

const finite = (n: number) => (Number.isFinite(n) ? n : 0);

function sumRows(rows: CostRow[]): Pick<CostRow, 'params' | 'flops' | 'activationBytes' | 'partial'> {
  return {
    params: rows.reduce((s, r) => s + r.params, 0),
    flops: rows.reduce((s, r) => s + r.flops, 0),
    activationBytes: rows.reduce((s, r) => s + r.activationBytes, 0),
    partial: rows.some((r) => r.partial),
  };
}

/** FLOPs of one call of a spec node; NaN when a shape is unknown. */
function specFlops(spec: NodeSpec, resolved: Resolved | undefined, inputs: Inputs, outputs: (TensorType | undefined)[] | undefined, values: SymbolValues): number {
  if (!spec.flops) return 0;
  const ins = inputs.map((t) => (t ? shapeValue(t.shape, values) : undefined));
  const outs = (outputs ?? []).map((t) => (t ? shapeValue(t.shape, values) : undefined));
  if (!resolved || ins.some((s) => !s) || outs.some((s) => !s) || outs.length === 0) return NaN;
  return spec.flops(resolved, ins as number[][], outs as number[][]);
}

function repeatCount(value: ParamValue, env: Env): number {
  if ('literal' in value) return typeof value.literal === 'number' ? value.literal : NaN;
  if ('expr' in value) {
    const r = evalExpr(value.expr, env);
    return r.ok && typeof r.value === 'number' ? r.value : NaN;
  }
  return NaN;
}

/** A repeated layer: every copy inferred again, since the first may differ and shapes change along the chain. */
function repeatedLayer(spec: NodeSpec, node: NodeInst, info: NodeAnalysis, env: Env, values: SymbolValues): { flops: number; bytes: number } {
  const count = repeatCount(node.repeat!.count, env);
  const bound = bindParams(withDefaults(spec, node.params), env);
  const first = node.repeat!.first ? bindParams({ ...withDefaults(spec, node.params), ...node.repeat!.first }, env) : bound;
  if (!Number.isInteger(count) || !('params' in bound) || !('params' in first)) return { flops: NaN, bytes: NaN };
  let flops = 0;
  let bytes = 0;
  let current = info.inputs;
  for (let i = 0; i < count; i++) {
    const result = spec.infer(current, i === 0 ? first.params : bound.params);
    if (!result.ok) return { flops: NaN, bytes: NaN };
    flops += specFlops(spec, result.resolved, current, result.outputs, values);
    bytes += tensorBytes(result.outputs[0], values);
    current = [result.outputs[0], ...current.slice(1)];
  }
  return { flops, bytes };
}

function rowsOf(project: LoomProject, analysis: Analysis, path: NodeId[], prefix: string, values: SymbolValues, top: boolean): CostRow[] {
  const mod = project.modules[analysis.moduleId];
  if (!mod) return [];
  const rows: CostRow[] = [];
  for (const nodeId of analysis.order) {
    const node = mod.nodes[nodeId];
    const info = analysis.nodes[nodeId];
    if (!node || node.kind === 'output' || (node.kind === 'input' && !top)) continue;
    const name = prefix + node.name;
    const base = { path, moduleId: mod.id, nodeId, name };
    const outputBytes = () => (info?.outputs ?? []).reduce((s, t) => s + tensorBytes(t, values), 0);

    if (node.kind === 'input') {
      const bytes = tensorBytes(info?.outputs?.[0], values);
      rows.push({ ...base, type: 'Input', params: 0, flops: 0, activationBytes: finite(bytes), partial: !Number.isFinite(bytes) });
      continue;
    }
    if (node.kind === 'code') {
      const bytes = info?.outputs ? outputBytes() : NaN;
      rows.push({ ...base, type: 'Python', params: 0, flops: 0, activationBytes: finite(bytes), partial: true });
      continue;
    }
    if (node.kind === 'module') {
      const def = project.modules[node.type];
      const type = def ? (def.sequential ? 'Sequential' : def.className) : node.type;
      const copies = info?.children ?? [];
      if (def?.opaque || copies.length === 0) {
        const bytes = info?.outputs ? outputBytes() : NaN;
        rows.push({ ...base, type, params: info?.paramCount ?? 0, flops: 0, activationBytes: finite(bytes), partial: true });
        continue;
      }
      const inner = [...path, nodeId];
      const children = node.repeat
        ? copies.map((copy, i): CostRow => {
            const kids = rowsOf(project, copy, inner, `${name}.${i}.`, values, false);
            return { ...base, name: `${name}.${i}`, type, ...sumRows(kids), children: kids };
          })
        : rowsOf(project, copies[0]!, inner, `${name}.`, values, false);
      rows.push({ ...base, type, ...sumRows(children), partial: sumRows(children).partial || copies.some((c) => c.partial), children });
      continue;
    }

    const spec = getSpec(node.type);
    if (!spec || !info?.outputs) {
      rows.push({ ...base, type: spec?.displayName.en ?? node.type, params: info?.paramCount ?? 0, flops: 0, activationBytes: 0, partial: true });
      continue;
    }
    const cost = node.repeat
      ? repeatedLayer(spec, node, info, analysis.env, values)
      : // A parameter or buffer is held, not produced by the forward pass (parameters count in paramBytes).
        { flops: specFlops(spec, info.resolved, info.inputs, info.outputs, values), bytes: spec.codegen.state ? 0 : outputBytes() };
    rows.push({
      ...base,
      type: spec.displayName.en,
      params: info.paramCount,
      flops: finite(cost.flops),
      activationBytes: finite(cost.bytes),
      partial: !Number.isFinite(cost.flops) || !Number.isFinite(cost.bytes),
    });
  }
  return rows;
}

/** The cost of a model (the entry analysis), or of one module instance. */
export function costOf(project: LoomProject, analysis: Analysis, values: SymbolValues, path: NodeId[] = []): Cost {
  const rows = rowsOf(project, analysis, path, '', values, true);
  const total = sumRows(rows);
  return { rows, params: total.params, paramBytes: total.params * 4, flops: total.flops, activationBytes: total.activationBytes, partial: total.partial || analysis.partial };
}

/** FLOPs of one node of an analysis (a module node: everything inside), NaN when unknown. */
export function nodeFlops(project: LoomProject, analysis: Analysis, nodeId: NodeId, values: SymbolValues): { flops: number; partial: boolean } | undefined {
  const row = rowsOf(project, analysis, [], '', values, true).find((r) => r.nodeId === nodeId);
  return row && { flops: row.flops, partial: row.partial };
}
