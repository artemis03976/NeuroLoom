// NodeSpec: the declarative definition of a node type (PRD section 16.2).
// One spec drives the palette, the inspector form, shape inference, parameter
// counting, code generation and import recognition.
import { isAuto, literalOf, type Literal, type NodeKind, type ParamValue } from './ir';
import { parseShape, shapeText, type TensorType } from './shape';

export type Category =
  | 'io'
  | 'linear'
  | 'conv'
  | 'pool'
  | 'norm'
  | 'activation'
  | 'dropout'
  | 'recurrent'
  | 'attention'
  | 'tensor'
  | 'math'
  | 'module'
  | 'code';

export interface Bilingual {
  en: string;
  zh: string;
}

export interface ParamSpec {
  name: string;
  type: 'int' | 'float' | 'bool' | 'shape';
  /** Absent means the parameter is required (a positional constructor argument). */
  default?: Literal;
  /** Value used when a node is first created from the palette. */
  initial?: Literal | 'auto';
  allowAuto?: boolean;
  min?: number;
  label: Bilingual;
}

export interface PortSpec {
  name: string;
}

export type Resolved = Record<string, Literal>;

export type InferResult = { ok: true; outputs: TensorType[]; resolved: Resolved } | { ok: false; error: string };

export interface CodegenSpec {
  /** Constructor for layer nodes, e.g. "nn.Linear". */
  ctor?: string;
  /** Callable for op nodes, e.g. "F.relu". */
  fn?: string;
}

export interface NodeSpec {
  type: string;
  kind: NodeKind;
  category: Category;
  displayName: Bilingual;
  summary: Bilingual;
  docUrl?: string;
  params: ParamSpec[];
  inputs: PortSpec[];
  outputs: PortSpec[];
  /** Base for generated variable names, e.g. "fc" gives fc1, fc2, ... */
  defaultName: string;
  infer(inputs: TensorType[], params: Record<string, ParamValue>): InferResult;
  paramCount?(resolved: Resolved): number;
  /** One-line summary shown on the node body. */
  describe?(resolved: Resolved): string;
  codegen: CodegenSpec;
}

const IN = [{ name: 'in' }];
const OUT = [{ name: 'out' }];

const input: NodeSpec = {
  type: 'neuroloom.Input',
  kind: 'input',
  category: 'io',
  displayName: { en: 'Input', zh: '输入' },
  summary: { en: 'A forward() argument with a declared example shape.', zh: 'forward() 的一个参数，需声明示例形状。' },
  params: [{ name: 'shape', type: 'shape', initial: 'B, 784', label: { en: 'shape', zh: '形状' } }],
  inputs: [],
  outputs: OUT,
  defaultName: 'x',
  infer(_inputs, params) {
    const parsed = parseShape(String(literalOf(params.shape) ?? ''));
    if (!parsed.ok) return parsed;
    return {
      ok: true,
      outputs: [{ shape: parsed.shape, dtype: 'float32' }],
      resolved: { shape: shapeText(parsed.shape) },
    };
  },
  describe: (r) => `[${r.shape ?? '?'}]`,
  codegen: {},
};

const output: NodeSpec = {
  type: 'neuroloom.Output',
  kind: 'output',
  category: 'io',
  displayName: { en: 'Output', zh: '输出' },
  summary: { en: 'A value returned from forward().', zh: 'forward() 的一个返回值。' },
  params: [],
  inputs: IN,
  outputs: [],
  defaultName: 'output',
  // Output has no output ports; the input type is recorded so the UI can show it.
  infer: ([x]) => ({ ok: true, outputs: [x!], resolved: {} }),
  codegen: {},
};

const linear: NodeSpec = {
  type: 'torch.nn.Linear',
  kind: 'layer',
  category: 'linear',
  displayName: { en: 'Linear', zh: '全连接' },
  summary: { en: 'Applies y = xAᵀ + b to the last dimension.', zh: '对最后一维做线性变换 y = xAᵀ + b。' },
  docUrl: 'https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html',
  params: [
    { name: 'in_features', type: 'int', allowAuto: true, min: 1, initial: 'auto', label: { en: 'in_features', zh: '输入特征数' } },
    { name: 'out_features', type: 'int', min: 1, initial: 64, label: { en: 'out_features', zh: '输出特征数' } },
    { name: 'bias', type: 'bool', default: true, label: { en: 'bias', zh: '偏置' } },
  ],
  inputs: IN,
  outputs: OUT,
  defaultName: 'fc',
  infer([x], params) {
    const last = x!.shape.at(-1);
    if (last === undefined) return { ok: false, error: 'Linear 的输入至少需要 1 维' };
    let inFeatures: number;
    if (isAuto(params.in_features)) {
      if (typeof last !== 'number') return { ok: false, error: `无法从符号维度 ${last} 推断 in_features，请填写具体数值` };
      inFeatures = last;
    } else {
      inFeatures = literalOf(params.in_features) as number;
      if (typeof last === 'number' && last !== inFeatures) {
        return { ok: false, error: `输入最后一维为 ${last}，但 in_features = ${inFeatures}` };
      }
    }
    const outFeatures = literalOf(params.out_features) as number;
    return {
      ok: true,
      outputs: [{ shape: [...x!.shape.slice(0, -1), outFeatures], dtype: x!.dtype }],
      resolved: { in_features: inFeatures, out_features: outFeatures, bias: literalOf(params.bias) as boolean },
    };
  },
  paramCount: (r) => (r.in_features as number) * (r.out_features as number) + (r.bias ? (r.out_features as number) : 0),
  describe: (r) => `${r.in_features ?? '?'} → ${r.out_features ?? '?'}`,
  codegen: { ctor: 'nn.Linear' },
};

/** Element-wise activations keep the input type unchanged. */
const elementwise = (inputs: TensorType[], params: Record<string, ParamValue>): InferResult => ({
  ok: true,
  outputs: [inputs[0]!],
  resolved: { inplace: literalOf(params.inplace) as boolean },
});

const relu: NodeSpec = {
  type: 'torch.nn.ReLU',
  kind: 'layer',
  category: 'activation',
  displayName: { en: 'ReLU', zh: 'ReLU 激活' },
  summary: { en: 'Element-wise max(0, x), as an nn.Module.', zh: '逐元素 max(0, x)，以 nn.Module 形式。' },
  docUrl: 'https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html',
  params: [{ name: 'inplace', type: 'bool', default: false, label: { en: 'inplace', zh: '原地操作' } }],
  inputs: IN,
  outputs: OUT,
  defaultName: 'act',
  infer: elementwise,
  codegen: { ctor: 'nn.ReLU' },
};

const reluFn: NodeSpec = {
  type: 'torch.nn.functional.relu',
  kind: 'op',
  category: 'activation',
  displayName: { en: 'F.relu', zh: 'relu 函数' },
  summary: { en: 'Element-wise max(0, x), as a function call.', zh: '逐元素 max(0, x)，以函数调用形式。' },
  docUrl: 'https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html',
  params: [{ name: 'inplace', type: 'bool', default: false, label: { en: 'inplace', zh: '原地操作' } }],
  inputs: IN,
  outputs: OUT,
  defaultName: 'relu',
  infer: elementwise,
  codegen: { fn: 'F.relu' },
};

export const BUILTIN_SPECS: NodeSpec[] = [input, output, linear, relu, reluFn];

const registry = new Map(BUILTIN_SPECS.map((s) => [s.type, s]));

export function getSpec(type: string | undefined): NodeSpec | undefined {
  return type === undefined ? undefined : registry.get(type);
}

export function allSpecs(): NodeSpec[] {
  return [...registry.values()];
}

/** Parameter values for a freshly created node. */
export function initialParams(spec: NodeSpec): Record<string, ParamValue> {
  const params: Record<string, ParamValue> = {};
  for (const p of spec.params) {
    if (p.initial === 'auto') params[p.name] = { auto: true };
    else if (p.initial !== undefined) params[p.name] = { literal: p.initial };
    else if (p.default !== undefined) params[p.name] = { literal: p.default };
  }
  return params;
}
