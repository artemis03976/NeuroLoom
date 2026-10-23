// Shape-changing tensor operations (PRD appendix: 张量形变).
import { axis, dimAdd, dimDiv, dimEq, dimProduct, UNKNOWN, type Dim, type TensorType } from '../shape';
import { fail, IN, L, lit, num, ok, OUT, P, resolveParams } from './helpers';
import type { Inputs, NodeSpec, Params, PythonCall } from './types';

const doc = (path: string) => `https://docs.pytorch.org/docs/stable/generated/${path}.html`;

function flattenShape(shape: Dim[], start: number, end: number): Dim[] | string {
  const s = axis(start, shape.length);
  const e = axis(end, shape.length);
  if (s === undefined || e === undefined) return `start_dim/end_dim 超出 ${shape.length} 维输入的范围`;
  if (s > e) return 'start_dim 不能在 end_dim 之后';
  return [...shape.slice(0, s), dimProduct(shape.slice(s, e + 1)), ...shape.slice(e + 1)];
}

/** Resolves a reshape target, including one -1. */
function reshapeShape(input: Dim[], target: number[]): Dim[] | string {
  const holes = target.filter((d) => d === -1).length;
  if (holes > 1) return '形状中最多只能有一个 -1';
  if (target.some((d) => d === 0 || d < -1)) return '目标形状中的维度必须为正数或 -1';
  const total = dimProduct(input);
  const known = target.filter((d) => d !== -1).reduce((a, b) => a * b, 1);
  if (holes === 0) {
    return dimEq(total, known) === false ? `元素总数 ${total} 无法重排为 [${target.join(', ')}]` : target;
  }
  const missing = dimDiv(total, known);
  if (missing === undefined) {
    return typeof total === 'number' ? `元素总数 ${total} 不能被 ${known} 整除` : target.map((d) => (d === -1 ? UNKNOWN : d));
  }
  return target.map((d) => (d === -1 ? missing : d));
}

function method(
  type: string,
  name: string,
  zh: string,
  summary: [string, string],
  params: NodeSpec['params'],
  infer: (x: TensorType, params: Params) => Dim[] | string,
  options: { methods?: string[]; functions?: string[]; example: NonNullable<NodeSpec['example']>; describe?: NodeSpec['describe'] },
): NodeSpec {
  const python: PythonCall = {
    methods: options.methods ?? [name],
    functions: options.functions,
    args: [{ name: 'input', port: 'in' }, ...params.map((p) => ({ name: p.name, param: p.name }))],
  };
  return {
    type,
    kind: 'op',
    category: 'tensor',
    displayName: L(`.${name}`, zh),
    summary: L(...summary),
    docUrl: doc(`torch.Tensor.${name}`),
    params,
    inputs: IN,
    outputs: OUT,
    defaultName: name,
    infer([x]: Inputs, params: Params) {
      const shape = infer(x!, params);
      return typeof shape === 'string' ? fail(shape) : ok([{ shape, dtype: x!.dtype }], resolveParams(params));
    },
    describe: options.describe,
    codegen: { method: name },
    python,
    example: options.example,
  };
}

const flattenModule: NodeSpec = {
  type: 'torch.nn.Flatten',
  kind: 'layer',
  category: 'tensor',
  displayName: L('Flatten', '展平层'),
  summary: L('Merges a range of dimensions into one (keeps the batch by default).', '把一段连续维度合并为一维（默认保留批维度）。'),
  docUrl: doc('torch.nn.Flatten'),
  params: [P.int('start_dim', '起始维', { default: 1 }), P.int('end_dim', '结束维', { default: -1 })],
  inputs: IN,
  outputs: OUT,
  defaultName: 'flatten',
  infer([x], params) {
    const shape = flattenShape(x!.shape, num(params, 'start_dim'), num(params, 'end_dim'));
    return typeof shape === 'string' ? fail(shape) : ok([{ shape, dtype: x!.dtype }], resolveParams(params));
  },
  codegen: { ctor: 'nn.Flatten' },
  example: { inputs: [{ shape: 'B, 2, 3, 4' }], params: { end_dim: 2 } },
};

const unflattenModule: NodeSpec = {
  type: 'torch.nn.Unflatten',
  kind: 'layer',
  category: 'tensor',
  displayName: L('Unflatten', '反展平层'),
  summary: L('Splits one dimension into several.', '把一维拆成多维。'),
  docUrl: doc('torch.nn.Unflatten'),
  params: [P.int('dim', '要拆分的维', { initial: 1 }), P.dims('unflattened_size', '拆分后的形状', { initial: [1, 28, 28] })],
  inputs: IN,
  outputs: OUT,
  defaultName: 'unflatten',
  infer([x], params) {
    const d = axis(num(params, 'dim'), x!.shape.length);
    if (d === undefined) return fail(`dim 超出 ${x!.shape.length} 维输入的范围`);
    const sizes = reshapeShape([x!.shape[d]!], lit(params, 'unflattened_size') as number[]);
    if (typeof sizes === 'string') return fail(sizes);
    return ok([{ shape: [...x!.shape.slice(0, d), ...sizes, ...x!.shape.slice(d + 1)], dtype: x!.dtype }], resolveParams(params));
  },
  codegen: { ctor: 'nn.Unflatten' },
  example: { inputs: [{ shape: 'B, 6, 4' }], params: { dim: 1, unflattened_size: [2, 3] } },
};

const flattenOp: NodeSpec = {
  type: 'torch.flatten',
  kind: 'op',
  category: 'tensor',
  displayName: L('torch.flatten', 'flatten 函数'),
  summary: L('Merges a range of dimensions into one.', '把一段连续维度合并为一维。'),
  docUrl: doc('torch.flatten'),
  params: [P.int('start_dim', '起始维', { default: 0, positional: true }), P.int('end_dim', '结束维', { default: -1, positional: true })],
  inputs: IN,
  outputs: OUT,
  defaultName: 'flatten',
  infer: flattenModule.infer,
  codegen: { fn: 'torch.flatten' },
  python: {
    functions: ['torch.flatten'],
    methods: ['flatten'],
    args: [
      { name: 'input', port: 'in' },
      { name: 'start_dim', param: 'start_dim' },
      { name: 'end_dim', param: 'end_dim' },
    ],
  },
  example: { inputs: [{ shape: 'B, 2, 3, 4' }], params: { start_dim: 1 } },
};

const reshape = method(
  'torch.Tensor.reshape',
  'reshape',
  '重排形状',
  ['Changes the shape without changing the data; -1 is inferred.', '在不改变数据的前提下改变形状；-1 由其余维推断。'],
  [P.dims('shape', '目标形状', { initial: [-1], spread: true })],
  (x, params) => reshapeShape(x.shape, lit(params, 'shape') as number[]),
  { methods: ['reshape', 'view'], functions: ['torch.reshape'], example: { inputs: [{ shape: 'B, 4, 6' }], params: { shape: [-1, 24] } } },
);

const permute = method(
  'torch.Tensor.permute',
  'permute',
  '交换维度顺序',
  ['Reorders dimensions.', '按给定顺序重排维度。'],
  [P.dims('dims', '新的维度顺序', { initial: [0, 2, 1], spread: true })],
  (x, params) => {
    const dims = lit(params, 'dims') as number[];
    if (dims.length !== x.shape.length) return `需要 ${x.shape.length} 个维度编号，收到 ${dims.length} 个`;
    const order = dims.map((d) => axis(d, x.shape.length));
    if (order.some((d) => d === undefined) || new Set(order).size !== order.length) return 'dims 必须是各维度编号的一个排列';
    return order.map((d) => x.shape[d!]!);
  },
  { functions: ['torch.permute'], example: { inputs: [{ shape: 'B, 3, 5' }], params: { dims: [0, 2, 1] } } },
);

const transpose = method(
  'torch.Tensor.transpose',
  'transpose',
  '交换两维',
  ['Swaps two dimensions.', '交换两个维度。'],
  [P.int('dim0', '第一维', { initial: 1 }), P.int('dim1', '第二维', { initial: 2 })],
  (x, params) => {
    const a = axis(num(params, 'dim0'), x.shape.length);
    const b = axis(num(params, 'dim1'), x.shape.length);
    if (a === undefined || b === undefined) return `dim0/dim1 超出 ${x.shape.length} 维输入的范围`;
    const shape = [...x.shape];
    [shape[a], shape[b]] = [shape[b]!, shape[a]!];
    return shape;
  },
  { functions: ['torch.transpose'], example: { inputs: [{ shape: 'B, 3, 5' }], params: { dim0: 1, dim1: 2 } } },
);

const unsqueeze = method(
  'torch.Tensor.unsqueeze',
  'unsqueeze',
  '插入大小为 1 的维',
  ['Inserts a dimension of size 1.', '插入一个大小为 1 的维度。'],
  [P.int('dim', '插入位置', { initial: 1 })],
  (x, params) => {
    const d = axis(num(params, 'dim'), x.shape.length + 1);
    return d === undefined ? 'dim 超出范围' : [...x.shape.slice(0, d), 1, ...x.shape.slice(d)];
  },
  { functions: ['torch.unsqueeze'], example: { inputs: [{ shape: 'B, 5' }], params: { dim: 1 } } },
);

const squeeze = method(
  'torch.Tensor.squeeze',
  'squeeze',
  '去掉大小为 1 的维',
  ['Removes dimensions of size 1 (all of them, or only `dim`).', '去掉大小为 1 的维度（全部，或只去掉 dim）。'],
  [P.int('dim', '维度（空 = 全部）', { nullable: true, default: null, positional: true })],
  (x, params) => {
    const dim = lit(params, 'dim');
    if (dim === null || dim === undefined) return x.shape.filter((d) => d !== 1);
    const d = axis(dim as number, x.shape.length);
    if (d === undefined) return 'dim 超出范围';
    return x.shape[d] === 1 ? [...x.shape.slice(0, d), ...x.shape.slice(d + 1)] : x.shape;
  },
  { functions: ['torch.squeeze'], example: { inputs: [{ shape: 'B, 1, 5' }], params: { dim: 1 } } },
);

const contiguous = method(
  'torch.Tensor.contiguous',
  'contiguous',
  '连续内存',
  ['Returns a tensor with contiguous memory; the shape is unchanged.', '返回内存连续的张量，形状不变。'],
  [],
  (x) => x.shape,
  { example: { inputs: [{ shape: 'B, 4' }] } },
);

function joinOp(name: 'cat' | 'stack'): NodeSpec {
  return {
    type: `torch.${name}`,
    kind: 'op',
    category: 'tensor',
    displayName: L(`torch.${name}`, name === 'cat' ? '拼接' : '堆叠'),
    summary:
      name === 'cat'
        ? L('Concatenates tensors along an existing dimension.', '沿已有维度拼接多个张量。')
        : L('Stacks tensors along a new dimension.', '沿新维度堆叠多个张量。'),
    docUrl: doc(`torch.${name}`),
    params: [P.int('dim', '维度', { default: 0 })],
    inputs: [{ name: 'in', variadic: true }],
    outputs: OUT,
    defaultName: name,
    infer(inputs, params) {
      const tensors = inputs.filter((t): t is TensorType => t !== undefined);
      if (tensors.length === 0) return fail('至少需要连接一个输入');
      const first = tensors[0]!;
      const rank = first.shape.length;
      if (tensors.some((t) => t.shape.length !== rank)) return fail('所有输入的维数必须相同');
      const d = axis(num(params, 'dim'), name === 'cat' ? rank : rank + 1);
      if (d === undefined) return fail('dim 超出范围');
      const shape: Dim[] = [];
      for (let i = 0; i < rank; i++) {
        if (name === 'cat' && i === d) {
          shape.push(tensors.map((t) => t.shape[i]!).reduce(dimAdd));
          continue;
        }
        const sizes = tensors.map((t) => t.shape[i]!);
        if (sizes.some((s) => dimEq(s, sizes[0]!) === false)) {
          return fail(`除拼接维外，各输入第 ${i} 维必须相同（${sizes.join('、')}）`);
        }
        shape.push(sizes[0]!);
      }
      if (name === 'stack') shape.splice(d, 0, tensors.length);
      return ok([{ shape, dtype: first.dtype }], resolveParams(params));
    },
    describe: (r) => `dim = ${r.dim}`,
    codegen: { fn: `torch.${name}` },
    python: {
      functions: name === 'cat' ? ['torch.cat', 'torch.concat'] : ['torch.stack'],
      args: [
        { name: 'tensors', ports: 'in' },
        { name: 'dim', param: 'dim' },
      ],
    },
    example:
      name === 'cat'
        ? { inputs: [{ shape: 'B, 3, 8' }, { shape: 'B, 5, 8' }], params: { dim: 1 } }
        : { inputs: [{ shape: 'B, 4' }, { shape: 'B, 4' }], params: { dim: 1 } },
  };
}

const chunk: NodeSpec = {
  type: 'torch.chunk',
  kind: 'op',
  category: 'tensor',
  displayName: L('torch.chunk', '等分'),
  summary: L('Splits a tensor into equal chunks along a dimension.', '沿某一维把张量等分成若干块。'),
  docUrl: doc('torch.chunk'),
  params: [P.int('chunks', '块数', { initial: 2, min: 1 }), P.int('dim', '维度', { default: 0 })],
  inputs: IN,
  outputs: [],
  dynamicOutputs: { param: 'chunks', prefix: 'out' },
  defaultName: 'chunk',
  infer([x], params) {
    const n = num(params, 'chunks');
    const d = axis(num(params, 'dim'), x!.shape.length);
    if (d === undefined) return fail('dim 超出范围');
    const len = x!.shape[d]!;
    if (typeof len === 'number' && len % n !== 0) return fail(`第 ${d} 维大小 ${len} 不能被 ${n} 整除（M1 只支持等分）`);
    const piece = typeof len === 'number' ? len / n : (dimDiv(len, n) ?? UNKNOWN);
    const shape = x!.shape.map((s, i) => (i === d ? piece : s));
    return ok(Array.from({ length: n }, () => ({ shape, dtype: x!.dtype })), resolveParams(params));
  },
  describe: (r) => `${r.chunks} 块 · dim = ${r.dim}`,
  codegen: { fn: 'torch.chunk' },
  python: {
    functions: ['torch.chunk'],
    methods: ['chunk'],
    args: [
      { name: 'input', port: 'in' },
      { name: 'chunks', param: 'chunks' },
      { name: 'dim', param: 'dim' },
    ],
  },
  example: { inputs: [{ shape: 'B, 6, 4' }], params: { chunks: 3, dim: 1 } },
};

export const TENSOR_SPECS: NodeSpec[] = [
  flattenModule,
  unflattenModule,
  flattenOp,
  reshape,
  permute,
  transpose,
  unsqueeze,
  squeeze,
  contiguous,
  joinOp('cat'),
  joinOp('stack'),
  chunk,
];
