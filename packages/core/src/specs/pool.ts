import { msg } from '../i18n';
import { dimDiv, UNKNOWN, type Dim } from '../shape';
import { convLength, windowDoc } from './conv';
import { fail, flag, IN, L, lit, ok, OUT, P, rankError, resolveParams, showTuple, tuple } from './helpers';
import type { Example, NodeSpec, ParamSpec } from './types';

const EXAMPLES: Record<string, Example> = {
  MaxPool1d: { inputs: [{ shape: 'B, 4, 20' }], params: { kernel_size: 3, stride: 2, padding: 1 } },
  MaxPool2d: { inputs: [{ shape: 'B, 4, 16, 16' }], params: { kernel_size: 2 } },
  MaxPool3d: { inputs: [{ shape: 'B, 2, 8, 8, 8' }], params: { kernel_size: 2, ceil_mode: true } },
  AvgPool1d: { inputs: [{ shape: 'B, 4, 20' }], params: { kernel_size: 2 } },
  AvgPool2d: { inputs: [{ shape: 'B, 4, 15, 15' }], params: { kernel_size: 3, stride: 2, padding: 1, count_include_pad: false } },
  AvgPool3d: { inputs: [{ shape: 'B, 2, 8, 8, 8' }], params: { kernel_size: 2 } },
  AdaptiveAvgPool1d: { inputs: [{ shape: 'B, 4, 20' }], params: { output_size: 8 } },
  AdaptiveAvgPool2d: { inputs: [{ shape: 'B, 4, 7, 7' }], params: { output_size: 1 } },
  AdaptiveAvgPool3d: { inputs: [{ shape: 'B, 2, 6, 6, 6' }], params: { output_size: 2 } },
  AdaptiveMaxPool1d: { inputs: [{ shape: 'B, 4, 20' }], params: { output_size: 4 } },
  AdaptiveMaxPool2d: { inputs: [{ shape: 'B, 4, 9, 9' }], params: { output_size: [2, 3] } },
};

function pool(n: 1 | 2 | 3, kind: 'Max' | 'Avg'): NodeSpec {
  const cls = `${kind}Pool${n}d`;
  const ints = (name: string, zh: string, extra: Partial<ParamSpec>) => P.ints(name, zh, { arity: n, ...extra });
  const params = [
    ints('kernel_size', '窗口大小', { initial: 2, min: 1 }),
    ints('stride', '步长（空 = 窗口大小）', { nullable: true, default: null, min: 1 }),
    ints('padding', '填充', { default: 0 }),
    ...(kind === 'Max' ? [ints('dilation', '空洞率', { default: 1, min: 1 })] : []),
    P.bool('ceil_mode', '向上取整', { default: false }),
    ...(kind === 'Avg' ? [P.bool('count_include_pad', '平均时计入填充', { default: true })] : []),
  ];
  return {
    type: `torch.nn.${cls}`,
    kind: 'layer',
    category: 'pool',
    displayName: L(cls, `${n}D ${kind === 'Max' ? '最大' : '平均'}池化`),
    summary: L(`${kind === 'Max' ? 'Max' : 'Average'} over sliding ${n}-D windows.`, `在 ${n} 维滑动窗口上取${kind === 'Max' ? '最大值' : '平均值'}。`),
    docUrl: `https://docs.pytorch.org/docs/stable/generated/torch.nn.${cls}.html`,
    params,
    inputs: IN,
    outputs: OUT,
    defaultName: 'pool',
    infer([x], params) {
      const bad = rankError(x!, [n + 2, n + 1], cls);
      if (bad) return fail(bad);
      const k = tuple(lit(params, 'kernel_size'), n);
      const strideValue = lit(params, 'stride');
      const s = strideValue === null || strideValue === undefined ? k : tuple(strideValue, n);
      const p = tuple(lit(params, 'padding'), n);
      const d = kind === 'Max' ? tuple(lit(params, 'dilation'), n) : Array<number>(n).fill(1);
      if (p.some((v, i) => v > k[i]! / 2)) return fail(msg('padding 不能超过窗口大小的一半', 'padding must be at most half of kernel_size'));
      const lead = x!.shape.length - n;
      const spatial: Dim[] = [];
      for (let i = 0; i < n; i++) {
        const len = x!.shape[lead + i]!;
        // A symbolic length pooled by non-overlapping windows divides exactly when possible.
        if (typeof len !== 'number' && k[i] === s[i] && p[i] === 0 && d[i] === 1) {
          spatial.push(dimDiv(len, k[i]!) ?? UNKNOWN);
          continue;
        }
        const out = convLength(len, k[i]!, s[i]!, p[i]!, d[i]!, flag(params, 'ceil_mode'));
        if ('error' in out) return fail(out.error);
        spatial.push(out.dim);
      }
      return ok([{ shape: [...x!.shape.slice(0, lead), ...spatial], dtype: x!.dtype }], resolveParams(params));
    },
    shapeDoc: windowDoc(
      n,
      kind === 'Max'
        ? (a) => `⌊(${a} + 2·padding − dilation·(kernel_size − 1) − 1) / stride⌋ + 1`
        : (a) => `⌊(${a} + 2·padding − kernel_size) / stride⌋ + 1`,
      L('stride defaults to kernel_size; ceil_mode rounds up', 'stride 默认等于 kernel_size；ceil_mode 时向上取整'),
    ),
    describe: (r) => `k${showTuple(r.kernel_size)}${r.stride !== null && showTuple(r.stride) !== showTuple(r.kernel_size) ? ` · s${showTuple(r.stride)}` : ''}`,
    codegen: { ctor: `nn.${cls}` },
    example: EXAMPLES[cls],
  };
}

function adaptive(n: 1 | 2 | 3, kind: 'Max' | 'Avg'): NodeSpec {
  const cls = `Adaptive${kind}Pool${n}d`;
  return {
    type: `torch.nn.${cls}`,
    kind: 'layer',
    category: 'pool',
    displayName: L(cls, `${n}D 自适应${kind === 'Max' ? '最大' : '平均'}池化`),
    summary: L(`Pools to a fixed output size regardless of input size.`, '无论输入多大，都池化到固定的输出尺寸。'),
    docUrl: `https://docs.pytorch.org/docs/stable/generated/torch.nn.${cls}.html`,
    params: [P.ints('output_size', '输出尺寸', { arity: n, initial: 1, min: 1 })],
    inputs: IN,
    outputs: OUT,
    defaultName: 'pool',
    infer([x], params) {
      const bad = rankError(x!, [n + 2, n + 1], cls);
      if (bad) return fail(bad);
      const size = tuple(lit(params, 'output_size'), n);
      const lead = x!.shape.length - n;
      return ok([{ shape: [...x!.shape.slice(0, lead), ...size], dtype: x!.dtype }], resolveParams(params));
    },
    shapeDoc: L(
      `The last ${n === 1 ? 'dimension becomes' : `${n} dimensions become`} output_size, whatever the input size`,
      `${n === 1 ? '最后一维' : `最后 ${n} 维`}变为 output_size，与输入大小无关`,
    ),
    describe: (r) => `→ ${showTuple(r.output_size)}`,
    codegen: { ctor: `nn.${cls}` },
    example: EXAMPLES[cls],
  };
}

/** The function form of a pooling layer (F.max_pool2d(x, 2)): same parameters, in the function's argument order. */
function functional(layer: NodeSpec, name: string): NodeSpec {
  const { ctor: _, ...codegen } = layer.codegen;
  return {
    ...layer,
    type: `torch.nn.functional.${name}`,
    kind: 'op',
    displayName: L(`F.${name}`, `${layer.displayName.zh}函数`),
    summary: L(`${layer.displayName.en} as a function call.`, `函数形式的 ${layer.displayName.en}。`),
    docUrl: `https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.${name}.html`,
    defaultName: name,
    codegen: { ...codegen, fn: `F.${name}` },
    python: {
      functions: [`torch.nn.functional.${name}`],
      args: [{ name: 'input', port: 'in' }, ...layer.params.map((p) => ({ name: p.name, param: p.name }))],
    },
  };
}

const LAYERS = {
  MaxPool1d: pool(1, 'Max'),
  MaxPool2d: pool(2, 'Max'),
  AvgPool1d: pool(1, 'Avg'),
  AvgPool2d: pool(2, 'Avg'),
  AdaptiveAvgPool1d: adaptive(1, 'Avg'),
  AdaptiveAvgPool2d: adaptive(2, 'Avg'),
  AdaptiveMaxPool2d: adaptive(2, 'Max'),
};

export const POOL_SPECS: NodeSpec[] = [
  LAYERS.MaxPool1d,
  LAYERS.MaxPool2d,
  pool(3, 'Max'),
  LAYERS.AvgPool1d,
  LAYERS.AvgPool2d,
  pool(3, 'Avg'),
  LAYERS.AdaptiveAvgPool1d,
  LAYERS.AdaptiveAvgPool2d,
  adaptive(3, 'Avg'),
  adaptive(1, 'Max'),
  LAYERS.AdaptiveMaxPool2d,
  functional(LAYERS.MaxPool1d, 'max_pool1d'),
  functional(LAYERS.MaxPool2d, 'max_pool2d'),
  functional(LAYERS.AvgPool1d, 'avg_pool1d'),
  functional(LAYERS.AvgPool2d, 'avg_pool2d'),
  functional(LAYERS.AdaptiveAvgPool1d, 'adaptive_avg_pool1d'),
  functional(LAYERS.AdaptiveAvgPool2d, 'adaptive_avg_pool2d'),
  functional(LAYERS.AdaptiveMaxPool2d, 'adaptive_max_pool2d'),
];
