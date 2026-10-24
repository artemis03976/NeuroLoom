import { msg } from '../i18n';
import { dimMul, UNKNOWN, type Dim } from '../shape';
import { fail, feature, IN, L, lit, num, ok, OUT, P, prod, rankError, resolveParams, showTuple, tuple } from './helpers';
import type { Bilingual, Example, NodeSpec } from './types';

export type Length = { dim: Dim } | { error: string };

/** Output length of a convolution (or pooling window) along one axis. */
export function convLength(len: Dim, k: number, s: number, p: number, d: number, ceil = false): Length {
  const span = d * (k - 1);
  if (typeof len === 'number') {
    const round = ceil ? Math.ceil : Math.floor;
    const out = round((len + 2 * p - span - 1) / s) + 1;
    if (out >= 1) return { dim: out };
    return {
      error: msg(
        `输出尺寸小于 1（输入 ${len}，核 ${k}，步长 ${s}，填充 ${p}）`,
        `the output size is less than 1 (input ${len}, kernel ${k}, stride ${s}, padding ${p})`,
      ),
    };
  }
  return { dim: s === 1 && 2 * p === span ? len : UNKNOWN };
}

function transposedLength(len: Dim, k: number, s: number, p: number, d: number, op: number): Dim {
  if (typeof len === 'number') return (len - 1) * s - 2 * p + d * (k - 1) + op + 1;
  // (L - 1)·s + s = L·s when the kernel equals the stride and nothing else is set.
  if (k === s && p === 0 && d === 1 && op === 0) return dimMul(len, s);
  return s === 1 && 2 * p === d * (k - 1) + op ? len : UNKNOWN;
}

/**
 * shapeDoc of a sliding-window layer: `formula` gives the output length along
 * the first spatial axis (L, H or D), and the other axes follow likewise.
 */
export function windowDoc(n: 1 | 2 | 3, formula: (axis: string) => string, note?: Bilingual): Bilingual {
  const [axis, restEn, restZh] = ({ 1: ['L', '', ''], 2: ['H', ' (likewise for W)', '（W 同理）'], 3: ['D', ' (likewise for H and W)', '（H、W 同理）'] } as const)[n];
  const line = `${axis}_out = ${formula(axis)}`;
  return L(`${line}${restEn}${note ? `; ${note.en}` : ''}`, `${line}${restZh}${note ? `；${note.zh}` : ''}`);
}

const EXAMPLES: Record<string, Example> = {
  Conv1d: { inputs: [{ shape: 'B, 4, 32' }], params: { out_channels: 8, kernel_size: 3, padding: 1 } },
  Conv2d: { inputs: [{ shape: 'B, 3, 16, 16' }], params: { out_channels: 8, kernel_size: [3, 5], stride: 2, padding: [1, 2], bias: false } },
  Conv3d: { inputs: [{ shape: 'B, 2, 8, 8, 8' }], params: { out_channels: 4, kernel_size: 3, padding: 'same' } },
  ConvTranspose1d: { inputs: [{ shape: 'B, 4, 10' }], params: { out_channels: 4, kernel_size: 3, stride: 2, padding: 1, output_padding: 1 } },
  ConvTranspose2d: { inputs: [{ shape: 'B, 8, 7, 7' }], params: { out_channels: 4, kernel_size: 2, stride: 2 } },
  ConvTranspose3d: { inputs: [{ shape: 'B, 4, 3, 3, 3' }], params: { out_channels: 2, kernel_size: 2, stride: 2, groups: 2 } },
};

function conv(n: 1 | 2 | 3, transposed: boolean): NodeSpec {
  const cls = `Conv${transposed ? 'Transpose' : ''}${n}d`;
  const ints = (name: string, zh: string, extra: object) => P.ints(name, zh, { arity: n, ...extra });
  const params = [
    P.int('in_channels', '输入通道数', { allowAuto: true, min: 1, initial: 'auto' }),
    P.int('out_channels', '输出通道数', { min: 1, initial: 16 }),
    ints('kernel_size', '卷积核大小', { initial: 3, min: 1 }),
    ints('stride', '步长', { default: 1, min: 1 }),
    transposed ? ints('padding', '填充', { default: 0 }) : ints('padding', '填充', { default: 0, options: ['same', 'valid'] }),
    ...(transposed ? [ints('output_padding', '输出额外填充', { default: 0 })] : []),
    ...(transposed ? [] : [ints('dilation', '空洞率', { default: 1, min: 1 })]),
    P.int('groups', '分组数', { default: 1, min: 1 }),
    P.bool('bias', '偏置', { default: true }),
    ...(transposed
      ? [ints('dilation', '空洞率', { default: 1, min: 1 })]
      : [P.enum('padding_mode', '填充方式', { default: 'zeros', options: ['zeros', 'reflect', 'replicate', 'circular'] })]),
  ];

  return {
    type: `torch.nn.${cls}`,
    kind: 'layer',
    category: 'conv',
    displayName: L(cls, `${n}D ${transposed ? '转置卷积' : '卷积'}`),
    summary: transposed
      ? L(`Upsampling ${n}-D convolution (deconvolution).`, `${n} 维转置卷积，用于上采样。`)
      : L(`${n}-D convolution over an input with channels.`, `对带通道的输入做 ${n} 维卷积。`),
    docUrl: `https://docs.pytorch.org/docs/stable/generated/torch.nn.${cls}.html`,
    params,
    inputs: IN,
    outputs: OUT,
    defaultName: transposed ? 'deconv' : 'conv',
    infer([x], params) {
      const bad = rankError(x!, [n + 2], cls);
      if (bad) return fail(bad);
      const inC = feature(params, 'in_channels', x!.shape[1], msg('输入通道数', 'the number of input channels'));
      if (typeof inC === 'string') return fail(inC);
      const outC = num(params, 'out_channels');
      const groups = num(params, 'groups');
      if (inC % groups !== 0 || outC % groups !== 0) {
        return fail(msg(`in_channels 与 out_channels 都必须能被 groups=${groups} 整除`, `in_channels and out_channels must both be divisible by groups=${groups}`));
      }

      const k = tuple(lit(params, 'kernel_size'), n);
      const s = tuple(lit(params, 'stride'), n);
      const d = tuple(lit(params, 'dilation'), n);
      const padding = lit(params, 'padding');
      if (padding === 'same' && s.some((v) => v !== 1)) return fail(msg('padding="same" 只支持 stride=1', 'padding="same" only supports stride=1'));
      const p = typeof padding === 'string' ? Array<number>(n).fill(0) : tuple(padding, n);
      const op = transposed ? tuple(lit(params, 'output_padding'), n) : [];
      for (const [name, values] of [['kernel_size', k], ['stride', s], ['dilation', d], ['padding', p]] as const) {
        if (values.length !== n) return fail(msg(`${name} 需要 ${n} 个值`, `${name} needs ${n} values`));
      }

      const spatial: Dim[] = [];
      for (let i = 0; i < n; i++) {
        const len = x!.shape[2 + i]!;
        if (padding === 'same') {
          spatial.push(len);
          continue;
        }
        const out = transposed ? { dim: transposedLength(len, k[i]!, s[i]!, p[i]!, d[i]!, op[i]!) } : convLength(len, k[i]!, s[i]!, p[i]!, d[i]!);
        if ('error' in out) return fail(out.error);
        spatial.push(out.dim);
      }
      return ok([{ shape: [x!.shape[0]!, outC, ...spatial], dtype: x!.dtype }], resolveParams(params, { in_channels: inC }));
    },
    paramCount: (r) => {
      const k = tuple(r.kernel_size, n).reduce((a, b) => a * b, 1);
      const inC = r.in_channels as number;
      const outC = r.out_channels as number;
      const g = r.groups as number;
      return (transposed ? inC * (outC / g) : outC * (inC / g)) * k + (r.bias ? outC : 0);
    },
    // FlopCounterMode: 2 · batch · positions · weight elements, the bias not counted.
    // The weight has in_channels · out_channels / groups · kernel elements either way;
    // a convolution slides it over the output positions, a transposed one over the input's.
    flops: (r, [x], [y]) => {
      const weight = ((r.in_channels as number) * (r.out_channels as number) * prod(tuple(r.kernel_size, n))) / (r.groups as number);
      return 2 * x![0]! * prod((transposed ? x! : y!).slice(2)) * weight;
    },
    shapeDoc: windowDoc(
      n,
      transposed
        ? (a) => `(${a} − 1)·stride − 2·padding + dilation·(kernel_size − 1) + output_padding + 1`
        : (a) => `⌊(${a} + 2·padding − dilation·(kernel_size − 1) − 1) / stride⌋ + 1`,
      L('channels become out_channels', '通道数变为 out_channels'),
    ),
    describe: (r) => {
      const stride = showTuple(r.stride);
      return `${r.in_channels ?? '?'} → ${r.out_channels} · k${showTuple(r.kernel_size)}${stride !== '1' ? ` · s${stride}` : ''}`;
    },
    codegen: { ctor: `nn.${cls}` },
    example: EXAMPLES[cls],
  };
}

export const CONV_SPECS: NodeSpec[] = [conv(1, false), conv(2, false), conv(3, false), conv(1, true), conv(2, true), conv(3, true)];
