import { isAuto } from '../ir';
import { dimEq } from '../shape';
import { fail, feature, IN, L, lit, num, ok, OUT, P, rankError, resolveParams, tuple } from './helpers';
import type { Example, NodeSpec } from './types';

const doc = (cls: string) => `https://docs.pytorch.org/docs/stable/generated/torch.nn.${cls}.html`;
const eps = () => P.float('eps', '数值稳定项 ε', { default: 1e-5 });

function channelNorm(cls: string, zh: string, ranks: number[], kind: 'batch' | 'instance', example: Example): NodeSpec {
  const isBatch = kind === 'batch';
  return {
    type: `torch.nn.${cls}`,
    kind: 'layer',
    category: 'norm',
    displayName: L(cls, zh),
    summary: isBatch
      ? L('Normalizes each channel over the batch.', '在批维度上对每个通道做归一化。')
      : L('Normalizes each channel of each sample separately.', '对每个样本的每个通道分别归一化。'),
    docUrl: doc(cls),
    params: [
      P.int('num_features', '通道数', { allowAuto: true, min: 1, initial: 'auto' }),
      eps(),
      P.float('momentum', '动量', { default: 0.1, nullable: true }),
      P.bool('affine', '可学习缩放与偏移', { default: isBatch }),
      P.bool('track_running_stats', '记录滑动统计量', { default: isBatch }),
    ],
    inputs: IN,
    outputs: OUT,
    defaultName: isBatch ? 'bn' : 'norm',
    infer([x], params) {
      const bad = rankError(x!, ranks, cls);
      if (bad) return fail(bad);
      // Unbatched instance-norm input has the channel first.
      const channelAxis = !isBatch && x!.shape.length === ranks[0]! - 1 ? 0 : 1;
      const c = feature(params, 'num_features', x!.shape[channelAxis], '通道数');
      if (typeof c === 'string') return fail(c);
      return ok([x!], resolveParams(params, { num_features: c }));
    },
    paramCount: (r) => (r.affine ? 2 * (r.num_features as number) : 0),
    describe: (r) => `${r.num_features ?? '?'} 通道`,
    codegen: { ctor: `nn.${cls}` },
    example,
  };
}

/** Shared by LayerNorm and RMSNorm: normalizes over the trailing `normalized_shape` dims. */
function trailing(params: Parameters<NodeSpec['infer']>[1], shape: (number | string)[]): number[] | string {
  if (isAuto(params.normalized_shape)) {
    const last = shape.at(-1);
    return typeof last === 'number' ? [last] : `无法从符号维度 ${last} 推断 normalized_shape，请填写具体数值`;
  }
  const ns = tuple(lit(params, 'normalized_shape'), 1);
  if (ns.length > shape.length) return 'normalized_shape 的维数比输入还多';
  for (let i = 1; i <= ns.length; i++) {
    if (dimEq(shape.at(-i)!, ns.at(-i)!) === false) return `normalized_shape ${JSON.stringify(ns)} 与输入末尾的维度 [${shape.slice(-ns.length).join(', ')}] 不一致`;
  }
  return ns;
}

const layerNorm: NodeSpec = {
  type: 'torch.nn.LayerNorm',
  kind: 'layer',
  category: 'norm',
  displayName: L('LayerNorm', '层归一化'),
  summary: L('Normalizes over the trailing dimensions of each sample.', '对每个样本的末尾若干维做归一化。'),
  docUrl: doc('LayerNorm'),
  params: [
    P.ints('normalized_shape', '归一化的形状', { allowAuto: true, initial: 'auto', min: 1 }),
    eps(),
    P.bool('elementwise_affine', '可学习缩放', { default: true }),
    P.bool('bias', '可学习偏移', { default: true }),
  ],
  inputs: IN,
  outputs: OUT,
  defaultName: 'norm',
  infer([x], params) {
    const ns = trailing(params, x!.shape);
    if (typeof ns === 'string') return fail(ns);
    return ok([x!], resolveParams(params, { normalized_shape: ns.length === 1 ? ns[0]! : ns }));
  },
  paramCount: (r) => {
    const n = tuple(r.normalized_shape, 1).reduce((a, b) => a * b, 1);
    return r.elementwise_affine ? n * (r.bias ? 2 : 1) : 0;
  },
  describe: (r) => `[${tuple(r.normalized_shape, 1).join(', ')}]`,
  codegen: { ctor: 'nn.LayerNorm' },
  example: { inputs: [{ shape: 'B, 6, 16' }] },
};

const rmsNorm: NodeSpec = {
  ...layerNorm,
  type: 'torch.nn.RMSNorm',
  displayName: L('RMSNorm', 'RMS 归一化'),
  summary: L('Scales by the root mean square of the trailing dimensions.', '按末尾若干维的均方根缩放。'),
  docUrl: doc('RMSNorm'),
  params: [
    P.ints('normalized_shape', '归一化的形状', { allowAuto: true, initial: 'auto', min: 1 }),
    P.float('eps', '数值稳定项 ε', { default: null, nullable: true }),
    P.bool('elementwise_affine', '可学习缩放', { default: true }),
  ],
  paramCount: (r) => (r.elementwise_affine ? tuple(r.normalized_shape, 1).reduce((a, b) => a * b, 1) : 0),
  codegen: { ctor: 'nn.RMSNorm' },
  example: { inputs: [{ shape: 'B, 5, 12' }], params: { eps: 1e-6 } },
};

const groupNorm: NodeSpec = {
  type: 'torch.nn.GroupNorm',
  kind: 'layer',
  category: 'norm',
  displayName: L('GroupNorm', '组归一化'),
  summary: L('Normalizes groups of channels within each sample.', '在每个样本内按通道分组归一化。'),
  docUrl: doc('GroupNorm'),
  params: [
    P.int('num_groups', '分组数', { min: 1, initial: 8 }),
    P.int('num_channels', '通道数', { allowAuto: true, min: 1, initial: 'auto' }),
    eps(),
    P.bool('affine', '可学习缩放与偏移', { default: true }),
  ],
  inputs: IN,
  outputs: OUT,
  defaultName: 'norm',
  infer([x], params) {
    if (x!.shape.length < 2) return fail('GroupNorm 需要至少 2 维输入');
    const c = feature(params, 'num_channels', x!.shape[1], '通道数');
    if (typeof c === 'string') return fail(c);
    if (c % num(params, 'num_groups') !== 0) return fail(`通道数 ${c} 不能被 num_groups=${num(params, 'num_groups')} 整除`);
    return ok([x!], resolveParams(params, { num_channels: c }));
  },
  paramCount: (r) => (r.affine ? 2 * (r.num_channels as number) : 0),
  describe: (r) => `${r.num_groups} 组 · ${r.num_channels ?? '?'} 通道`,
  codegen: { ctor: 'nn.GroupNorm' },
  example: { inputs: [{ shape: 'B, 8, 4, 4' }], params: { num_groups: 2 } },
};

export const NORM_SPECS: NodeSpec[] = [
  channelNorm('BatchNorm1d', '1D 批归一化', [2, 3], 'batch', { inputs: [{ shape: 'B, 8' }] }),
  channelNorm('BatchNorm2d', '2D 批归一化', [4], 'batch', { inputs: [{ shape: 'B, 4, 8, 8' }], params: { eps: 0.001, momentum: null } }),
  channelNorm('BatchNorm3d', '3D 批归一化', [5], 'batch', { inputs: [{ shape: 'B, 2, 4, 4, 4' }], params: { affine: false } }),
  channelNorm('InstanceNorm1d', '1D 实例归一化', [3, 2], 'instance', { inputs: [{ shape: 'B, 4, 10' }], params: { affine: true } }),
  channelNorm('InstanceNorm2d', '2D 实例归一化', [4, 3], 'instance', { inputs: [{ shape: 'B, 3, 8, 8' }] }),
  layerNorm,
  rmsNorm,
  groupNorm,
];

