import { msg } from '../i18n';
import { fail, feature, IN, L, lit, num, ok, OUT, P, resolveParams } from './helpers';
import type { NodeSpec } from './types';

const doc = (name: string) => `https://docs.pytorch.org/docs/stable/generated/torch.nn.${name}.html`;

export const linear: NodeSpec = {
  type: 'torch.nn.Linear',
  kind: 'layer',
  category: 'linear',
  displayName: L('Linear', '全连接'),
  summary: L('Applies y = xAᵀ + b to the last dimension.', '对最后一维做线性变换 y = xAᵀ + b。'),
  docUrl: doc('Linear'),
  params: [
    P.int('in_features', '输入特征数', { allowAuto: true, min: 1, initial: 'auto' }),
    P.int('out_features', '输出特征数', { min: 1, initial: 64 }),
    P.bool('bias', '偏置', { default: true }),
  ],
  inputs: IN,
  outputs: OUT,
  defaultName: 'fc',
  infer([x], params) {
    const inFeatures = feature(params, 'in_features', x!.shape.at(-1), msg('输入最后一维', 'the last input dimension'));
    if (typeof inFeatures === 'string') return fail(inFeatures);
    const outFeatures = num(params, 'out_features');
    return ok([{ shape: [...x!.shape.slice(0, -1), outFeatures], dtype: x!.dtype }], resolveParams(params, { in_features: inFeatures }));
  },
  paramCount: (r) => (r.in_features as number) * (r.out_features as number) + (r.bias ? (r.out_features as number) : 0),
  describe: (r) => `${r.in_features ?? '?'} → ${r.out_features ?? '?'}`,
  codegen: { ctor: 'nn.Linear' },
  example: { inputs: [{ shape: 'B, 16' }], params: { out_features: 8, bias: false } },
};

export const identity: NodeSpec = {
  type: 'torch.nn.Identity',
  kind: 'layer',
  category: 'linear',
  displayName: L('Identity', '恒等'),
  summary: L('Returns its input unchanged; a placeholder layer.', '原样返回输入，常用作占位层。'),
  docUrl: doc('Identity'),
  params: [],
  inputs: IN,
  outputs: OUT,
  defaultName: 'identity',
  infer: ([x]) => ok([x!], {}),
  codegen: { ctor: 'nn.Identity' },
  example: { inputs: [{ shape: 'B, 4' }] },
};

export const embedding: NodeSpec = {
  type: 'torch.nn.Embedding',
  kind: 'layer',
  category: 'linear',
  displayName: L('Embedding', '嵌入'),
  summary: L('Looks up a learned vector for each integer index.', '为每个整数索引查出一个可学习向量。'),
  docUrl: doc('Embedding'),
  params: [
    P.int('num_embeddings', '词表大小', { min: 1, initial: 1000 }),
    P.int('embedding_dim', '向量维度', { min: 1, initial: 64 }),
    P.int('padding_idx', '填充索引', { nullable: true, default: null }),
  ],
  inputs: IN,
  outputs: OUT,
  defaultName: 'embed',
  infer([x], params) {
    if (x!.dtype !== 'int64') {
      return fail(
        msg('嵌入层的输入必须是整数张量（把输入节点的数据类型设为 int64）', "the embedding input must be an integer tensor (set the input node's dtype to int64)"),
      );
    }
    const pad = lit(params, 'padding_idx');
    if (typeof pad === 'number' && (pad >= num(params, 'num_embeddings') || pad < -num(params, 'num_embeddings'))) {
      return fail(msg('padding_idx 超出词表范围', 'padding_idx is out of range for num_embeddings'));
    }
    return ok([{ shape: [...x!.shape, num(params, 'embedding_dim')], dtype: 'float32' }], resolveParams(params));
  },
  paramCount: (r) => (r.num_embeddings as number) * (r.embedding_dim as number),
  describe: (r) => `${r.num_embeddings} × ${r.embedding_dim}`,
  codegen: { ctor: 'nn.Embedding' },
  example: { inputs: [{ shape: 'B, 12', dtype: 'int64' }], params: { num_embeddings: 100, embedding_dim: 16, padding_idx: 0 } },
};

export const LINEAR_SPECS = [linear, identity, embedding];
