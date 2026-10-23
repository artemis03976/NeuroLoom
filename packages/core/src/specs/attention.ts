import { msg } from '../i18n';
import { dimEq } from '../shape';
import { fail, feature, flag, L, num, ok, OUT, P, rankError, resolveParams } from './helpers';
import type { NodeSpec, Params } from './types';

const doc = (cls: string) => `https://docs.pytorch.org/docs/stable/generated/torch.nn.${cls}.html`;

/** Parameters of nn.MultiheadAttention for embedding size e (with or without biases). */
const mhaParams = (e: number, bias: boolean) => 4 * e * e + (bias ? 4 * e : 0);

const layerParams = () => [
  P.int('d_model', '模型维度', { allowAuto: true, min: 1, initial: 'auto' }),
  P.int('nhead', '注意力头数', { min: 1, initial: 8 }),
  P.int('dim_feedforward', '前馈层维度', { default: 2048, min: 1 }),
  P.float('dropout', 'Dropout', { default: 0.1 }),
  P.enum('activation', '激活函数', { default: 'relu', options: ['relu', 'gelu'] }),
  P.float('layer_norm_eps', 'LayerNorm ε', { default: 1e-5 }),
  P.bool('batch_first', '批维度在前', { default: false }),
  P.bool('norm_first', '先归一化（Pre-LN）', { default: false }),
  P.bool('bias', '偏置', { default: true }),
];

/** `what` is the localized name of the last input dimension. */
function checkModel(params: Params, x: { shape: (number | string)[] }, what: string): number | string {
  const d = feature(params, 'd_model', x.shape[2], what);
  if (typeof d === 'string') return d;
  const nhead = num(params, 'nhead');
  if (d % nhead !== 0) return msg(`d_model=${d} 不能被 nhead=${nhead} 整除`, `d_model=${d} is not divisible by nhead=${nhead}`);
  return d;
}

export const multiheadAttention: NodeSpec = {
  type: 'torch.nn.MultiheadAttention',
  kind: 'layer',
  category: 'attention',
  displayName: L('MultiheadAttention', '多头注意力'),
  summary: L('Attends from queries to keys/values with several heads.', '用多个头从 query 关注 key/value。'),
  docUrl: doc('MultiheadAttention'),
  params: [
    P.int('embed_dim', '嵌入维度', { allowAuto: true, min: 1, initial: 'auto' }),
    P.int('num_heads', '头数', { min: 1, initial: 8 }),
    P.float('dropout', 'Dropout', { default: 0.0 }),
    P.bool('bias', '偏置', { default: true }),
    P.bool('batch_first', '批维度在前', { default: false }),
  ],
  inputs: [{ name: 'query' }, { name: 'key' }, { name: 'value' }],
  outputs: [
    { name: 'out', path: [0] },
    { name: 'weights', path: [1] },
  ],
  defaultName: 'attn',
  infer([q, k, v], params) {
    for (const [t, name] of [[q, 'query'], [k, 'key'], [v, 'value']] as const) {
      const bad = rankError(t!, [3], name);
      if (bad) return fail(bad);
    }
    const e = feature(params, 'embed_dim', q!.shape[2], msg('query 最后一维', 'the last dimension of query'));
    if (typeof e === 'string') return fail(e);
    const heads = num(params, 'num_heads');
    if (e % heads !== 0) return fail(msg(`embed_dim=${e} 不能被 num_heads=${heads} 整除`, `embed_dim=${e} is not divisible by num_heads=${heads}`));
    if (dimEq(k!.shape[2]!, e) === false || dimEq(v!.shape[2]!, e) === false) {
      return fail(msg('key 和 value 的最后一维必须等于 embed_dim', 'the last dimension of key and value must equal embed_dim'));
    }
    const batchFirst = flag(params, 'batch_first');
    const [batch, target] = batchFirst ? [q!.shape[0]!, q!.shape[1]!] : [q!.shape[1]!, q!.shape[0]!];
    const source = batchFirst ? k!.shape[1]! : k!.shape[0]!;
    if (dimEq(k!.shape[batchFirst ? 1 : 0]!, v!.shape[batchFirst ? 1 : 0]!) === false) {
      return fail(msg('key 与 value 的序列长度不一致', 'key and value have different sequence lengths'));
    }
    return ok([q!, { shape: [batch, target, source], dtype: q!.dtype }], resolveParams(params, { embed_dim: e }));
  },
  paramCount: (r) => mhaParams(r.embed_dim as number, r.bias as boolean),
  describe: (r) => msg(`${r.embed_dim ?? '?'} 维 · ${r.num_heads} 头`, `${r.embed_dim ?? '?'} dims · ${r.num_heads} heads`),
  codegen: { ctor: 'nn.MultiheadAttention', unpack: '{out}, {weights}' },
  example: { inputs: [{ shape: 'B, 5, 16' }, { shape: 'B, 7, 16' }, { shape: 'B, 7, 16' }], params: { num_heads: 4, batch_first: true } },
};

function transformerLayer(decoder: boolean): NodeSpec {
  const cls = decoder ? 'TransformerDecoderLayer' : 'TransformerEncoderLayer';
  return {
    type: `torch.nn.${cls}`,
    kind: 'layer',
    category: 'attention',
    displayName: L(cls, decoder ? 'Transformer 解码层' : 'Transformer 编码层'),
    summary: decoder
      ? L('Self-attention, cross-attention over memory, then a feed-forward block.', '自注意力、对 memory 的交叉注意力，再接前馈层。')
      : L('Self-attention followed by a feed-forward block.', '自注意力后接前馈层。'),
    docUrl: doc(cls),
    params: layerParams(),
    inputs: decoder ? [{ name: 'tgt' }, { name: 'memory' }] : [{ name: 'src' }],
    outputs: OUT,
    defaultName: decoder ? 'decoder_layer' : 'encoder_layer',
    infer([x, memory], params) {
      const bad = rankError(x!, [3], cls);
      if (bad) return fail(bad);
      const d = checkModel(params, x!, decoder ? msg('tgt 最后一维', 'the last dimension of tgt') : msg('输入最后一维', 'the last input dimension'));
      if (typeof d === 'string') return fail(d);
      if (decoder) {
        const badMemory = rankError(memory!, [3], 'memory');
        if (badMemory) return fail(badMemory);
        if (dimEq(memory!.shape[2]!, d) === false) return fail(msg('memory 的最后一维必须等于 d_model', 'the last dimension of memory must equal d_model'));
      }
      return ok([x!], resolveParams(params, { d_model: d }));
    },
    paramCount: (r) => {
      const d = r.d_model as number;
      const ff = r.dim_feedforward as number;
      const bias = r.bias as boolean;
      const attention = (decoder ? 2 : 1) * mhaParams(d, bias);
      const feedForward = 2 * d * ff + (bias ? ff + d : 0);
      const norms = (decoder ? 3 : 2) * d * (bias ? 2 : 1);
      return attention + feedForward + norms;
    },
    describe: (r) => msg(`${r.d_model ?? '?'} 维 · ${r.nhead} 头 · FF ${r.dim_feedforward}`, `${r.d_model ?? '?'} dims · ${r.nhead} heads · FF ${r.dim_feedforward}`),
    codegen: { ctor: `nn.${cls}` },
    example: decoder
      ? { inputs: [{ shape: 'B, 5, 16' }, { shape: 'B, 7, 16' }], params: { nhead: 4, dim_feedforward: 32, batch_first: true } }
      : { inputs: [{ shape: 'B, 6, 16' }], params: { nhead: 4, dim_feedforward: 32, batch_first: true, activation: 'gelu', norm_first: true } },
  };
}

export const ATTENTION_SPECS: NodeSpec[] = [multiheadAttention, transformerLayer(false), transformerLayer(true)];
