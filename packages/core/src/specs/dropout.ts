import { msg } from '../i18n';
import { fail, IN, inplaceParam, L, num, ok, OUT, P, rankError, resolveParams, SAME_SHAPE } from './helpers';
import type { NodeSpec } from './types';

function dropout(cls: string, zh: string, ranks: number[] | undefined, summary: [string, string], example: string): NodeSpec {
  return {
    type: `torch.nn.${cls}`,
    kind: 'layer',
    category: 'dropout',
    displayName: L(cls, zh),
    summary: L(...summary),
    docUrl: `https://docs.pytorch.org/docs/stable/generated/torch.nn.${cls}.html`,
    params: [P.float('p', '丢弃概率', { default: 0.5 }), inplaceParam()],
    inputs: IN,
    outputs: OUT,
    defaultName: 'dropout',
    infer([x], params) {
      const p = num(params, 'p');
      if (p < 0 || p > 1) return fail(msg('p 必须在 0 到 1 之间', 'p must be between 0 and 1'));
      const bad = ranks && rankError(x!, ranks, cls);
      return bad ? fail(bad) : ok([x!], resolveParams(params));
    },
    shapeDoc: SAME_SHAPE,
    describe: (r) => `p = ${r.p}`,
    codegen: { ctor: `nn.${cls}` },
    example: { inputs: [{ shape: example }], params: { p: 0.1 } },
  };
}

export const DROPOUT_SPECS: NodeSpec[] = [
  dropout('Dropout', 'Dropout', undefined, ['Zeroes random elements during training.', '训练时随机将元素置零。'], 'B, 8'),
  dropout('Dropout1d', 'Dropout1d', [2, 3], ['Zeroes whole channels of 1-D feature maps.', '随机将一维特征图的整个通道置零。'], 'B, 4, 10'),
  dropout('Dropout2d', 'Dropout2d', [3, 4], ['Zeroes whole channels of 2-D feature maps.', '随机将二维特征图的整个通道置零。'], 'B, 4, 6, 6'),
  dropout('Dropout3d', 'Dropout3d', [4, 5], ['Zeroes whole channels of 3-D feature maps.', '随机将三维特征图的整个通道置零。'], 'B, 2, 4, 4, 4'),
  dropout('AlphaDropout', 'AlphaDropout', undefined, ['Dropout that keeps SELU self-normalization.', '保持 SELU 自归一化性质的 Dropout。'], 'B, 8'),
];
