import type { Category, NodeSpec } from '@neuroloom/core';

const BY_TYPE: Record<string, string> = {
  'neuroloom.Input': 'in',
  'neuroloom.Output': 'out',
  'torch.nn.Linear': 'FC',
  'torch.nn.Identity': '=',
  'torch.nn.Embedding': 'E',
  'torch.nn.Upsample': '↑',
  'torch.nn.PixelShuffle': '↑',
  'torch.add': '+',
  'torch.sub': '−',
  'torch.mul': '×',
  'torch.div': '÷',
  'torch.matmul': '@',
  'torch.Tensor.mean': 'μ',
  'torch.Tensor.sum': 'Σ',
  'torch.cat': '⧺',
  'torch.stack': '≡',
  'torch.chunk': '⫼',
};

const BY_CATEGORY: Record<Category, string> = {
  io: 'io',
  linear: 'FC',
  conv: 'C',
  pool: 'P',
  norm: 'N',
  activation: 'σ',
  dropout: 'D',
  recurrent: '↻',
  attention: 'A',
  tensor: '⇄',
  math: '±',
  module: 'M',
  code: '{}',
};

/** A short badge shown on nodes and in the palette. */
export function glyphOf(spec: NodeSpec): string {
  if (BY_TYPE[spec.type]) return BY_TYPE[spec.type]!;
  if (spec.category === 'activation') return spec.kind === 'op' ? 'ƒ' : spec.displayName.en.slice(0, 1);
  if (spec.category === 'conv') return spec.type.includes('Transpose') ? 'CT' : 'C';
  return BY_CATEGORY[spec.category];
}

export const categoryColor = (spec: NodeSpec | undefined): string => `var(--nl-cat-${spec?.category ?? 'code'})`;

export const CATEGORY_LABEL: Record<Category, string> = {
  io: '输入输出',
  linear: '线性与嵌入',
  conv: '卷积与上采样',
  pool: '池化',
  norm: '归一化',
  activation: '激活',
  dropout: 'Dropout',
  recurrent: '循环',
  attention: '注意力与 Transformer',
  tensor: '张量形变',
  math: '运算',
  module: '模块',
  code: '代码块',
};
