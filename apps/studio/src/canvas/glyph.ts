import { getSpec, type Category, type LoomProject, type NodeInst, type NodeSpec } from '@neuroloom/core';
import { pick, tr } from '../i18n';

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

export const categoryLabel = (category: Category): string =>
  ({
    io: tr('输入输出', 'Inputs and outputs'),
    linear: tr('线性与嵌入', 'Linear and embedding'),
    conv: tr('卷积与上采样', 'Convolution and upsampling'),
    pool: tr('池化', 'Pooling'),
    norm: tr('归一化', 'Normalization'),
    activation: tr('激活', 'Activation'),
    dropout: 'Dropout',
    recurrent: tr('循环', 'Recurrent'),
    attention: tr('注意力与 Transformer', 'Attention and Transformer'),
    tensor: tr('张量形变', 'Tensor reshaping'),
    math: tr('运算', 'Arithmetic'),
    module: tr('模块', 'Modules'),
    code: tr('代码块', 'Code blocks'),
  })[category];

export interface NodeBadge {
  glyph: string;
  color: string;
  /** The second line on the node: the PyTorch class, the module's class, or "Python". */
  type: string;
  /** Long name for the inspector header. */
  title: string;
}

/** How a node of any kind is labelled. */
export function nodeBadge(project: LoomProject, node: NodeInst): NodeBadge {
  if (node.kind === 'module') {
    const def = project.modules[node.type];
    const kind = def?.sequential ? 'nn.Sequential' : def?.opaque ? (def.opaque.source ? tr('Python 源码', 'Python source') : tr('外部模块', 'external')) : '';
    return {
      glyph: def?.sequential ? '≡' : def?.opaque ? '⟨⟩' : 'M',
      color: 'var(--nl-cat-module)',
      type: def ? (def.sequential ? 'nn.Sequential' : def.className) : node.type,
      title: def ? `${def.sequential ? tr('序列容器', 'Sequential container') : def.className}${kind && !def.sequential ? ` · ${kind}` : ''}` : node.type,
    };
  }
  if (node.kind === 'code') return { glyph: '{}', color: 'var(--nl-cat-code)', type: 'Python', title: tr('代码块', 'Code block') };
  const spec = getSpec(node.type);
  if (!spec) return { glyph: '?', color: 'var(--nl-cat-code)', type: node.type, title: node.type };
  return { glyph: glyphOf(spec), color: categoryColor(spec), type: spec.displayName.en, title: pick(spec.displayName) };
}
