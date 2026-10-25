import { literalOf } from '../ir';
import { parseShape, shapeText } from '../shape';
import { msg } from '../i18n';
import { L, ok, OUT, P, resolveParams } from './helpers';
import type { NodeSpec } from './types';

export const input: NodeSpec = {
  type: 'neuroloom.Input',
  kind: 'input',
  category: 'io',
  displayName: L('Input', '输入'),
  summary: L('A forward() argument with a declared example shape.', 'forward() 的一个参数，需声明示例形状。'),
  params: [
    P.shape('shape', '形状', { initial: 'B, 784' }),
    P.enum('dtype', '数据类型', { default: 'float32', options: ['float32', 'int64', 'bool'] }),
  ],
  inputs: [],
  outputs: OUT,
  defaultName: 'x',
  infer(_inputs, params) {
    const parsed = parseShape(String(literalOf(params.shape) ?? ''));
    if (!parsed.ok) return parsed;
    const dtype = String(literalOf(params.dtype) ?? 'float32');
    return ok([{ shape: parsed.shape, dtype }], { shape: shapeText(parsed.shape), dtype });
  },
  describe: (r) => `[${r.shape ?? '?'}]${r.dtype && r.dtype !== 'float32' ? ` ${r.dtype}` : ''}`,
  codegen: {},
};

export const output: NodeSpec = {
  type: 'neuroloom.Output',
  kind: 'output',
  category: 'io',
  displayName: L('Output', '输出'),
  summary: L('A value returned from forward().', 'forward() 的一个返回值。'),
  params: [],
  inputs: [{ name: 'in' }],
  outputs: [],
  defaultName: 'output',
  // Output has no output ports; the input type is recorded so the UI can show it.
  infer: ([x]) => ok([x!], {}),
  codegen: {},
};

const STATE_DTYPES = ['float32', 'float64', 'float16', 'bfloat16', 'int64', 'int32', 'bool'];

/** A tensor the module holds: an nn.Parameter (learned) or a buffer (state such as an index table). */
function state(kind: 'parameter' | 'buffer'): NodeSpec {
  const parameter = kind === 'parameter';
  return {
    type: parameter ? 'neuroloom.Parameter' : 'neuroloom.Buffer',
    kind: 'layer',
    category: 'io',
    displayName: parameter ? L('Parameter', '参数张量') : L('Buffer', '缓冲区'),
    summary: parameter
      ? L('A learned tensor of the module (nn.Parameter), such as ViT’s class token.', '模块自己的可学习张量（nn.Parameter），如 ViT 的 class token。')
      : L('A tensor the module keeps but does not learn (register_buffer), such as an index table.', '模块保存但不学习的张量（register_buffer），如索引表。'),
    docUrl: parameter
      ? 'https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html'
      : 'https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer',
    params: [
      P.dims('shape', '形状', { initial: [1, 1, 64] }),
      P.enum('init', '初始值', { default: 'zeros', options: ['zeros', 'ones', 'randn', 'rand', 'empty'] }),
      P.enum('dtype', '数据类型', { default: 'float32', options: STATE_DTYPES }),
      parameter ? P.bool('requires_grad', '参与训练', { default: true }) : P.bool('persistent', '保存在 state_dict 中', { default: true }),
    ],
    inputs: [],
    outputs: OUT,
    defaultName: parameter ? 'weight' : 'buffer',
    infer(_inputs, params) {
      const shape = literalOf(params.shape);
      if (!Array.isArray(shape) || !shape.every((d) => typeof d === 'number' && Number.isInteger(d) && d >= 0)) {
        return { ok: false, error: msg('形状必须是非负整数列表', 'the shape must be a list of non-negative integers') };
      }
      const dtype = String(literalOf(params.dtype) ?? 'float32');
      return ok([{ shape: shape as number[], dtype }], resolveParams(params));
    },
    paramCount: parameter ? (r) => (r.shape as number[]).reduce((a, b) => a * b, 1) : undefined,
    shapeDoc: L('The declared shape', '声明的形状'),
    describe: (r) => `[${Array.isArray(r.shape) ? r.shape.join(', ') : '?'}]${r.dtype && r.dtype !== 'float32' ? ` ${r.dtype}` : ''}`,
    codegen: { state: kind },
    example: { inputs: [], params: { shape: [1, 4, 8] } },
  };
}

export const parameter = state('parameter');
export const buffer = state('buffer');

export const IO_SPECS = [input, output, parameter, buffer];
