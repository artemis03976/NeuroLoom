import { literalOf } from '../ir';
import { parseShape, shapeText } from '../shape';
import { L, ok, OUT, P } from './helpers';
import type { NodeSpec } from './types';

export const input: NodeSpec = {
  type: 'neuroloom.Input',
  kind: 'input',
  category: 'io',
  displayName: L('Input', '输入'),
  summary: L('A forward() argument with a declared example shape.', 'forward() 的一个参数，需声明示例形状。'),
  params: [
    P.shape('shape', '形状', { initial: 'B, 784' }),
    P.enum('dtype', '数据类型', { default: 'float32', options: ['float32', 'int64'] }),
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

export const IO_SPECS = [input, output];
