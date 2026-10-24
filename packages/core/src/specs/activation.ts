import { msg } from '../i18n';
import { axis } from '../shape';
import { fail, inplaceParam, IN, L, lit, num, ok, OUT, P, resolveParams, SAME_SHAPE } from './helpers';
import type { Inputs, NodeSpec, ParamSpec, Params, PythonCall } from './types';

const doc = (path: string) => `https://docs.pytorch.org/docs/stable/generated/${path}.html`;

/** Element-wise: the output has the input's shape. Softmax-like ones also check `dim`. */
function elementwise([x]: Inputs, params: Params) {
  const dim = lit(params, 'dim');
  if (typeof dim === 'number' && axis(dim, x!.shape.length) === undefined) {
    return fail(msg(`dim=${dim} 超出 ${x!.shape.length} 维输入的范围`, `dim=${dim} is out of range for ${x!.shape.length}-D input`));
  }
  return ok([x!], resolveParams(params));
}

const EXAMPLE = { inputs: [{ shape: 'B, 8' }] };

function layer(
  cls: string,
  zh: string,
  summary: [string, string],
  params: ParamSpec[] = [],
  extra: Partial<NodeSpec> = {},
): NodeSpec {
  return {
    type: `torch.nn.${cls}`,
    kind: 'layer',
    category: 'activation',
    displayName: L(cls, zh),
    summary: L(...summary),
    docUrl: doc(`torch.nn.${cls}`),
    params,
    inputs: IN,
    outputs: OUT,
    defaultName: 'act',
    infer: elementwise,
    shapeDoc: SAME_SHAPE,
    codegen: { ctor: `nn.${cls}` },
    example: EXAMPLE,
    ...extra,
  };
}

function fn(
  type: string,
  call: { fn: string; display: string; zh: string; summary: [string, string]; doc: string; name: string },
  params: ParamSpec[],
  python: Omit<PythonCall, 'args'>,
  example = EXAMPLE as NodeSpec['example'],
): NodeSpec {
  return {
    type,
    kind: 'op',
    category: 'activation',
    displayName: L(call.display, call.zh),
    summary: L(...call.summary),
    docUrl: doc(call.doc),
    params,
    inputs: IN,
    outputs: OUT,
    defaultName: call.name,
    infer: elementwise,
    shapeDoc: SAME_SHAPE,
    codegen: { fn: call.fn },
    python: {
      functions: python.functions,
      methods: python.methods,
      args: [{ name: 'input', port: 'in' }, ...params.map((p) => ({ name: p.name, param: p.name }))],
    },
    example,
  };
}

const dimParam = (extra: Partial<ParamSpec> = {}) => P.int('dim', '维度', { nullable: true, default: null, initial: -1, ...extra });

export const ACTIVATION_SPECS: NodeSpec[] = [
  layer('ReLU', 'ReLU', ['Element-wise max(0, x).', '逐元素 max(0, x)。'], [inplaceParam()], {
    example: { inputs: [{ shape: 'B, 8' }], params: { inplace: true } },
  }),
  layer('ReLU6', 'ReLU6', ['min(max(0, x), 6).', 'min(max(0, x), 6)。'], [inplaceParam()]),
  layer('LeakyReLU', 'LeakyReLU', ['x for x > 0, slope·x otherwise.', 'x>0 时为 x，否则为 斜率·x。'], [
    P.float('negative_slope', '负半轴斜率', { default: 0.01 }),
    inplaceParam(),
  ], { example: { inputs: [{ shape: 'B, 8' }], params: { negative_slope: 0.2 } } }),
  layer('PReLU', 'PReLU', ['LeakyReLU with a learned slope.', '斜率可学习的 LeakyReLU。'], [
    P.int('num_parameters', '斜率个数', { default: 1, min: 1 }),
  ], {
    infer([x], params) {
      const n = num(params, 'num_parameters');
      const channels = x!.shape[1];
      if (n !== 1 && typeof channels === 'number' && channels !== n) {
        return fail(msg(`num_parameters 应为 1 或通道数 ${channels}`, `num_parameters must be 1 or the number of channels (${channels})`));
      }
      return ok([x!], resolveParams(params));
    },
    paramCount: (r) => r.num_parameters as number,
    example: { inputs: [{ shape: 'B, 4, 6' }], params: { num_parameters: 4 } },
  }),
  layer('ELU', 'ELU', ['x for x > 0, α(eˣ − 1) otherwise.', 'x>0 时为 x，否则为 α(eˣ−1)。'], [
    P.float('alpha', 'α', { default: 1.0 }),
    inplaceParam(),
  ]),
  layer('GELU', 'GELU', ['Gaussian error linear unit.', '高斯误差线性单元。'], [
    P.enum('approximate', '近似方式', { default: 'none', options: ['none', 'tanh'] }),
  ], { example: { inputs: [{ shape: 'B, 8' }], params: { approximate: 'tanh' } } }),
  layer('SiLU', 'SiLU', ['x · sigmoid(x), also called Swish.', 'x·sigmoid(x)，又称 Swish。'], [inplaceParam()]),
  layer('Mish', 'Mish', ['x · tanh(softplus(x)).', 'x·tanh(softplus(x))。'], [inplaceParam()]),
  layer('Sigmoid', 'Sigmoid', ['1 / (1 + e⁻ˣ).', '1/(1+e⁻ˣ)。']),
  layer('Tanh', 'Tanh', ['Hyperbolic tangent.', '双曲正切。']),
  layer('Softmax', 'Softmax', ['Normalizes along `dim` to sum to 1.', '沿 dim 归一化为和为 1 的概率。'], [dimParam()], {
    example: { inputs: [{ shape: 'B, 3, 8' }], params: { dim: 1 } },
  }),
  layer('LogSoftmax', 'LogSoftmax', ['log(softmax(x)) along `dim`.', '沿 dim 的 log(softmax(x))。'], [dimParam()], {
    example: { inputs: [{ shape: 'B, 8' }], params: { dim: -1 } },
  }),
  layer('Hardswish', 'Hardswish', ['A cheap approximation of Swish.', 'Swish 的低成本近似。'], [inplaceParam()]),
  layer('Softplus', 'Softplus', ['A smooth approximation of ReLU.', 'ReLU 的平滑近似。'], [
    P.float('beta', 'β', { default: 1.0 }),
    P.float('threshold', '阈值', { default: 20.0 }),
  ], { example: { inputs: [{ shape: 'B, 8' }], params: { beta: 2.0 } } }),

  fn(
    'torch.nn.functional.relu',
    { fn: 'F.relu', display: 'F.relu', zh: 'relu 函数', summary: ['ReLU as a function call.', '函数形式的 ReLU。'], doc: 'torch.nn.functional.relu', name: 'relu' },
    [inplaceParam()],
    { functions: ['torch.nn.functional.relu', 'torch.relu'], methods: ['relu'] },
  ),
  fn(
    'torch.nn.functional.gelu',
    { fn: 'F.gelu', display: 'F.gelu', zh: 'gelu 函数', summary: ['GELU as a function call.', '函数形式的 GELU。'], doc: 'torch.nn.functional.gelu', name: 'gelu' },
    [P.enum('approximate', '近似方式', { default: 'none', options: ['none', 'tanh'], keyword: true })],
    { functions: ['torch.nn.functional.gelu'] },
  ),
  fn(
    'torch.nn.functional.silu',
    { fn: 'F.silu', display: 'F.silu', zh: 'silu 函数', summary: ['SiLU as a function call.', '函数形式的 SiLU。'], doc: 'torch.nn.functional.silu', name: 'silu' },
    [inplaceParam()],
    { functions: ['torch.nn.functional.silu'] },
  ),
  fn(
    'torch.sigmoid',
    { fn: 'torch.sigmoid', display: 'torch.sigmoid', zh: 'sigmoid 函数', summary: ['Sigmoid as a function call.', '函数形式的 Sigmoid。'], doc: 'torch.sigmoid', name: 'sigmoid' },
    [],
    { functions: ['torch.sigmoid', 'torch.nn.functional.sigmoid'], methods: ['sigmoid'] },
  ),
  fn(
    'torch.tanh',
    { fn: 'torch.tanh', display: 'torch.tanh', zh: 'tanh 函数', summary: ['Tanh as a function call.', '函数形式的 Tanh。'], doc: 'torch.tanh', name: 'tanh' },
    [],
    { functions: ['torch.tanh', 'torch.nn.functional.tanh'], methods: ['tanh'] },
  ),
  fn(
    'torch.nn.functional.softmax',
    { fn: 'F.softmax', display: 'F.softmax', zh: 'softmax 函数', summary: ['Softmax along `dim`.', '沿 dim 做 softmax。'], doc: 'torch.nn.functional.softmax', name: 'softmax' },
    [P.int('dim', '维度', { initial: -1, keyword: true })],
    { functions: ['torch.nn.functional.softmax', 'torch.softmax'], methods: ['softmax'] },
    { inputs: [{ shape: 'B, 4, 8' }], params: { dim: -1 } },
  ),
  fn(
    'torch.nn.functional.log_softmax',
    { fn: 'F.log_softmax', display: 'F.log_softmax', zh: 'log_softmax 函数', summary: ['log(softmax(x)) along `dim`.', '沿 dim 的 log(softmax(x))。'], doc: 'torch.nn.functional.log_softmax', name: 'log_softmax' },
    [P.int('dim', '维度', { initial: -1, keyword: true })],
    { functions: ['torch.nn.functional.log_softmax', 'torch.log_softmax'], methods: ['log_softmax'] },
    { inputs: [{ shape: 'B, 8' }], params: { dim: 1 } },
  ),
  fn(
    'torch.nn.functional.leaky_relu',
    { fn: 'F.leaky_relu', display: 'F.leaky_relu', zh: 'leaky_relu 函数', summary: ['LeakyReLU as a function call.', '函数形式的 LeakyReLU。'], doc: 'torch.nn.functional.leaky_relu', name: 'leaky_relu' },
    [P.float('negative_slope', '负半轴斜率', { default: 0.01, positional: true }), inplaceParam()],
    { functions: ['torch.nn.functional.leaky_relu'] },
    { inputs: [{ shape: 'B, 8' }], params: { negative_slope: 0.2 } },
  ),
];
