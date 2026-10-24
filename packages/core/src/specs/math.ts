// Arithmetic and reductions (PRD appendix: 算术与归约).
import { msg } from '../i18n';
import { axis, broadcast, dimEq, type Dim, type TensorType } from '../shape';
import { fail, L, lit, ok, OUT, P, prod, resolveParams } from './helpers';
import type { NodeSpec } from './types';

const doc = (path: string) => `https://docs.pytorch.org/docs/stable/generated/${path}.html`;

function binary(name: 'add' | 'sub' | 'mul' | 'div', operator: string, zh: string, functions: string[]): NodeSpec {
  return {
    type: `torch.${name}`,
    kind: 'op',
    category: 'math',
    displayName: L(`${operator}  (${name})`, zh),
    summary: L(
      `Element-wise ${name} with broadcasting; b can be a constant instead of a tensor.`,
      `带广播的逐元素${zh}；b 也可以不连接，改用常数。`,
    ),
    docUrl: doc(`torch.${name}`),
    params: [P.float('other', '常数 b（未连接 b 时使用）', { nullable: true, default: null })],
    inputs: [{ name: 'a' }, { name: 'b', optional: true }],
    outputs: OUT,
    defaultName: name,
    infer([a, b], params) {
      if (!b) {
        const other = lit(params, 'other');
        if (other === null || other === undefined) return fail(msg('请连接 b，或在参数中填写常数', 'connect b, or enter a constant in the parameters'));
        return ok([a!], resolveParams(params));
      }
      const shape = broadcast([a!.shape, b.shape]);
      return typeof shape === 'string' ? fail(shape) : ok([{ shape, dtype: a!.dtype }], resolveParams(params));
    },
    shapeDoc: L('The broadcast shape of a and b (the shape of a with a constant)', 'a 与 b 广播后的形状（b 为常数时同 a）'),
    describe: (r) => (r.other === null || r.other === undefined ? `a ${operator} b` : `a ${operator} ${r.other}`),
    codegen: { operator },
    python: {
      functions,
      args: [
        { name: 'input', port: 'a' },
        { name: 'other', port: 'b', scalarParam: 'other' },
      ],
    },
    example:
      name === 'add'
        ? { inputs: [{ shape: 'B, 4' }, { shape: 'B, 4' }] }
        : name === 'sub'
          ? { inputs: [{ shape: 'B, 3, 4' }, { shape: '4' }] }
          : { inputs: [{ shape: 'B, 4' }], params: { other: name === 'mul' ? 0.5 : 2.0 } },
  };
}

const matmul: NodeSpec = {
  type: 'torch.matmul',
  kind: 'op',
  category: 'math',
  displayName: L('@  (matmul)', '矩阵乘'),
  summary: L('Matrix product, batched over leading dimensions.', '矩阵乘法，前导维度按批处理并广播。'),
  docUrl: doc('torch.matmul'),
  params: [],
  inputs: [{ name: 'a' }, { name: 'b' }],
  outputs: OUT,
  defaultName: 'matmul',
  infer([a, b]) {
    const x = a!.shape;
    const y = b!.shape;
    if (x.length === 0 || y.length === 0) return fail(msg('matmul 不支持 0 维张量', 'matmul does not support 0-D tensors'));
    const xs = x.length === 1 ? [1, x[0]!] : x;
    const ys = y.length === 1 ? [y[0]!, 1] : y;
    if (dimEq(xs.at(-1)!, ys.at(-2)!) === false) {
      return fail(
        msg(
          `a 的最后一维 ${xs.at(-1)} 与 b 的倒数第二维 ${ys.at(-2)} 不一致`,
          `the last dimension of a (${xs.at(-1)}) does not match the second-to-last dimension of b (${ys.at(-2)})`,
        ),
      );
    }
    const batch = broadcast([xs.slice(0, -2), ys.slice(0, -2)]);
    if (typeof batch === 'string') return fail(batch);
    const shape: Dim[] = [...batch, ...(x.length === 1 ? [] : [xs.at(-2)!]), ...(y.length === 1 ? [] : [ys.at(-1)!])];
    return ok([{ shape, dtype: a!.dtype }], {});
  },
  // mm or bmm over the broadcast batch: 2 · output elements · contracted size. A 1-D b
  // makes PyTorch call mv or dot, which FlopCounterMode has no formula for, so 0. (With
  // a ≥3-D operand and a 1-D one the path also depends on strides and requires_grad,
  // which shapes cannot tell.)
  flops: (_r, [a, b], [out]) => (b!.length === 1 ? 0 : 2 * prod(out!) * a!.at(-1)!),
  shapeDoc: L(
    '[*, n, k] @ [*, k, m] → [*, n, m]; leading dimensions broadcast, a 1-D operand loses its dimension',
    '[*, n, k] @ [*, k, m] → [*, n, m]；前导维度广播，1 维操作数的那一维被去掉',
  ),
  codegen: { operator: '@' },
  python: {
    functions: ['operator.matmul', 'torch.matmul'],
    methods: ['matmul'],
    args: [
      { name: 'input', port: 'a' },
      { name: 'other', port: 'b' },
    ],
  },
  example: { inputs: [{ shape: 'B, 3, 4' }, { shape: 'B, 4, 5' }] },
};

function reduction(name: 'mean' | 'sum', zh: string): NodeSpec {
  return {
    type: `torch.Tensor.${name}`,
    kind: 'op',
    category: 'math',
    displayName: L(`.${name}`, zh),
    summary: L(`${name === 'mean' ? 'Average' : 'Sum'} over a dimension (or all of them).`, `沿某一维（或全部）求${zh.slice(1)}。`),
    docUrl: doc(`torch.Tensor.${name}`),
    params: [
      P.int('dim', '维度（空 = 全部）', { nullable: true, default: null, positional: true, initial: 1 }),
      P.bool('keepdim', '保留该维', { default: false }),
    ],
    inputs: [{ name: 'in' }],
    outputs: OUT,
    defaultName: name,
    infer([x], params) {
      const dim = lit(params, 'dim');
      const keep = lit(params, 'keepdim') === true;
      let shape: Dim[];
      if (dim === null || dim === undefined) {
        shape = keep ? x!.shape.map(() => 1) : [];
      } else {
        const d = axis(dim as number, x!.shape.length);
        if (d === undefined) return fail(msg(`dim=${dim} 超出 ${x!.shape.length} 维输入的范围`, `dim=${dim} is out of range for ${x!.shape.length}-D input`));
        shape = keep ? x!.shape.map((s, i) => (i === d ? 1 : s)) : x!.shape.filter((_, i) => i !== d);
      }
      return ok([{ shape, dtype: (x as TensorType).dtype }], resolveParams(params));
    },
    shapeDoc: L(
      'Dimension dim is removed (kept as size 1 with keepdim); dim = None reduces every dimension',
      '去掉第 dim 维（keepdim 时保留为 1）；dim 为空时归约所有维',
    ),
    describe: (r) => (r.dim === null ? msg('全部', 'all') : `dim = ${r.dim}`),
    codegen: { method: name },
    python: {
      functions: [`torch.${name}`],
      methods: [name],
      args: [
        { name: 'input', port: 'in' },
        { name: 'dim', param: 'dim' },
        { name: 'keepdim', param: 'keepdim' },
      ],
    },
    example: name === 'mean' ? { inputs: [{ shape: 'B, 4, 6' }], params: { dim: 1 } } : { inputs: [{ shape: 'B, 4, 6' }], params: { dim: -1, keepdim: true } },
  };
}

export const MATH_SPECS: NodeSpec[] = [
  binary('add', '+', '加法', ['operator.add', 'torch.add']),
  binary('sub', '-', '减法', ['operator.sub', 'torch.sub']),
  binary('mul', '*', '乘法', ['operator.mul', 'torch.mul']),
  binary('div', '/', '除法', ['operator.truediv', 'torch.div']),
  matmul,
  reduction('mean', '求平均'),
  reduction('sum', '求和'),
];
