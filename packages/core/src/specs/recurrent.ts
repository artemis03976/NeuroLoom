import { fail, feature, flag, IN, L, num, ok, P, rankError, resolveParams } from './helpers';
import type { Example, NodeSpec, OutputSpec } from './types';

/** Gates per cell: RNN 1, GRU 3, LSTM 4. */
const GATES = { RNN: 1, GRU: 3, LSTM: 4 } as const;

const EXAMPLES: Record<keyof typeof GATES, Example> = {
  RNN: { inputs: [{ shape: 'B, 7, 4' }], params: { hidden_size: 6, batch_first: true, nonlinearity: 'relu' } },
  GRU: { inputs: [{ shape: 'T, B, 8' }], params: { hidden_size: 12 } },
  LSTM: { inputs: [{ shape: 'B, 10, 8' }], params: { hidden_size: 16, num_layers: 2, batch_first: true, bidirectional: true } },
};

function recurrent(cls: keyof typeof GATES): NodeSpec {
  const isLstm = cls === 'LSTM';
  const outputs: OutputSpec[] = isLstm
    ? [
        { name: 'out', path: [0] },
        { name: 'h_n', path: [1, 0] },
        { name: 'c_n', path: [1, 1] },
      ]
    : [
        { name: 'out', path: [0] },
        { name: 'h_n', path: [1] },
      ];
  return {
    type: `torch.nn.${cls}`,
    kind: 'layer',
    category: 'recurrent',
    displayName: L(cls, { RNN: '循环网络', GRU: 'GRU 门控循环', LSTM: 'LSTM 长短期记忆' }[cls]),
    summary: L(
      'Runs a recurrent network over a sequence; returns every step and the final hidden state.',
      '在序列上运行循环网络，返回每一步的输出和最终隐藏状态。',
    ),
    docUrl: `https://docs.pytorch.org/docs/stable/generated/torch.nn.${cls}.html`,
    params: [
      P.int('input_size', '输入特征数', { allowAuto: true, min: 1, initial: 'auto' }),
      P.int('hidden_size', '隐藏层大小', { min: 1, initial: 64 }),
      P.int('num_layers', '层数', { default: 1, min: 1 }),
      ...(cls === 'RNN' ? [P.enum('nonlinearity', '激活函数', { default: 'tanh', options: ['tanh', 'relu'] })] : []),
      P.bool('bias', '偏置', { default: true }),
      P.bool('batch_first', '批维度在前', { default: false }),
      P.float('dropout', '层间 Dropout', { default: 0.0 }),
      P.bool('bidirectional', '双向', { default: false }),
    ],
    inputs: IN,
    outputs,
    defaultName: cls.toLowerCase(),
    infer([x], params) {
      const bad = rankError(x!, [3], cls);
      if (bad) return fail(bad);
      const inSize = feature(params, 'input_size', x!.shape[2], '输入最后一维');
      if (typeof inSize === 'string') return fail(inSize);
      const hidden = num(params, 'hidden_size');
      const dirs = flag(params, 'bidirectional') ? 2 : 1;
      const [a, b] = x!.shape;
      const batch = flag(params, 'batch_first') ? a! : b!;
      const out = { shape: [a!, b!, hidden * dirs], dtype: x!.dtype };
      const state = { shape: [num(params, 'num_layers') * dirs, batch, hidden], dtype: x!.dtype };
      return ok(isLstm ? [out, state, state] : [out, state], resolveParams(params, { input_size: inSize }));
    },
    paramCount: (r) => {
      const g = GATES[cls];
      const h = r.hidden_size as number;
      const dirs = r.bidirectional ? 2 : 1;
      let total = 0;
      for (let layer = 0; layer < (r.num_layers as number); layer++) {
        const inSize = layer === 0 ? (r.input_size as number) : h * dirs;
        total += dirs * (g * h * (inSize + h) + (r.bias ? 2 * g * h : 0));
      }
      return total;
    },
    describe: (r) => `${r.input_size ?? '?'} → ${r.hidden_size}${(r.num_layers as number) > 1 ? ` × ${r.num_layers} 层` : ''}${r.bidirectional ? ' · 双向' : ''}`,
    codegen: { ctor: `nn.${cls}`, unpack: isLstm ? '{out}, ({h_n}, {c_n})' : '{out}, {h_n}' },
    example: EXAMPLES[cls],
  };
}

export const RECURRENT_SPECS: NodeSpec[] = [recurrent('RNN'), recurrent('LSTM'), recurrent('GRU')];
