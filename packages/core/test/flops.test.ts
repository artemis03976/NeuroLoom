// Every spec's FLOPs formula against PyTorch's own count: the generated code of
// the spec's example (and of a few variants) runs one forward pass under
// torch.utils.flop_counter.FlopCounterMode, with every symbolic dimension = 2.
import { beforeAll, describe, expect, it } from 'vitest';
import { allSpecs, analyzeProject, concretize, generateCode, getSpec, type Example, type NodeSpec, type TensorType } from '../src';
import { exampleProject } from './fixtures';
import { hasTorch, runPython } from './python';

/** The value of every symbolic dimension (B, T). */
const SYMBOL = 2;

/**
 * Cases where a spec's formula deliberately differs from what FlopCounterMode
 * reports, keyed by case name, with the reason. Each one must still differ, so
 * stale entries get noticed. None at the moment; the deliberate choices whose
 * cases still agree are noted at the formulas themselves:
 * - LSTM counts 0, because on CPU it runs as the fused mkldnn_rnn_layer op that
 *   FlopCounterMode has no formula for.
 * - The transformer layers count the attention core even though PyTorch's CPU
 *   flash kernel (used when dropout = 0 or in eval mode) has no formula; with the
 *   default dropout = 0.1 in training mode it runs as counted bmm calls.
 * - matmul with a 1-D b counts 0 (mv and dot have no formula).
 */
const KNOWN_DIFFERENCES: Record<string, string> = {};

/** More shapes and parameters than the examples cover, for the specs that count FLOPs. */
const VARIANTS: [type: string, example: Example][] = [
  ['torch.nn.Linear', { inputs: [{ shape: 'B, 3, 16' }], params: { out_features: 8 } }],
  ['torch.nn.Linear', { inputs: [{ shape: '16' }], params: { out_features: 4 } }],
  [
    'torch.nn.Conv2d',
    { inputs: [{ shape: 'B, 4, 9, 9' }], params: { out_channels: 8, kernel_size: 3, padding: 1, dilation: 2, groups: 2, padding_mode: 'reflect' } },
  ],
  ['torch.nn.Conv1d', { inputs: [{ shape: 'B, 4, 9' }], params: { out_channels: 6, kernel_size: 4, padding: 'same', groups: 2 } }],
  ['torch.nn.ConvTranspose2d', { inputs: [{ shape: 'B, 4, 5, 5' }], params: { out_channels: 6, kernel_size: 3, stride: 2, dilation: 2, groups: 2 } }],
  ['torch.matmul', { inputs: [{ shape: 'B, 1, 3, 4' }, { shape: '3, 4, 5' }] }],
  ['torch.matmul', { inputs: [{ shape: '3, 4' }, { shape: 'B, 4, 5' }] }],
  ['torch.matmul', { inputs: [{ shape: '4' }, { shape: '4, 5' }] }],
  ['torch.matmul', { inputs: [{ shape: 'B, 3, 4' }, { shape: '4' }] }],
  ['torch.matmul', { inputs: [{ shape: '4' }, { shape: '4' }] }],
  ['torch.nn.MultiheadAttention', { inputs: [{ shape: '5, B, 16' }, { shape: '7, B, 16' }, { shape: '7, B, 16' }], params: { num_heads: 2, bias: false } }],
  ['torch.nn.TransformerEncoderLayer', { inputs: [{ shape: 'T, B, 16' }], params: { nhead: 2, dim_feedforward: 24 } }],
  ['torch.nn.TransformerDecoderLayer', { inputs: [{ shape: '4, B, 8' }, { shape: '6, B, 8' }], params: { nhead: 2, dim_feedforward: 12, bias: false } }],
  ['torch.nn.RNN', { inputs: [{ shape: 'T, B, 4' }], params: { hidden_size: 6, num_layers: 2, bidirectional: true, bias: false } }],
  ['torch.nn.GRU', { inputs: [{ shape: 'B, 5, 8' }], params: { hidden_size: 4, num_layers: 3, batch_first: true } }],
  ['torch.nn.LSTM', { inputs: [{ shape: 'T, B, 8' }], params: { hidden_size: 16 } }],
];

const concrete = (types: (TensorType | undefined)[] = []): number[][] => types.map((t) => (t ? concretize(t.shape, SYMBOL) : []));

function flopCase(name: string, spec: NodeSpec) {
  const project = exampleProject(spec);
  const analysis = analyzeProject(project);
  const node = analysis.entry.nodes.node!;
  const expected = spec.flops?.(node.resolved ?? {}, concrete(node.inputs), concrete(node.outputs)) ?? 0;
  const inputs = spec.example!.inputs.map((input, i) => [concrete(analysis.entry.nodes[`in${i}`]!.outputs)[0]!, input.dtype ?? 'float32']);
  return { name, code: generateCode(project, analysis).code, inputs, expected };
}

const cases = [
  ...allSpecs()
    .filter((s) => s.example)
    .map((spec) => flopCase(spec.type, spec)),
  ...VARIANTS.map(([type, example], i) => flopCase(`${type} (variant ${i + 1})`, { ...getSpec(type)!, example })),
];

// One process for every case: build the model in its default (training) mode, make
// inputs of the example shapes (int64 ones are indices from torch.randint), run one
// forward pass under FlopCounterMode and print each total, or the error.
const SCRIPT = `
import json, sys, torch
from torch.utils.flop_counter import FlopCounterMode

results = []
for case in json.load(sys.stdin):
    try:
        scope = {}
        exec(compile(case['source'], case['name'], 'exec'), scope)
        model = scope['Example']()
        args = [torch.randint(0, 2, shape) if dtype == 'int64' else torch.randn(shape) for shape, dtype in case['inputs']]
        with FlopCounterMode(display=False) as counter:
            model(*args)
        results.append(counter.get_total_flops())
    except Exception as e:
        results.append(type(e).__name__ + ': ' + str(e))
print(json.dumps(results))
`;

describe('FLOPs formulas', () => {
  it('give a non-negative integer for every example', () => {
    const bad = cases.filter((c) => !Number.isSafeInteger(c.expected) || c.expected < 0).map((c) => `${c.name}: ${c.expected}`);
    expect(bad).toEqual([]);
  });

  it('list only real cases as known differences', () => {
    const names = new Set(cases.map((c) => c.name));
    expect(Object.keys(KNOWN_DIFFERENCES).filter((name) => !names.has(name))).toEqual([]);
  });
});

describe.skipIf(!hasTorch)('FLOPs agree with torch.utils.flop_counter.FlopCounterMode', () => {
  let torchFlops: (number | string)[] = [];

  beforeAll(() => {
    const input = cases.map((c) => ({ name: c.name, source: c.code, inputs: c.inputs }));
    torchFlops = JSON.parse(runPython(['-c', SCRIPT], JSON.stringify(input)));
  }, 300_000);

  cases.forEach(({ name, code, expected }, i) => {
    it(name, () => {
      const actual = torchFlops[i];
      expect(typeof actual, `${actual}\n\n${code}`).toBe('number');
      if (name in KNOWN_DIFFERENCES) expect(expected, KNOWN_DIFFERENCES[name]).not.toBe(actual);
      else expect(expected, code).toBe(actual);
    });
  });
});
