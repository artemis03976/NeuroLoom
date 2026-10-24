// Model cost (PRD F8.2): totals against PyTorch itself (FlopCounterMode and the
// real parameter count), and the per-instance breakdown.
import { beforeAll, describe, expect, it } from 'vitest';
import { analyzeProject, costOf, dimValue, entryModule, generateCode, literal, symbolsOf, tensorBytes, type CostRow, type LoomProject } from '../src';
import { build, editEntry, resnet } from './fixtures';
import { hasTorch, inputShapesOf, runPython } from './python';

/** Linear(16, 16) three times in series, the first one from 8 features. */
function repeatedLinear(): LoomProject {
  const p = build(
    'Stack',
    [
      ['x', 'Input', { shape: literal('B, T, 8') }],
      ['fc', 'Linear', { in_features: literal(16), out_features: literal(16) }],
      ['output', 'Output'],
    ],
    ['x -> fc', 'fc -> output'],
  );
  return editEntry(p, (m) => ({ ...m, nodes: { ...m.nodes, fc: { ...m.nodes.fc!, repeat: { count: literal(3), first: { in_features: literal(8) } } } } }));
}

const CASES: Record<string, () => LoomProject> = { resnet, repeatedLinear };
const SYMBOLS = { B: 2, T: 5 };

const flatten = (rows: CostRow[]): CostRow[] => rows.flatMap((r) => [r, ...flatten(r.children ?? [])]);

describe('model cost', () => {
  it('evaluates symbolic dimensions and tensor sizes', () => {
    expect(dimValue('B*4+2', { B: 3 })).toBe(14);
    expect(dimValue('B*T', { B: 2 })).toBeNaN();
    expect(tensorBytes({ shape: ['B', 10], dtype: 'int64' }, { B: 2 })).toBe(160);
    expect(symbolsOf(analyzeProject(repeatedLinear()).entry)).toEqual(['B', 'T']);
  });

  it('names instances like named_modules() and sums them up the tree', () => {
    const p = resnet();
    const cost = costOf(p, analyzeProject(p).entry, { B: 1 });
    const blocks = cost.rows.find((r) => r.name === 'blocks')!;
    expect(blocks.children!.map((r) => r.name)).toEqual(['blocks.0', 'blocks.1']);
    expect(blocks.children![0]!.children!.map((r) => r.name)).toContain('blocks.0.conv1');
    expect(blocks.flops).toBe(blocks.children!.reduce((s, r) => s + r.flops, 0));
    expect(cost.params).toBe(analyzeProject(p).entry.totalParams);
    expect(cost.paramBytes).toBe(cost.params * 4);
    expect(cost.partial).toBe(false);
    // Every tensor is counted once: the input, then every layer and op output.
    const conv1 = flatten(cost.rows).find((r) => r.name === 'blocks.0.conv1')!;
    expect(conv1.activationBytes).toBe(16 * 32 * 32 * 4);
  });

  it('marks what it cannot count', () => {
    const p = editEntry(resnet(), (m) => ({ ...m, nodes: { ...m.nodes, x: { ...m.nodes.x!, params: { shape: literal('B, 3, H, H') } } } }));
    const cost = costOf(p, analyzeProject(p).entry, { B: 1 });
    expect(cost.partial).toBe(true);
    expect(costOf(p, analyzeProject(p).entry, { B: 1, H: 32 }).partial).toBe(false);
  });
});

// The generated code, built and run once with every symbol at its value.
const SCRIPT = `
import json, sys, torch
from torch.utils.flop_counter import FlopCounterMode
results = []
for case in json.load(sys.stdin):
    scope = {}
    exec(compile(case["source"], case["name"], "exec"), scope)
    model = scope[case["className"]]()
    args = [torch.randn(shape) for shape in case["inputs"]]
    with FlopCounterMode(display=False) as counter:
        model(*args)
    results.append({"flops": counter.get_total_flops(), "params": sum(p.numel() for p in model.parameters())})
print(json.dumps(results))
`;

describe.skipIf(!hasTorch)('model cost agrees with PyTorch', () => {
  let torch: { flops: number; params: number }[] = [];
  const names = Object.keys(CASES);

  beforeAll(() => {
    const input = names.map((name) => {
      const p = CASES[name]!();
      const inputs = Object.values(inputShapesOf(p)).map((s) => s.split(':')[0]!.split(',').map((d) => dimValue(/^\d+$/.test(d) ? Number(d) : d, SYMBOLS)));
      return { name, source: generateCode(p).code, className: entryModule(p).className, inputs };
    });
    torch = JSON.parse(runPython(['-c', SCRIPT], JSON.stringify(input)));
  }, 120_000);

  names.forEach((name, i) => {
    it(name, () => {
      const p = CASES[name]!();
      const cost = costOf(p, analyzeProject(p).entry, SYMBOLS);
      expect({ flops: cost.flops, params: cost.params }).toEqual(torch[i]);
    });
  });
});
