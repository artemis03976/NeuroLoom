// Parameter and Buffer nodes (PRD appendix 20): tensors a module holds and reads in
// forward() as self.<name>, with shapes that may use the module's hyperparameters.
import { describe, expect, it } from 'vitest';
import { analyzeProject, costOf, generateCode, literal, parseProject } from '../src';
import { moduleOf, projectOf } from './fixtures';
import { hasTorch, importCode, inputShapesOf } from './python';

/** x * self.scale + self.offset, with scale shaped (dim,) from a hyperparameter. */
function scaled() {
  const mod = moduleOf(
    'main',
    'Scaled',
    [
      ['x', 'Input', { shape: literal('B, 8') }],
      ['scale', 'neuroloom.Parameter', { shape: { expr: '(dim,)' }, init: { literal: 'ones' } }],
      ['offset', 'neuroloom.Buffer', { shape: { literal: [8] }, persistent: { literal: false } }],
      ['mul', 'torch.mul'],
      ['add', 'torch.add'],
      ['output', 'Output'],
    ],
    ['x -> mul.a', 'scale -> mul.b', 'mul -> add.a', 'offset -> add.b', 'add -> output'],
    { hyperparams: [{ name: 'dim', annotation: 'int', default: { literal: 8 } }] },
  );
  return projectOf('main', [mod]);
}

describe('Parameter and Buffer nodes', () => {
  it('take their shape from hyperparameters and count as parameters, not activations', () => {
    const p = scaled();
    const analysis = analyzeProject(p);
    expect(analysis.entry.diagnostics.filter((d) => d.severity === 'error')).toEqual([]);
    expect(analysis.entry.nodes.scale!.outputs![0]!.shape).toEqual([8]);
    expect(analysis.entry.totalParams).toBe(8);
    const cost = costOf(p, analysis.entry, { B: 1 });
    expect(cost.rows.find((r) => r.name === 'scale')!.activationBytes).toBe(0);
  });

  it('are created in __init__ and read in forward()', () => {
    const { code } = generateCode(scaled());
    expect(code).toContain('self.scale = nn.Parameter(torch.ones(dim))');
    expect(code).toContain('self.register_buffer("offset", torch.zeros(8), persistent=False)');
    expect(code).toContain('mul = x * self.scale');
    expect(code).toContain('return mul + self.offset');
    expect(code).not.toMatch(/self\.scale = (?!nn\.Parameter)/);
  });

  it.skipIf(!hasTorch)('read back as the same graph', { timeout: 60_000 }, () => {
    const p = scaled();
    const code = generateCode(p).code;
    const again = parseProject(importCode(code, inputShapesOf(p)).project);
    expect(generateCode(again).code).toBe(code);
  });
});
