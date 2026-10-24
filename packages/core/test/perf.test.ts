// Performance budgets (PRD 13): after an edit, shape inference of a 500-node graph
// within 100 ms and code generation within 200 ms. Measured on residual chains
// (Linear → ReLU → add with a skip edge), the median of several runs after a
// warm-up. CI machines are slower, so the assertions allow twice the budget; the
// measured times are printed.
import { afterAll, describe, expect, it } from 'vitest';
import { analyzeProject, generateCode, literal, SCHEMA_VERSION, emptyModule, type LoomProject } from '../src';

/** `blocks` residual blocks of 3 nodes and 4 edges each, plus input and output. */
export function residualChain(blocks: number): LoomProject {
  const mod = emptyModule('main', 'Deep');
  const add = (id: string, type: string, params: Record<string, ReturnType<typeof literal>> = {}, kind: 'input' | 'output' | 'layer' | 'op' = 'layer') => {
    mod.nodes[id] = { id, kind, type, name: id, params };
    mod.layout[id] = { x: 0, y: 0 };
  };
  let edge = 0;
  const link = (a: string, b: string, port = 'in') => mod.edges.push({ id: `e${edge++}`, source: { node: a, port: 'out' }, target: { node: b, port } });
  add('x', 'neuroloom.Input', { shape: literal('B, 64') }, 'input');
  mod.interface.inputs.push('x');
  let prev = 'x';
  for (let i = 0; i < blocks; i++) {
    add(`fc${i}`, 'torch.nn.Linear', { in_features: literal(64), out_features: literal(64) });
    add(`act${i}`, 'torch.nn.ReLU');
    add(`add${i}`, 'torch.add', {}, 'op');
    link(prev, `fc${i}`);
    link(`fc${i}`, `act${i}`);
    link(`act${i}`, `add${i}`, 'a');
    link(prev, `add${i}`, 'b');
    prev = `add${i}`;
  }
  add('output', 'neuroloom.Output', {}, 'output');
  mod.interface.outputs.push('output');
  link(prev, 'output');
  return { schemaVersion: SCHEMA_VERSION, name: 'Deep', entry: 'main', modules: { main: mod } };
}

function median(run: () => void, times = 7): number {
  run();
  const samples = Array.from({ length: times }, () => {
    const start = performance.now();
    run();
    return performance.now() - start;
  });
  return samples.sort((a, b) => a - b)[Math.floor(times / 2)]!;
}

const report: string[] = [];
afterAll(() => console.log(`\nperformance (median ms)\n${report.join('\n')}`));

describe('performance', () => {
  for (const blocks of [166, 333]) {
    const p = residualChain(blocks);
    const nodes = Object.keys(p.modules.main!.nodes).length;
    it(`${nodes} nodes`, () => {
      expect(analyzeProject(p).entry.diagnostics.filter((d) => d.severity === 'error')).toEqual([]);
      const analysis = analyzeProject(p);
      const infer = median(() => analyzeProject(p));
      const codegen = median(() => generateCode(p, analysis));
      report.push(`${String(nodes).padStart(5)} nodes  inference ${infer.toFixed(1).padStart(7)}  codegen ${codegen.toFixed(1).padStart(7)}`);
      if (nodes <= 520) {
        expect(infer).toBeLessThan(200);
        expect(codegen).toBeLessThan(400);
      }
    });
  }
});
