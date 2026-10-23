// Every NodeSpec round-trips through the real Python backend (the M0 report's rule:
// each new spec comes with a round-trip case). The cases are the specs' own examples.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  allSpecs,
  analyzeModule,
  diffSignatures,
  emptyModule,
  entryModule,
  generateCode,
  initialParams,
  literal,
  manifestText,
  outputsOf,
  parseProject,
  SCHEMA_VERSION,
  structuralSignature,
  withDefaults,
  type LoomProject,
  type ModuleDef,
  type NodeSpec,
  type ParamValue,
} from '../src';
import { hasTorch, importMany, inputShapesOf, pythonDir, shapePairs, type ImportResult } from './python';

const INPUT_NAMES = ['x', 'y', 'z', 'w'];
const testable = allSpecs().filter((s) => s.kind === 'layer' || s.kind === 'op');

/** Inputs -> the spec's node -> one output per output port. */
export function exampleProject(spec: NodeSpec): LoomProject {
  const example = spec.example!;
  const mod: ModuleDef = emptyModule('main', 'Example');

  example.inputs.forEach((input, i) => {
    const id = `in${i}`;
    mod.nodes[id] = {
      id,
      kind: 'input',
      type: 'neuroloom.Input',
      name: INPUT_NAMES[i]!,
      params: { shape: literal(input.shape), dtype: literal(input.dtype ?? 'float32') },
    };
    mod.layout[id] = { x: 0, y: i * 120 };
    mod.interface.inputs.push(id);
  });

  const params: Record<string, ParamValue> = { ...initialParams(spec) };
  for (const [k, v] of Object.entries(example.params ?? {})) params[k] = v === 'auto' ? { auto: true } : literal(v);
  const name = spec.kind === 'layer' ? `${spec.defaultName}1` : spec.defaultName;
  mod.nodes.node = { id: 'node', kind: spec.kind, type: spec.type, name, params };
  mod.layout.node = { x: 260, y: 0 };

  let next = 0;
  for (const port of spec.inputs) {
    const count = port.variadic ? example.inputs.length - next : next < example.inputs.length ? 1 : 0;
    for (let i = 0; i < count; i++, next++) {
      mod.edges.push({
        id: `e${mod.edges.length}`,
        source: { node: `in${next}`, port: 'out' },
        target: { node: 'node', port: port.variadic ? `${port.name}${i}` : port.name },
      });
    }
  }

  outputsOf(spec, withDefaults(spec, params)).forEach((out, i) => {
    const id = `out${i}`;
    mod.nodes[id] = { id, kind: 'output', type: 'neuroloom.Output', name: i === 0 ? 'output' : `output_${i}`, params: {} };
    mod.layout[id] = { x: 520, y: i * 120 };
    mod.interface.outputs.push(id);
    mod.edges.push({ id: `e${mod.edges.length}`, source: { node: 'node', port: out.name }, target: { node: id, port: 'in' } });
  });

  return { schemaVersion: SCHEMA_VERSION, name: spec.type, entry: 'main', modules: { main: mod } };
}

describe('node library', () => {
  it('gives every layer and op an example', () => {
    expect(testable.filter((s) => !s.example).map((s) => s.type)).toEqual([]);
  });

  it('has a valid graph for every example', () => {
    const broken = testable.flatMap((s) => {
      const errors = analyzeModule(exampleProject(s)).diagnostics.filter((d) => d.severity === 'error');
      return errors.map((d) => `${s.type}: ${d.message}`);
    });
    expect(broken).toEqual([]);
  });

  it('ships an up-to-date manifest for the Python importer (run `npm run specs` if this fails)', () => {
    expect(readFileSync(join(pythonDir, 'neuroloom/specs.json'), 'utf8')).toBe(manifestText());
  });
});

describe.skipIf(!hasTorch)('every spec round-trips through PyTorch', () => {
  const cases = testable.map((spec) => {
    const project = exampleProject(spec);
    return { spec, project, code: generateCode(project).code };
  });
  let statics: ImportResult[] = [];
  let traces: ImportResult[] = [];

  beforeAll(() => {
    const requests = cases.map((c) => ({ source: c.code, inputShapes: inputShapesOf(c.project) }));
    statics = importMany(requests);
    traces = importMany(requests.map((r) => ({ ...r, mode: 'trace' as const })));
  }, 300_000);

  cases.forEach(({ spec, project, code }, i) => {
    it(spec.type, () => {
      // Read back statically: the same graph, and the same code when generated again.
      const result = statics[i]!;
      expect(result.error?.message, code).toBeUndefined();
      const imported = parseProject(result.project);
      expect(diffSignatures(structuralSignature(project), structuralSignature(imported))).toEqual({ onlyInA: [], onlyInB: [] });
      expect(generateCode(imported).code).toBe(code);

      // Traced by PyTorch: the same graph, and every shape agrees with inference.
      const trace = traces[i]!;
      expect(trace.error?.message, code).toBeUndefined();
      const traced = parseProject(trace.project);
      expect(diffSignatures(structuralSignature(project), structuralSignature(traced))).toEqual({ onlyInA: [], onlyInB: [] });
      for (const pair of shapePairs(traced, analyzeModule(traced), trace)) expect(pair.ts, pair.label).toEqual(pair.torch);
    });
  });
});
