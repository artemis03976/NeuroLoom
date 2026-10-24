// Every NodeSpec round-trips through the real Python backend (the M0 report's rule:
// each new spec comes with a round-trip case). The cases are the specs' own examples.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { allSpecs, analyzeModule, diffSignatures, generateCode, manifestText, parseProject, structuralSignature } from '../src';
import { exampleProject } from './fixtures';
import { hasTorch, importMany, inputShapesOf, pythonDir, shapePairs, type ImportResult } from './python';

const testable = allSpecs().filter((s) => s.kind === 'layer' || s.kind === 'op');

describe('node library', () => {
  it('gives every layer and op an example', () => {
    expect(testable.filter((s) => !s.example).map((s) => s.type)).toEqual([]);
  });

  it('gives every layer and op a shape rule', () => {
    expect(testable.filter((s) => !s.shapeDoc?.en || !s.shapeDoc.zh).map((s) => s.type)).toEqual([]);
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
