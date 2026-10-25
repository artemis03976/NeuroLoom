// PRD 13 success metric: static import of model files from common repositories
// (python/tests/corpus/manifest.json, chosen as docs/import-corpus-report.md describes).
// A file counts when
//   1. the static import of the file succeeds;
//   2. the generated code is the same model: tests/corpus/realworld.py builds the
//      original as its repository does and the generated code beside it, loads the
//      original weights with strict=True and compares the outputs;
//   3. no class the entry uses is kept as source (L3). Code blocks are allowed.
// The rate over the 50 measured files must not fall below the baseline; the 12
// held-out files, which importer changes are not tuned on, are reported apart.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { generateCode, parseProject, type LoomProject, type ModuleDef } from '../src';
import { canImport, hasTorch, importMany, pythonDir, runPython, type ImportResult } from './python';

const corpus = join(pythonDir, 'tests/corpus');
const hasCorpus = hasTorch && canImport('timm, transformers, diffusers, einops, scipy');

/** Measured files that pass today; the test fails if fewer do. */
const FLOOR = 31;

interface Entry {
  id: string;
  set: 'measure' | 'heldout';
  category: string;
  repo: string;
  path: string;
  entry: string;
  inputs: Record<string, string>;
}

interface Outcome {
  entry: Entry;
  /** Where the file failed; undefined when it passes. */
  stage?: 'import' | 'generate' | 'L3' | 'original' | 'load' | 'build' | 'state_dict' | 'run' | 'outputs';
  detail?: string;
  /** "Class:level/code blocks" for every class the entry uses. */
  classes: string[];
  /** No code blocks either. */
  strict: boolean;
}

/** The classes the entry module uses, itself included, by className. */
function usedClasses(project: LoomProject): Set<string> {
  const seen = new Set<string>();
  const visit = (mod: ModuleDef) => {
    if (seen.has(mod.className)) return;
    seen.add(mod.className);
    if (mod.opaque) return;
    for (const node of Object.values(mod.nodes)) {
      if (node.kind === 'module' && project.modules[node.type]) visit(project.modules[node.type]!);
    }
  };
  visit(project.modules[project.entry]!);
  return seen;
}

describe.skipIf(!hasCorpus)('third-party import corpus (PRD 13)', () => {
  const entries: Entry[] = JSON.parse(readFileSync(join(corpus, 'manifest.json'), 'utf8'));
  const outcomes: Outcome[] = [];

  beforeAll(() => {
    const sources = entries.map((e) => readFileSync(join(corpus, 'third_party', e.id, e.path), 'utf8'));
    const imported: ImportResult[] = importMany(entries.map((e, i) => ({ source: sources[i]!, className: e.entry, inputShapes: e.inputs, mode: 'static', locale: 'en' })));
    const generated: { id: string; generated: string }[] = [];
    entries.forEach((entry, i) => {
      const result = imported[i]!;
      if (result.error) {
        outcomes.push({ entry, stage: 'import', detail: result.error.message, classes: [], strict: false });
        return;
      }
      const project = parseProject(result.project);
      const used = usedClasses(project);
      const report = (result.report ?? []).filter((r) => used.has(r.className));
      const classes = report.map((r) => `${r.className}:${r.level}${r.codeBlocks ? `/${r.codeBlocks}` : ''}`);
      const strict = report.every((r) => r.codeBlocks === 0 && r.level !== 'L1+L3' && r.level !== 'L3');
      const kept = report.filter((r) => r.level === 'L3');
      try {
        generated.push({ id: entry.id, generated: generateCode(project).code });
      } catch (e) {
        outcomes.push({ entry, stage: 'generate', detail: String(e), classes, strict });
        return;
      }
      outcomes.push(
        kept.length > 0
          ? { entry, stage: 'L3', detail: kept.map((r) => `${r.className}: ${r.notes[0] ?? ''}`).join('; '), classes, strict }
          : { entry, classes, strict },
      );
    });
    // Every file that imported is checked, so a kept class (L3) still shows whether its code runs.
    const checks: { id: string; ok: boolean; stage?: Outcome['stage']; error?: string }[] = JSON.parse(
      runPython(['tests/corpus/realworld.py', 'check'], JSON.stringify(generated)),
    );
    for (const check of checks) {
      const outcome = outcomes.find((o) => o.entry.id === check.id)!;
      if (!check.ok && (outcome.stage === undefined || check.stage === 'original')) {
        outcome.stage = check.stage;
        outcome.detail = check.error;
      }
    }
  }, 900_000);

  afterAll(() => {
    if (outcomes.length === 0) return;
    const line = (o: Outcome) =>
      `${o.stage ? '·' : '✓'} ${o.entry.id.padEnd(28)} ${(o.stage ?? (o.strict ? 'pass' : 'pass, code blocks')).padEnd(18)} ${o.classes.join(' ').slice(0, 90)}${o.detail ? `\n      ${o.detail.slice(0, 200)}` : ''}`;
    const summary = (set: Entry['set']) => {
      const rows = outcomes.filter((o) => o.entry.set === set);
      const passed = rows.filter((o) => !o.stage);
      return `${passed.length}/${rows.length} pass (${Math.round((100 * passed.length) / rows.length)}%), ${passed.filter((o) => o.strict).length} without code blocks`;
    };
    const order = (set: Entry['set']) => outcomes.filter((o) => o.entry.set === set).map(line).join('\n');
    console.log(`\nthird-party corpus, measured: ${summary('measure')}\n${order('measure')}\n\nheld out: ${summary('heldout')}\n${order('heldout')}`);
  });

  it('builds every original from its repository files (the manifest is right)', () => {
    expect(outcomes.filter((o) => o.stage === 'original').map((o) => `${o.entry.id}: ${o.detail}`)).toEqual([]);
  });

  it(`imports at least ${FLOOR} of the 50 measured files`, () => {
    expect(entries.filter((e) => e.set === 'measure')).toHaveLength(50);
    expect(outcomes.filter((o) => o.entry.set === 'measure' && !o.stage).length).toBeGreaterThanOrEqual(FLOOR);
  });
});
