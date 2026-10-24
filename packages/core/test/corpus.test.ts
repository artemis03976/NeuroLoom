// The handwritten import corpus (python/tests/corpus/handwritten): model files
// written in the styles of real repositories (the files taken from repositories
// are in realworld.test.ts). Each one is imported statically and dynamically; the code
// NeuroLoom generates must behave exactly like the original, and generating it
// again from its own import must give the same code. The levels reached are
// printed, so coverage can be followed as the importers improve.
//
// Every Python step runs once for all cases (import, equivalence, re-import):
// starting Python and importing torch costs more than the work itself.
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { analyzeProject, entryModule, generateCode, parseProject, type LoomProject, type ProjectAnalysis } from '../src';
import { hasTorch, importMany, inputShapesOf, pythonDir, runPython, type ImportMode, type ImportResult } from './python';

const corpusDir = join(pythonDir, 'tests/corpus/handwritten');
const files = readdirSync(corpusDir).filter((f) => f.endsWith('.py')).sort();
const modes: ImportMode[] = ['static', 'dynamic'];

/** "# key: value" lines at the top of a corpus file. */
function header(source: string): Record<string, string> {
  return Object.fromEntries([...source.matchAll(/^# (\w+): (.*)$/gm)].map((m) => [m[1]!, m[2]!.trim()]));
}

// For each case: loads both sources, copies weights with strict=True and compares outputs in eval mode.
const EQUIVALENCE = `
import json, sys, traceback, torch
def load(name, src, class_name):
    module = type(sys)(name)
    sys.modules[name] = module
    exec(src, module.__dict__)
    return module.__dict__[class_name]()
def check(i, p):
    torch.manual_seed(0)
    a = load(f"corpus_original_{i}", p["original"], p["className"])
    b = load(f"corpus_generated_{i}", p["generated"], p["className"])
    a.eval(); b.eval()
    missing = sorted(set(a.state_dict()) ^ set(b.state_dict()))
    if missing:
        return {"ok": False, "keys": missing[:8]}
    b.load_state_dict(a.state_dict(), strict=True)
    inputs = []
    for spec in p["inputs"]:
        dims, _, dtype = spec.partition(":")
        size = [int(d) if d.isdigit() else 2 for d in dims.split(",")]
        inputs.append(torch.randint(0, 10, size) if dtype == "int64" else torch.randn(size))
    with torch.no_grad():
        oa, ob = a(*inputs), b(*inputs)
    oa = oa if isinstance(oa, tuple) else (oa,)
    ob = ob if isinstance(ob, tuple) else (ob,)
    return {"ok": len(oa) == len(ob) and all(torch.allclose(x, y, atol=1e-6) for x, y in zip(oa, ob))}
results = []
for i, p in enumerate(json.load(sys.stdin)):
    try:
        results.append(check(i, p))
    except Exception:
        results.append({"ok": False, "error": traceback.format_exc(limit=4)})
print(json.dumps(results))
`;

interface Case {
  name: string;
  source: string;
  inputs: Record<string, string>;
  mode: ImportMode;
  imported?: ImportResult;
  project?: LoomProject;
  analysis?: ProjectAnalysis;
  code?: string;
  failure?: string;
  check?: { ok: boolean };
  again?: ImportResult;
}

const cases: Case[] = files.flatMap((file) => {
  const source = readFileSync(join(corpusDir, file), 'utf8');
  const inputs = Object.fromEntries(header(source).inputs!.split(';').map((s) => s.trim().split('=') as [string, string]));
  return modes.map((mode) => ({ name: `${file} (${mode})`, source, inputs, mode }));
});

describe.skipIf(!hasTorch)('import corpus', () => {
  beforeAll(() => {
    const imported = importMany(cases.map((c) => ({ source: c.source, inputShapes: c.inputs, mode: c.mode })));
    cases.forEach((c, i) => {
      c.imported = imported[i];
      if (c.imported?.error) return;
      try {
        c.project = parseProject(c.imported!.project);
        c.analysis = analyzeProject(c.project);
        c.code = generateCode(c.project, c.analysis).code;
      } catch (e) {
        c.failure = String(e);
      }
    });
    const ready = cases.filter((c) => c.code !== undefined);
    const checks = JSON.parse(
      runPython(
        ['-c', EQUIVALENCE],
        JSON.stringify(ready.map((c) => ({ original: c.source, generated: c.code, className: entryModule(c.project!).className, inputs: Object.values(inputShapesOf(c.project!)) }))),
      ),
    );
    const again = importMany(ready.map((c) => ({ source: c.code!, inputShapes: inputShapesOf(c.project!) })));
    ready.forEach((c, i) => {
      c.check = checks[i];
      c.again = again[i];
    });
  }, 600_000);

  afterAll(() => {
    const levels = cases.map((c) => `${c.name.padEnd(36)} ${(c.imported?.report ?? []).map((r) => `${r.className}:${r.level}/${r.codeBlocks}`).join(' ')}`);
    console.log(`\nimport corpus (class:level/code blocks)\n${levels.join('\n')}`);
  });

  for (const c of cases) {
    it(c.name, () => {
      expect(c.imported?.error?.message).toBeUndefined();
      expect(c.failure).toBeUndefined();
      expect(c.analysis!.entry.diagnostics.filter((d) => d.severity === 'error'), JSON.stringify(c.analysis!.entry.diagnostics)).toEqual([]);
      expect(c.check, c.code).toEqual({ ok: true });

      // Reading the generated code back gives the same code.
      expect(c.again?.error?.message).toBeUndefined();
      expect(generateCode(parseProject(c.again!.project)).code).toBe(c.code);
    });
  }
});
