// PRD success metric: torchvision classification models through dynamic import.
// Every model in torchvision.models is imported the way user code would use it
// (self.net = torchvision.models.resnet18()); a model counts as structured when
// no class falls back to code (L1+L3 / L3), no statement becomes a code block, and
// no module with submodules is kept as an untraced external call. The rate is
// printed and must not fall below today's. For a set of smaller models from every
// structured family, the generated code must also load the original weights, give
// the same outputs, and read back as the same code.
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { generateCode, parseProject, type LoomProject } from '../src';
import { canImport, hasTorch, importMany, inputShapesOf, runPython, type ImportResult } from './python';

const hasTorchvision = hasTorch && canImport('torchvision');

/** Structured today; the test fails if fewer are. */
const FLOOR = 60;

/** Checked against the original in PyTorch: one or two per structured family, small enough for CI. */
const EQUIVALENCE_SET = [
  'alexnet',
  'resnet18',
  'resnext50_32x4d',
  'squeezenet1_1',
  'googlenet',
  'inception_v3',
  'mobilenet_v2',
  'mobilenet_v3_small',
  'mnasnet0_5',
  'efficientnet_b0',
  'efficientnet_v2_s',
  'regnet_x_400mf',
  'regnet_y_400mf',
  'convnext_tiny',
];

const source = (name: string) =>
  `import torchvision\nfrom torch import nn\n\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = torchvision.models.${name}()\n\n    def forward(self, x):\n        return self.net(x)\n`;
const inputShape = (name: string) => (name === 'inception_v3' ? 'B,3,299,299' : 'B,3,224,224');

interface Outcome {
  name: string;
  result: ImportResult;
  codeBlocks: number;
  fallback: string[];
  untraced: string[];
  structured: boolean;
}

function outcome(name: string, result: ImportResult): Outcome {
  if (result.error) return { name, result, codeBlocks: 0, fallback: [result.error.message], untraced: [], structured: false };
  const modules = Object.values((result.project as LoomProject).modules);
  const codeBlocks = modules.reduce((n, m) => n + Object.values(m.nodes).filter((node) => node.kind === 'code').length, 0);
  const fallback = (result.report ?? []).filter((r) => r.level === 'L1+L3' || r.level === 'L3').map((r) => `${r.className}:${r.level}`);
  const untraced = (result.warnings ?? []).filter((w) => w.includes('cannot be traced'));
  return { name, result, codeBlocks, fallback, untraced, structured: codeBlocks === 0 && fallback.length === 0 && untraced.length === 0 };
}

// The original (torchvision's own model) and the generated code, weights copied with strict=True.
const EQUIVALENCE = `
import json, sys, torch
results = []
for i, p in enumerate(json.load(sys.stdin)):
    print(f"equivalence {i}", file=sys.stderr, flush=True)
    try:
        def load(name, src):
            module = type(sys)(name)
            sys.modules[name] = module
            exec(src, module.__dict__)
            return module.__dict__["Model"]()
        torch.manual_seed(0)
        a = load(f"tv_original_{i}", p["original"]).eval()
        b = load(f"tv_generated_{i}", p["generated"]).eval()
        missing = sorted(set(a.state_dict()) ^ set(b.state_dict()))
        if missing:
            results.append({"ok": False, "keys": missing[:6]})
            continue
        b.load_state_dict(a.state_dict(), strict=True)
        x = torch.randn([2, 3, p["size"], p["size"]])
        with torch.no_grad():
            ya, yb = a(x), b(x)
        results.append({"ok": bool(ya.shape == yb.shape and torch.allclose(ya, yb, atol=1e-5, rtol=1e-4)), "diff": float((ya - yb).abs().max())})
        del a, b
    except Exception as e:
        results.append({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
print(json.dumps(results))
`;

describe.skipIf(!hasTorchvision)('torchvision classification models (dynamic import)', () => {
  let names: string[] = [];
  let outcomes: Outcome[] = [];
  const checks = new Map<string, { ok: boolean; error?: string; keys?: string[]; diff?: number }>();
  const again = new Map<string, { code: string; regenerated: string }>();

  beforeAll(() => {
    names = JSON.parse(runPython(['-c', 'import json, torchvision.models as m; print(json.dumps(m.list_models(module=m)))'], ''));
    const results = importMany(names.map((name) => ({ source: source(name), inputShapes: { x: inputShape(name) }, mode: 'dynamic', locale: 'en' })));
    outcomes = names.map((name, i) => outcome(name, results[i]!));

    const picked = outcomes.filter((o) => EQUIVALENCE_SET.includes(o.name) && o.structured);
    const generated = picked.map((o) => generateCode(parseProject(o.result.project)).code);
    const verdicts = JSON.parse(
      runPython(['-c', EQUIVALENCE], JSON.stringify(picked.map((o, i) => ({ original: source(o.name), generated: generated[i], size: o.name === 'inception_v3' ? 299 : 224 })))),
    );
    picked.forEach((o, i) => checks.set(o.name, verdicts[i]));
    const reread = importMany(picked.map((o, i) => ({ source: generated[i]!, inputShapes: inputShapesOf(parseProject(o.result.project)) })));
    // Compared without shape comments: shapes observed for external modules (StochasticDepth)
    // are kept in the project, not in the code, so inference cannot follow them after reading back.
    const plain = (p: unknown) => generateCode(parseProject(p), undefined, { export: { shapeComments: false } }).code;
    picked.forEach((o, i) => again.set(o.name, { code: plain(o.result.project), regenerated: reread[i]!.error ? reread[i]!.error!.message : plain(reread[i]!.project) }));
  }, 900_000);

  afterAll(() => {
    if (outcomes.length === 0) return;
    const structured = outcomes.filter((o) => o.structured).length;
    const lines = outcomes.map(
      (o) => `${o.structured ? '✓' : '·'} ${o.name.padEnd(20)} ${o.structured ? '' : [...o.fallback, ...o.untraced.map((u) => u.slice(0, 110)), o.codeBlocks ? `${o.codeBlocks} code blocks` : ''].filter(Boolean).join('; ')}`,
    );
    console.log(`\ntorchvision: ${structured}/${outcomes.length} fully structured (${Math.round((100 * structured) / outcomes.length)}%)\n${lines.join('\n')}`);
  });

  it(`structures at least ${FLOOR} of the classification models`, () => {
    expect(outcomes.length).toBeGreaterThanOrEqual(80);
    expect(outcomes.filter((o) => o.result.error).map((o) => `${o.name}: ${o.result.error!.message}`)).toEqual([]);
    expect(outcomes.filter((o) => o.structured).length).toBeGreaterThanOrEqual(FLOOR);
  });

  for (const name of EQUIVALENCE_SET) {
    it(`${name}: the generated code is the same model`, () => {
      const o = outcomes.find((x) => x.name === name)!;
      expect(o.structured, [...o.fallback, ...o.untraced].join('\n')).toBe(true);
      expect(checks.get(name), JSON.stringify(checks.get(name))).toMatchObject({ ok: true });
      const { code, regenerated } = again.get(name)!;
      expect(regenerated).toBe(code);
    });
  }
});
