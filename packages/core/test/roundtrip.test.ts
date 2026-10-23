// M0 gate: graph -> code -> graph and code -> graph -> code, through the real
// Python backend. Skipped when no Python with torch is available.
import { describe, expect, it } from 'vitest';
import {
  analyzeModule,
  diffSignatures,
  entryModule,
  generateCode,
  parseProject,
  parseShape,
  structuralSignature,
  type LoomProject,
} from '../src';
import { ROUND_TRIP_FIXTURES } from './fixtures';
import { hasTorch, importCode, inputShapesOf, runPython, shapePairs } from './python';

describe.skipIf(!hasTorch)('graph -> code -> graph', () => {
  for (const [name, make] of Object.entries(ROUND_TRIP_FIXTURES)) {
    it(name, { timeout: 60_000 }, () => {
      const original = make();
      const mod = entryModule(original);
      const analysis = analyzeModule(mod);
      expect(analysis.diagnostics).toEqual([]);
      const { code } = generateCode(original, { [mod.id]: analysis });

      const result = importCode(code, inputShapesOf(original));
      const imported = parseProject(result.project);
      const mod2 = entryModule(imported);
      const analysis2 = analyzeModule(mod2);

      // Same structure, parameters and names.
      expect(diffSignatures(structuralSignature(mod, analysis), structuralSignature(mod2, analysis2))).toEqual({
        onlyInA: [],
        onlyInB: [],
      });
      // Regenerating from the imported graph gives byte-identical code.
      expect(generateCode(imported).code).toBe(code);
      // TypeScript shape inference agrees with PyTorch running the model.
      for (const pair of shapePairs(imported, analysis2, result)) expect(pair.ts, pair.label).toEqual(pair.torch);
    });
  }
});

// Handwritten code in styles NeuroLoom does not emit itself (nested calls, x.relu()).
const HANDWRITTEN: Record<string, string> = {
  nestedCalls: `import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
`,
  methodCallsAndModules: `import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 64)
        self.act = nn.ReLU(inplace=True)
        self.mu = nn.Linear(64, 8)
        self.logvar = nn.Linear(64, 8)

    def forward(self, x: torch.Tensor):
        h = self.act(self.proj(x)).relu()
        return self.mu(h), self.logvar(h)
`,
  // Regression: after auto-layout put the second output left of the first,
  // the return tuple came out swapped.
  tupleOrder: `import torch.nn.functional as F
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(16, 4)
        self.dec = nn.Linear(4, 16)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(F.relu(z)), z
`,
};

/** Places nodes in reverse creation order, so every layout-based tie-break flips. */
function scrambleLayout(p: LoomProject): LoomProject {
  const mod = entryModule(p);
  const ids = Object.keys(mod.nodes);
  const layout = Object.fromEntries(ids.map((id, i) => [id, { x: (ids.length - i) * 100, y: (ids.length - i) * 37 }]));
  return { ...p, modules: { ...p.modules, [mod.id]: { ...mod, layout } } };
}

// Loads both sources, copies weights with strict=True and compares outputs.
const EQUIVALENCE_CHECK = `
import json, sys, torch
p = json.load(sys.stdin)
def load(src):
    ns = {"__name__": "m"}
    exec(src, ns)
    return ns[p["className"]]()
torch.manual_seed(0)
a, b = load(p["original"]), load(p["generated"])
b.load_state_dict(a.state_dict(), strict=True)
inputs = [torch.randn(*s) for s in p["shapes"]]
oa, ob = a(*inputs), b(*inputs)
oa = oa if isinstance(oa, tuple) else (oa,)
ob = ob if isinstance(ob, tuple) else (ob,)
print(json.dumps({"ok": len(oa) == len(ob) and all(torch.allclose(x, y) for x, y in zip(oa, ob))}))
`;

describe.skipIf(!hasTorch)('code -> graph -> code', () => {
  for (const [name, source] of Object.entries(HANDWRITTEN)) {
    it(name, { timeout: 60_000 }, () => {
      // A scrambled layout may reorder statements, but must not change behaviour.
      const first = scrambleLayout(parseProject(importCode(source).project));
      const generated = generateCode(first).code;
      expect(analyzeModule(entryModule(first)).diagnostics).toEqual([]);

      // Semantically equivalent: same state_dict keys, same outputs for the same weights.
      const shapes = Object.values(inputShapesOf(first)).map((text) => {
        const parsed = parseShape(text);
        if (!parsed.ok) throw new Error(parsed.error);
        return parsed.shape.map((d) => (typeof d === 'number' ? d : 3));
      });
      const check = JSON.parse(
        runPython(
          ['-c', EQUIVALENCE_CHECK],
          JSON.stringify({ original: source, generated, className: entryModule(first).className, shapes }),
        ),
      );
      expect(check).toEqual({ ok: true });

      // And the generated code is a fixed point: importing it again changes nothing.
      const second = parseProject(importCode(generated, inputShapesOf(first)).project);
      expect(generateCode(second).code).toBe(generated);
    });
  }
});
