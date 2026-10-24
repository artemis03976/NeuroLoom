// Round-trip gate (PRD 11.5): graph -> code -> graph and code -> graph -> code,
// through the real Python backend. Skipped when no Python with torch is available.
import { describe, expect, it } from 'vitest';
import {
  analyzeModule,
  analyzeProject,
  concretize,
  diffSignatures,
  entryModule,
  generateCode,
  parseProject,
  structuralSignature,
  type LoomProject,
} from '../src';
import { ROUND_TRIP_FIXTURES, resnet, sequentialHead, withCodeBlock } from './fixtures';
import { hasTorch, importCode, inputShapesOf, runPython, shapePairs } from './python';

const NO_DIFF = { onlyInA: [], onlyInB: [] };

// Runs generated code once: output shapes and parameter count (symbolic dimensions = 2).
const RUN = `
import json, sys, torch
p = json.load(sys.stdin)
ns = {"__name__": "m"}
exec(p["code"], ns)
model = ns[p["className"]]()
model.eval()
inputs = []
for spec in p["inputs"]:
    dims, _, dtype = spec.partition(":")
    size = [int(d) if d.isdigit() else 2 for d in dims.split(",")]
    inputs.append(torch.randint(0, 10, size) if dtype == "int64" else torch.randn(size))
with torch.no_grad():
    out = model(*inputs)
out = out if isinstance(out, tuple) else (out,)
print(json.dumps({"shapes": [list(o.shape) for o in out], "params": sum(t.numel() for t in model.parameters())}))
`;

function runModel(project: LoomProject, code: string): { shapes: number[][]; params: number } {
  const input = JSON.stringify({ code, className: entryModule(project).className, inputs: Object.values(inputShapesOf(project)) });
  return JSON.parse(runPython(['-c', RUN], input));
}

describe.skipIf(!hasTorch)('graph -> code -> graph', () => {
  const fixtures: Record<string, () => LoomProject> = { ...ROUND_TRIP_FIXTURES, resnet, sequentialHead, withCodeBlock };
  for (const [name, make] of Object.entries(fixtures)) {
    it(name, { timeout: 60_000 }, () => {
      const original = make();
      const pa = analyzeProject(original);
      expect(pa.entry.diagnostics).toEqual([]);
      const { code } = generateCode(original, pa);

      // Static import: the same modules, structure, names and expressions.
      const imported = parseProject(importCode(code, inputShapesOf(original)).project);
      expect(diffSignatures(structuralSignature(original), structuralSignature(imported))).toEqual(NO_DIFF);
      // Regenerating from the imported graph gives byte-identical code. Declared
      // code-block shapes are editor annotations (like layout), so the shape
      // comments they lead to are not part of the comparison.
      const declared = Object.values(original.modules).some((m) => Object.values(m.nodes).some((n) => n.shapes));
      const comparable = (text: string) => (declared ? text.replace(/ {2}# \[.*\]$/gm, '') : text);
      expect(comparable(generateCode(imported).code)).toBe(comparable(code));

      // The code runs, and PyTorch agrees with the inferred output shapes and parameter count.
      const run = runModel(original, code);
      expect(run.shapes).toEqual(pa.entry.outputs.map((t) => concretize(t!.shape, 2)));
      expect(run.params).toBe(pa.entry.totalParams);
    });
  }
});

describe.skipIf(!hasTorch)('tracing agrees with shape inference', () => {
  for (const [name, make] of Object.entries(ROUND_TRIP_FIXTURES)) {
    it(name, { timeout: 60_000 }, () => {
      const original = make();
      const { code } = generateCode(original);
      const result = importCode(code, inputShapesOf(original), 'trace');
      const imported = parseProject(result.project);
      expect(diffSignatures(structuralSignature(original), structuralSignature(imported))).toEqual(NO_DIFF);
      for (const pair of shapePairs(imported, analyzeModule(imported), result)) expect(pair.ts, pair.label).toEqual(pair.torch);
    });
  }
});

// Handwritten code in styles NeuroLoom does not emit itself.
const HANDWRITTEN: Record<string, { source: string; inputs?: Record<string, string>; warnings?: number }> = {
  nestedCalls: {
    source: `import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
`,
  },
  methodCallsAndModules: {
    source: `import torch
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
  },
  // Regression: after auto-layout put the second output left of the first,
  // the return tuple came out swapped.
  tupleOrder: {
    source: `import torch.nn.functional as F
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
  },
  // Submodules, hyperparameters, an __init__ local, a ModuleList loop and an nn.Sequential head.
  transformerish: {
    inputs: { tokens: 'B,12:int64' },
    source: `import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim, hidden_ratio=4, dropout=0.1):
        super().__init__()
        hidden = dim * hidden_ratio
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc2(F.gelu(self.fc1(self.norm(x))))
        return x + self.drop(h)


class Encoder(nn.Module):
    def __init__(self, vocab=100, dim=32, depth=3, num_classes=5):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, tokens):
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x.mean(1))
`,
  },
  // Module-level constants, a plain attribute, weight init and a statement kept as a code block.
  codeBlocks: {
    inputs: { x: 'B,3,16,16' },
    source: `import math
import torch
from torch import nn

WIDTH = 8


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH * 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.scale = 1 / math.sqrt(WIDTH)
        self.fc = nn.Linear(WIDTH * 2 * 8 * 8, num_classes)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x)) * self.scale
        x = x.view(x.size(0), -1)
        return self.fc(x)
`,
  },
  // A ResNet-style stage whose first copy differs from the rest.
  firstCopyDiffers: {
    inputs: { x: 'B,16' },
    source: `import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.fc = nn.Linear(cin, cout)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class Stack(nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        self.layers = nn.ModuleList([Block(16, 32)] + [Block(32, 32) for _ in range(depth - 1)])
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)
`,
  },
  // A parameter read before a call, and a submodule forward() never uses: both keep
  // their place in __init__ when the generated code is read back.
  stateAndUnused: {
    inputs: { x: 'B,8' },
    warnings: 1, // self.spare: not used in forward()
    source: `import torch
from torch import nn


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.spare = nn.Linear(8, 8)
        self.scale = nn.Parameter(torch.ones(8))
        self.register_buffer("offset", torch.zeros(8))

    def forward(self, x):
        return self.scale * self.fc(x) + self.offset
`,
  },
};

/** Places nodes in reverse creation order, so every layout-based tie-break flips. */
function scrambleLayout(p: LoomProject): LoomProject {
  const modules = Object.fromEntries(
    Object.entries(p.modules).map(([id, mod]) => {
      const ids = Object.keys(mod.nodes);
      const layout = Object.fromEntries(ids.map((n, i) => [n, { x: (ids.length - i) * 100, y: (ids.length - i) * 37 }]));
      return [id, { ...mod, layout }];
    }),
  );
  return { ...p, modules };
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
a.eval(); b.eval()
b.load_state_dict(a.state_dict(), strict=True)
inputs = []
for spec in p["inputs"]:
    dims, _, dtype = spec.partition(":")
    size = [int(d) if d.isdigit() else 3 for d in dims.split(",")]
    inputs.append(torch.randint(0, 10, size) if dtype == "int64" else torch.randn(size))
with torch.no_grad():
    oa, ob = a(*inputs), b(*inputs)
oa = oa if isinstance(oa, tuple) else (oa,)
ob = ob if isinstance(ob, tuple) else (ob,)
print(json.dumps({"ok": len(oa) == len(ob) and all(torch.allclose(x, y) for x, y in zip(oa, ob))}))
`;

describe.skipIf(!hasTorch)('code -> graph -> code', () => {
  for (const [name, { source, inputs, warnings }] of Object.entries(HANDWRITTEN)) {
    it(name, { timeout: 60_000 }, () => {
      // A scrambled layout may reorder statements, but must not change behaviour.
      const first = scrambleLayout(parseProject(importCode(source, inputs).project));
      const generated = generateCode(first).code;
      const diagnostics = analyzeModule(first).diagnostics;
      expect(diagnostics.filter((d) => d.severity === 'error'), generated).toEqual([]);
      expect(diagnostics.filter((d) => d.severity === 'warning').length, generated).toBe(warnings ?? 0);

      // Semantically equivalent: same state_dict keys, same outputs for the same weights.
      const check = JSON.parse(
        runPython(
          ['-c', EQUIVALENCE_CHECK],
          JSON.stringify({ original: source, generated, className: entryModule(first).className, inputs: Object.values(inputShapesOf(first)) }),
        ),
      );
      expect(check, generated).toEqual({ ok: true });

      // And the generated code is a fixed point: importing it again changes nothing.
      const second = parseProject(importCode(generated, inputShapesOf(first)).project);
      expect(generateCode(second).code).toBe(generated);
    });
  }
});
