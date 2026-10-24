// Templates and blocks are composite round-trip tests: the code NeuroLoom
// generates for each one must behave exactly like the hand-written source it was
// imported from.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { analyzeModule, analyzeProject, costOf, generateCode, literalOf, symbolsOf, type LoomProject, type ModuleDef } from '@neuroloom/core';
import { beforeAll, describe, expect, it } from 'vitest';
import { hasTorch, inputArg, runPython } from '../../../../packages/core/test/python';
import { BLOCKS, TEMPLATES } from './index';

const studioDir = join(__dirname, '../..');

// One Python process checks every model: load both sources, copy the weights with
// strict=True, run both in eval mode on the same inputs and the same random noise
// (the VAE samples), compare every output and count parameters.
const CHECK = `
import gc, json, sys, torch
from torch.utils.flop_counter import FlopCounterMode
results = []
for p in json.load(sys.stdin):
    try:
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
            size = [int(d) if d.isdigit() else 2 for d in dims.split(",")]
            inputs.append(torch.randint(0, 50, size) if dtype == "int64" else torch.randn(size))
        def run(model):
            torch.manual_seed(1)
            with torch.no_grad():
                out = model(*inputs)
            return list(out) if isinstance(out, tuple) else [out]
        oa, ob = run(a), run(b)
        # FLOPs as F8.2 counts them: one forward pass in training mode, like flops.test.ts.
        a.train()
        with torch.backends.mkldnn.flags(enabled=False), FlopCounterMode(display=False) as counter:
            run(a)
        count = lambda m: sum(t.numel() for t in m.parameters())
        results.append({
            "same": len(oa) == len(ob) and all(x.shape == y.shape and torch.allclose(x, y) for x, y in zip(oa, ob)),
            "params": count(a),
            "flops": counter.get_total_flops(),
            "generatedParams": count(b),
            # nn.Parameter attributes of the model's own classes (ViT's class token and
            # position embeddings): kept as constants, not counted by NeuroLoom.
            "direct": sum(t.numel() for m in a.modules() if type(m).__module__ == "m" for t in m.parameters(recurse=False)),
        })
        del a, b, oa, ob
    except Exception as e:
        results.append({"error": f"{type(e).__name__}: {e}"})
    gc.collect()
print(json.dumps(results))
`;

interface Item {
  id: string;
  project: LoomProject;
}

interface Check {
  same: boolean;
  params: number;
  flops: number;
  generatedParams: number;
  direct: number;
  error?: string;
}

/** The models in a project: the entry, and other modules whose inputs declare shapes (DCGAN's discriminator). */
function models(p: LoomProject): ModuleDef[] {
  return Object.values(p.modules).filter(
    (m) =>
      m.id === p.entry ||
      (!m.sequential && !m.opaque && m.interface.inputs.length > 0 && m.interface.inputs.every((id) => String(literalOf(m.nodes[id]!.params.shape) ?? '').trim() !== '')),
  );
}

const inputsOf = (m: ModuleDef): string[] =>
  m.interface.inputs.map((id) => inputArg(String(literalOf(m.nodes[id]!.params.shape)), literalOf(m.nodes[id]!.params.dtype) as string | undefined));

function suite(kind: 'templates' | 'blocks', items: Item[]) {
  const cases = items.flatMap((item) => models(item.project).map((mod) => ({ item, mod, key: `${item.id}/${mod.className}` })));
  let checks: Record<string, Check> = {};

  beforeAll(() => {
    if (!hasTorch) return;
    const requests = cases.map(({ item, mod }) => ({
      original: readFileSync(join(studioDir, kind, `${item.id}.py`), 'utf8'),
      generated: generateCode(item.project).code,
      className: mod.className,
      inputs: inputsOf(mod),
    }));
    const results = JSON.parse(runPython(['-c', CHECK], JSON.stringify(requests))) as Check[];
    checks = Object.fromEntries(cases.map((c, i) => [c.key, results[i]!]));
  }, 600_000);

  for (const item of items) {
    it(`${item.id} analyzes cleanly, with every shape known`, () => {
      const pa = analyzeProject(item.project);
      for (const analysis of Object.values(pa.byModule).flat()) expect(analysis.diagnostics).toEqual([]);
      for (const mod of models(item.project)) {
        const analysis = analyzeModule(item.project, mod.id);
        expect(analysis.diagnostics).toEqual([]);
        expect(analysis.outputs.length).toBeGreaterThan(0);
        expect(analysis.outputs.every(Boolean)).toBe(true);
      }
      expect(generateCode(item.project).todos).toBe(0);
    });
  }

  for (const { item, mod, key } of cases) {
    it.skipIf(!hasTorch)(`${key} generates code equivalent to its source`, () => {
      const check = checks[key]!;
      expect(check.error).toBeUndefined();
      expect(check.same).toBe(true);
      expect(check.generatedParams).toBe(check.params);
      expect(analyzeModule(item.project, mod.id).totalParams).toBe(check.params - check.direct);
    });

    it.skipIf(!hasTorch)(`${key} counts the FLOPs PyTorch counts`, () => {
      const analysis = analyzeModule(item.project, mod.id);
      // Every symbolic dimension is 2 in the Python run.
      const cost = costOf(item.project, analysis, Object.fromEntries(symbolsOf(analysis).map((s) => [s, 2])));
      // Code blocks are not counted: then the editor's figure is a lower bound.
      if (cost.partial) expect(cost.flops).toBeLessThanOrEqual(checks[key]!.flops);
      else expect(cost.flops).toBe(checks[key]!.flops);
    });
  }
}

describe('templates', () => {
  it('ships the twelve templates of PRD 12.2, simple to complex, with titles in both languages and a paper', () => {
    expect(TEMPLATES.map((t) => t.id)).toEqual([
      'mlp',
      'lenet5',
      'vae',
      'vgg11',
      'dcgan',
      'resnet18',
      'resnet50',
      'unet',
      'lstm_classifier',
      'transformer_classifier',
      'vit_tiny',
      'gpt2_small',
    ]);
    for (const t of TEMPLATES) {
      expect(t.title && t.titleZh && t.description && t.descriptionZh).toBeTruthy();
      expect(t.paper).toMatch(/^https:\/\//);
    }
  });

  it('keeps both DCGAN models in one project, the generator as the entry', () => {
    const dcgan = TEMPLATES.find((t) => t.id === 'dcgan')!;
    expect(dcgan.project.modules[dcgan.project.entry]!.className).toBe('Generator');
    expect(models(dcgan.project).map((m) => m.className)).toEqual(['Generator', 'Discriminator']);
  });

  suite('templates', TEMPLATES);
});

describe('blocks', () => {
  it('ships the composite modules of PRD 9.6 with titles in both languages', () => {
    expect(BLOCKS.map((b) => [b.id, b.project.modules[b.project.entry]!.className])).toEqual([
      ['residual_block', 'ResidualBlock'],
      ['bottleneck', 'Bottleneck'],
      ['se_block', 'SEBlock'],
      ['multi_head_attention', 'MultiHeadAttention'],
      ['transformer_block', 'TransformerBlock'],
      ['unet_down', 'Down'],
      ['unet_up', 'Up'],
    ]);
    for (const b of BLOCKS) expect(b.title && b.titleZh && b.description && b.descriptionZh).toBeTruthy();
  });

  suite('blocks', BLOCKS);
});
