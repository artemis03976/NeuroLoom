// Templates are composite round-trip tests: the code NeuroLoom generates for each
// one must behave exactly like the hand-written source it was imported from.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { analyzeModule, entryModule, generateCode } from '@neuroloom/core';
import { describe, expect, it } from 'vitest';
import { hasTorch, inputShapesOf, runPython } from '../../../../packages/core/test/python';
import { TEMPLATES } from './index';

const templatesDir = join(__dirname, '../../templates');

const CHECK = `
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
    inputs.append(torch.randint(0, 50, size) if dtype == "int64" else torch.randn(size))
with torch.no_grad():
    oa, ob = a(*inputs), b(*inputs)
print(json.dumps({"same": torch.allclose(oa, ob), "params": sum(t.numel() for t in a.parameters())}))
`;

describe('templates', () => {
  it('ships six templates with titles in both languages', () => {
    expect(TEMPLATES.map((t) => t.id)).toEqual(['mlp', 'lenet5', 'vgg_cnn', 'unet_tiny', 'lstm_classifier', 'transformer_classifier']);
    for (const t of TEMPLATES) expect(t.title && t.titleZh && t.description && t.descriptionZh).toBeTruthy();
  });

  for (const t of TEMPLATES) {
    it(`${t.id} analyzes cleanly`, () => {
      const analysis = analyzeModule(entryModule(t.project));
      expect(analysis.diagnostics).toEqual([]);
      expect(generateCode(t.project).todos).toBe(0);
    });

    it.skipIf(!hasTorch)(`${t.id} generates code equivalent to its source`, { timeout: 60_000 }, () => {
      const original = readFileSync(join(templatesDir, `${t.id}.py`), 'utf8');
      const result = JSON.parse(
        runPython(
          ['-c', CHECK],
          JSON.stringify({
            original,
            generated: generateCode(t.project).code,
            className: entryModule(t.project).className,
            inputs: Object.values(inputShapesOf(t.project)),
          }),
        ),
      );
      expect(result.same).toBe(true);
      expect(analyzeModule(entryModule(t.project)).totalParams).toBe(result.params);
    });
  }
});
