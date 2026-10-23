// "Run once to fill in shapes", end to end through the real Python backend:
// static import, instrumented code, two runs, symbolic shapes back.
import { analyzeProject, dimFromRuns, parseProject } from '@neuroloom/core';
import { describe, expect, it } from 'vitest';
import { hasTorch, importCode, runPython } from '../../../../packages/core/test/python';
import { observedShapes, probeRequest, type ProbeResult } from './probe';

describe('dimensions from two runs', () => {
  const first = { B: 2, T: 3 };
  const second = { B: 3, T: 5 };
  it('tells constants from products of symbols', () => {
    expect(dimFromRuns(256, 256, first, second)).toBe(256);
    expect(dimFromRuns(2, 3, first, second)).toBe('B');
    expect(dimFromRuns(6, 15, first, second)).toBe('B*T');
    expect(dimFromRuns(8, 12, first, second)).toBe('B*4');
    expect(dimFromRuns(5, 8, first, second)).toBeUndefined(); // B + 3 is not a product
  });
});

const SOURCE = `import torch
from torch import nn


class Gate(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x * torch.sigmoid(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.gate = Gate(4)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.gate(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
`;

describe.skipIf(!hasTorch)('running the model to fill in shapes', () => {
  it('declares the shapes of code blocks and opaque modules', { timeout: 60_000 }, () => {
    const project = parseProject(importCode(SOURCE, { x: 'B,3,8,8' }).project);
    // Before: the opaque Gate and the x.view(...) block hide every shape after the conv.
    expect(analyzeProject(project).entry.outputs[0]).toBeUndefined();

    const request = probeRequest(project);
    const result = JSON.parse(runPython(['-m', 'neuroloom', 'probe'], JSON.stringify(request))) as ProbeResult;
    const observed = observedShapes(result);
    expect(observed.map((o) => o.text).sort()).toEqual(['B, 256', 'B, 4, 8, 8']);

    // After: declared on the nodes, inference reaches the output again.
    const net = project.modules.Net!;
    for (const o of observed) {
      const node = project.modules[o.moduleId]!.nodes[o.nodeId]!;
      node.shapes = { ...node.shapes, [o.port]: o.text };
    }
    const after = analyzeProject(project);
    expect(after.entry.outputs[0]?.shape).toEqual(['B', 10]);
    expect(after.entry.diagnostics).toEqual([]);
    expect(Object.values(net.nodes).some((n) => n.kind === 'code')).toBe(true);
  });
});
