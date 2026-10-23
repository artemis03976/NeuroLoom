import { describe, expect, it } from 'vitest';
import {
  analyzeModule,
  connect,
  copyNodes,
  parseClip,
  pasteClip,
  entryModule,
  generateCode,
  literal,
  parseProject,
  parseProjectText,
  parseShape,
  serializeProject,
  renameNode,
  setLayout,
  setParam,
} from '../src';
import { mlp, mlpFunctional, twoHeads, twoInputs } from './fixtures';

describe('parseShape', () => {
  it('accepts symbols, numbers and optional brackets', () => {
    expect(parseShape('[B, 3, 224, 224]')).toEqual({ ok: true, shape: ['B', 3, 224, 224] });
    expect(parseShape('B,784')).toEqual({ ok: true, shape: ['B', 784] });
  });
  it('rejects zero and junk', () => {
    expect(parseShape('B, 0').ok).toBe(false);
    expect(parseShape('B, 3.5').ok).toBe(false);
    expect(parseShape('').ok).toBe(false);
  });
});

describe('analyzeModule', () => {
  it('propagates shapes and resolves auto parameters', () => {
    const mod = entryModule(mlp());
    const a = analyzeModule(mod);
    expect(a.diagnostics).toEqual([]);
    expect(a.nodes.fc1?.outputs?.[0]?.shape).toEqual(['B', 256]);
    expect(a.nodes.fc2?.resolved).toEqual({ in_features: 256, out_features: 10, bias: true });
    expect(a.edgeTypes.e3?.shape).toEqual(['B', 10]);
    expect(a.totalParams).toBe(784 * 256 + 256 + 256 * 10 + 10);
  });

  it('reports a feature-count mismatch on the offending node', () => {
    const mod = setParam(entryModule(mlp()), 'fc2', 'in_features', literal(128));
    const errors = analyzeModule(mod).diagnostics.filter((d) => d.severity === 'error');
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatchObject({ nodeId: 'fc2' });
    expect(errors[0]!.message).toContain('in_features = 128');
  });

  it('does not cascade an upstream failure', () => {
    const mod = setParam(entryModule(mlp()), 'x', 'shape', literal('B, T'));
    const errors = analyzeModule(mod).diagnostics.filter((d) => d.severity === 'error');
    expect(errors.map((e) => e.nodeId)).toEqual(['fc1']);
  });

  it('warns about values nothing consumes', () => {
    let mod = entryModule(mlp());
    mod = { ...mod, edges: mod.edges.filter((e) => e.id !== 'e3') };
    const warnings = analyzeModule(mod).diagnostics.filter((d) => d.severity === 'warning');
    expect(warnings.map((w) => `${w.nodeId}: ${w.message}`)).toContain('fc2: 输出未被使用');
  });

  it('flags unconnected inputs and invalid names', () => {
    let mod = entryModule(mlp());
    mod = { ...mod, edges: mod.edges.filter((e) => e.id !== 'e1') };
    mod = renameNode(mod, 'fc2', 'class');
    const messages = analyzeModule(mod).diagnostics.map((d) => `${d.nodeId}: ${d.message}`);
    expect(messages).toContain('act1: 输入端口 in 未连接');
    expect(messages.some((m) => m.startsWith('fc2:') && m.includes('不是合法'))).toBe(true);
  });
});

describe('generateCode', () => {
  it('writes readable, runnable PyTorch for an MLP', () => {
    expect(generateCode(mlp()).code).toBe(`import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)  # [B, 256]
        x = self.act1(x)  # [B, 256]
        return self.fc2(x)  # [B, 10]
`);
  });

  it('uses F for functional ops and only writes non-default keywords', () => {
    const { code } = generateCode(mlpFunctional());
    expect(code).toContain('import torch.nn.functional as F\nfrom torch import nn');
    expect(code).toContain('x = F.relu(x)  # [B, 256]');
    expect(code).toContain('self.fc2 = nn.Linear(256, 10, bias=False)');
  });

  it('keeps values used twice in their own variable and returns a tuple', () => {
    const { code } = generateCode(twoHeads());
    expect(code).toContain('def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:');
    expect(code).toContain('        x = self.fc(x)  # [B, 64]\n        x = self.act(x)  # [B, 64]\n');
    expect(code).toContain('        head1 = self.head1(x)  # [B, 10]\n        head2 = self.head2(x)  # [B, 1]\n');
    expect(code).toContain('        return head1, head2\n');
  });

  it('keeps forward arguments and return values in interface order, whatever the layout', () => {
    const inputs = twoInputs();
    inputs.modules.main = setLayout(entryModule(inputs), { a: { x: 0, y: 300 }, b: { x: 0, y: 0 } });
    expect(generateCode(inputs).code).toContain('def forward(self, a: torch.Tensor, b: torch.Tensor)');

    const heads = twoHeads();
    heads.modules.main = setLayout(entryModule(heads), { output: { x: 900, y: 300 }, output_1: { x: 600, y: 0 } });
    expect(generateCode(heads).code).toContain('        return head1, head2\n');
  });

  it('maps each node to its constructor and forward lines', () => {
    const { code, lineMap } = generateCode(mlp());
    const lines = code.split('\n');
    expect(lineMap.fc1!.map((n) => lines[n - 1]!.trim())).toEqual(['self.fc1 = nn.Linear(784, 256)', 'x = self.fc1(x)  # [B, 256]']);
  });

  it('writes shape errors next to the offending call', () => {
    const p = mlp();
    p.modules.main = setParam(entryModule(p), 'fc2', 'in_features', literal(64));
    const { code, todos } = generateCode(p);
    expect(todos).toBe(1);
    expect(code).toContain('return self.fc2(x)  # TODO(neuroloom): 输入最后一维为 256，但 in_features = 64');
  });

  it('marks unfinished graphs with TODO comments instead of failing', () => {
    let mod = entryModule(mlp());
    mod = { ...mod, edges: mod.edges.filter((e) => e.id !== 'e0') };
    const p = mlp();
    p.modules.main = mod;
    const { code, todos } = generateCode(p);
    expect(todos).toBeGreaterThan(0);
    expect(code).toContain('TODO(neuroloom)');
  });
});

describe('editing', () => {
  it('refuses connections that would form a cycle', () => {
    const mod = entryModule(mlp());
    expect(connect(mod, { node: 'fc2', port: 'out' }, { node: 'fc1', port: 'in' })).toBeNull();
  });
  it('replaces the existing edge on an input port', () => {
    const mod = connect(entryModule(mlp()), { node: 'x', port: 'out' }, { node: 'act1', port: 'in' })!;
    expect(mod.edges.filter((e) => e.target.node === 'act1')).toHaveLength(1);
    expect(mod.edges.find((e) => e.target.node === 'act1')!.source.node).toBe('x');
  });
});

describe('.loom serialization', () => {
  it('is byte-stable regardless of key insertion order', () => {
    const p = mlp();
    const reordered = JSON.parse(JSON.stringify(p), (_k, v) =>
      v && typeof v === 'object' && !Array.isArray(v) ? Object.fromEntries(Object.entries(v).reverse()) : v,
    );
    expect(serializeProject(reordered)).toBe(serializeProject(p));
    expect(serializeProject(p).endsWith('}\n')).toBe(true);
  });

  it('round-trips without changing the generated code', () => {
    const p = twoHeads();
    const loaded = parseProjectText(serializeProject(p));
    expect(generateCode(loaded).code).toBe(generateCode(p).code);
  });

  it('keeps the return order even though node ids sort the other way', () => {
    const p = twoHeads();
    const mod = entryModule(p);
    // Rename ids so the second output sorts first alphabetically.
    const text = serializeProject(p).replaceAll('"output_1"', '"a_out"').replaceAll('"output"', '"z_out"');
    const loaded = parseProjectText(text);
    expect(entryModule(loaded).interface.outputs).toEqual(['z_out', 'a_out']);
    expect(generateCode(loaded).code).toContain('return head1, head2');
    expect(mod.interface.outputs).toEqual(['output', 'output_1']);
  });

  it('migrates 0.1 files, deriving the interface from node order', () => {
    const v01 = JSON.parse(JSON.stringify(twoInputs()));
    v01.schemaVersion = '0.1';
    delete v01.modules.main.interface;
    const loaded = parseProject(v01);
    expect(loaded.schemaVersion).toBe('0.2');
    expect(entryModule(loaded).interface).toEqual({ inputs: ['a', 'b'], outputs: ['output', 'output_1'] });
  });

  it('repairs an interface that lists missing nodes or omits some', () => {
    const p = JSON.parse(JSON.stringify(twoInputs()));
    p.modules.main.interface = { inputs: ['b', 'ghost'], outputs: [] };
    expect(entryModule(parseProject(p)).interface).toEqual({ inputs: ['b', 'a'], outputs: ['output', 'output_1'] });
  });

  it('rejects files from a newer schema', () => {
    const p = JSON.parse(JSON.stringify(mlp()));
    p.schemaVersion = '9.0';
    expect(() => parseProject(p)).toThrow('9.0');
  });
});

describe('parseProject', () => {
  it('accepts a valid project and rejects unknown node types', () => {
    expect(() => parseProject(JSON.parse(JSON.stringify(mlp())))).not.toThrow();
    const bad = JSON.parse(JSON.stringify(mlp()));
    bad.modules.main.nodes.fc1.type = 'torch.nn.Mystery';
    expect(() => parseProject(bad)).toThrow('torch.nn.Mystery');
  });
});

describe('clipboard', () => {
  it('copies nodes with the edges between them and pastes them with fresh names', () => {
    const mod = entryModule(mlp());
    const clip = parseClip(JSON.stringify(copyNodes(mod, ['fc1', 'act1'])))!;
    expect(clip.nodes.map((n) => n.name)).toEqual(['fc1', 'act1']);
    expect(clip.edges).toHaveLength(1);

    const { mod: pasted, nodeIds } = pasteClip(mod, clip, { x: 40, y: 40 });
    const names = nodeIds.map((id) => pasted.nodes[id]!.name);
    expect(names).toEqual(['fc3', 'act2']);
    expect(pasted.edges.filter((e) => nodeIds.includes(e.source.node) && nodeIds.includes(e.target.node))).toHaveLength(1);
    expect(pasted.layout[nodeIds[0]!]).toEqual({ x: mod.layout.fc1!.x + 40, y: mod.layout.fc1!.y + 40 });
  });

  it('keeps names that are free and adds pasted inputs to the interface', () => {
    const source = entryModule(mlp());
    const target = entryModule(twoInputs());
    const { mod, nodeIds } = pasteClip(target, copyNodes(source, ['x', 'fc2']), { x: 0, y: 0 });
    expect(nodeIds.map((id) => mod.nodes[id]!.name)).toEqual(['x', 'fc2']);
    expect(mod.interface.inputs).toEqual(['a', 'b', nodeIds[0]]);
  });

  it('ignores clipboard text that is not a NeuroLoom clip', () => {
    expect(parseClip('hello')).toBeNull();
    expect(parseClip('{"kind":"other"}')).toBeNull();
  });
});
