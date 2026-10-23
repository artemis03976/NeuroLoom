// M2: hyperparameters and expressions, modules and instances, repeats,
// nn.Sequential, code blocks, and the editing operations that build them.
import { describe, expect, it } from 'vitest';
import {
  analyzeModule,
  analyzeProject,
  deleteModule,
  diffSignatures,
  entryModule,
  evalExpr,
  freeNames,
  generateCode,
  makeUnique,
  packAsModule,
  parseProjectText,
  renameInExpr,
  serializeProject,
  setHyperparams,
  setParam,
  structuralSignature,
  substituteExpr,
  subtractOne,
  unpackModule,
  withModule,
  literal,
  type LoomProject,
} from '../src';
import { editEntry, mlp, moduleOf, projectOf, resnet, sequentialHead, withCodeBlock } from './fixtures';

describe('expressions', () => {
  it('evaluates the Python subset used in parameters', () => {
    const env = { d_model: 512, nhead: 8, use_bias: true, ks: [3, 5] };
    const value = (text: string) => {
      const r = evalExpr(text, env);
      return r.ok ? r.value : r.error;
    };
    expect(value('d_model * 4')).toBe(2048);
    expect(value('d_model // nhead')).toBe(64);
    expect(value('-7 // 2')).toBe(-4);
    expect(value('-7 % 3')).toBe(2);
    expect(value('2 ** -1')).toBe(0.5);
    expect(value('-2 ** 2')).toBe(-4);
    expect(value('max(1, d_model // 1024)')).toBe(1);
    expect(value('64 if use_bias else 32')).toBe(64);
    expect(value('(d_model, nhead)')).toEqual([512, 8]);
    expect(value('ks[1]')).toBe(5);
    expect(value('int(math.sqrt(d_model // 2))')).toBe(16);
    expect(value('1 < nhead <= 8')).toBe(true);
    expect(value('not use_bias or "x"')).toBe('x');
  });

  it('reports undefined names, unknown values and syntax errors', () => {
    expect(evalExpr('width * 2', {})).toMatchObject({ ok: false, error: expect.stringContaining('width') });
    expect(evalExpr('width * 2', { width: undefined })).toMatchObject({ ok: false, unknown: true });
    expect(evalExpr('d_model *', { d_model: 1 }).ok).toBe(false);
    expect(evalExpr('__import__("os")', {}).ok).toBe(false);
  });

  it('lists, renames and substitutes names, keeping the text as written', () => {
    expect(freeNames('max(1, c // r) + math.floor(x)')).toEqual(['c', 'r', 'x']);
    expect(renameInExpr('d_model*4 + f(d_model=d_model)', (n) => (n === 'd_model' ? 'self.d_model' : undefined))).toBe(
      'self.d_model*4 + f(d_model=self.d_model)',
    );
    expect(substituteExpr('in_ch * 2', { in_ch: 'a + b' })).toBe('(a + b) * 2');
    expect(substituteExpr('x ** -y', { y: '1' })).toBe('x ** -1');
    expect(subtractOne('depth')).toBe('depth - 1');
    expect(subtractOne('a if c else b')).toBe('(a if c else b) - 1');
  });
});

describe('modules in context', () => {
  it('analyzes each instance with its own hyperparameters and input shapes', () => {
    const pa = analyzeProject(resnet());
    expect(pa.entry.diagnostics).toEqual([]);
    const blocks = pa.entry.nodes.blocks!;
    expect(blocks.children).toHaveLength(2);
    expect(blocks.resolved).toEqual({ ch: 16 });
    expect(blocks.outputs?.[0]?.shape).toEqual(['B', 16, 32, 32]);
    expect(pa.entry.nodes.fc?.resolved).toMatchObject({ in_features: 16, out_features: 10 });
    // conv (3*16*9 + 16) + 2 blocks * (2 convs 16*16*9 + 2 bn 32) + fc (16*10 + 10)
    expect(pa.entry.totalParams).toBe(448 + 2 * (2 * 2304 + 2 * 32) + 170);
  });

  it('reports errors inside a module on the instance', () => {
    const p = editEntry(resnet(), (m) => setParam(m, 'stem', 'out_channels', literal(8)));
    const errors = analyzeModule(p).diagnostics.filter((d) => d.severity === 'error');
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatchObject({ nodeId: 'blocks' });
    expect(errors[0]!.message).toContain('ResBlock');
  });

  it('flags missing hyperparameters, bad repeat counts and recursion', () => {
    const p = editEntry(resnet(), (m) => ({ ...m, nodes: { ...m.nodes, blocks: { ...m.nodes.blocks!, params: {}, repeat: { count: literal(0) } } } }));
    expect(analyzeModule(p).diagnostics.map((d) => d.message).join('\n')).toContain('ch');
    const looped = withModule(resnet(), 'ResBlock', (m) => ({
      ...m,
      nodes: { ...m.nodes, again: { id: 'again', kind: 'module', type: 'main', name: 'again', params: {} } },
    }));
    expect(analyzeProject(looped).byModule.ResBlock![0]!.diagnostics.some((d) => d.message.includes('递归'))).toBe(true);
  });

  it('evaluates expressions with the entry defaults and rejects required entry hyperparameters', () => {
    const p = editEntry(resnet(), (m) => ({ ...m, hyperparams: m.hyperparams.map((h) => (h.name === 'num_classes' ? { name: h.name } : h)) }));
    expect(analyzeModule(p).diagnostics.some((d) => d.message.includes('num_classes'))).toBe(true);
  });
});

describe('code generation for modules', () => {
  it('writes one class per module, dependencies first, with ModuleList repeats', () => {
    expect(generateCode(resnet()).code).toBe(`import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)  # [B, 16, 32, 32]
        conv1 = self.bn1(conv1)  # [B, 16, 32, 32]
        conv1 = F.relu(conv1)  # [B, 16, 32, 32]
        conv1 = self.conv2(conv1)  # [B, 16, 32, 32]
        conv1 = self.bn2(conv1)  # [B, 16, 32, 32]
        add = conv1 + x  # [B, 16, 32, 32]
        return F.relu(add)  # [B, 16, 32, 32]


class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 16, depth: int = 2):
        super().__init__()
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)  # [B, 16, 32, 32]
        for block in self.blocks:
            x = block(x)  # [B, 16, 32, 32]
        x = self.pool(x)  # [B, 16, 1, 1]
        x = torch.flatten(x, 1)  # [B, 16]
        return self.fc(x)  # [B, 10]
`);
  });

  it('writes a different first copy for ResNet-style stages', () => {
    const p = editEntry(resnet(), (m) => ({
      ...m,
      nodes: { ...m.nodes, blocks: { ...m.nodes.blocks!, repeat: { count: { expr: 'depth' }, first: { ch: { expr: 'width' } } } } },
    }));
    expect(generateCode(p).code).toContain('self.blocks = nn.ModuleList([ResBlock(width)] + [ResBlock(width) for _ in range(depth - 1)])');
  });

  it('inlines nn.Sequential in the scope of its parent', () => {
    const { code } = generateCode(sequentialHead());
    expect(analyzeModule(sequentialHead()).diagnostics).toEqual([]);
    expect(code).toContain('        self.head = nn.Sequential(nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 10))\n');
    expect(code).toContain('        return self.head(x)  # [B, 10]\n');
    expect(code).not.toContain('class Head');
  });

  it('writes code blocks verbatim and trusts their declared shapes', () => {
    const p = withCodeBlock();
    const a = analyzeModule(p);
    expect(a.diagnostics).toEqual([]);
    expect(a.nodes.fc?.resolved?.in_features).toBe(256);
    expect(generateCode(p).code).toContain(`        x = self.conv(x)  # [B, 4, 8, 8]
        x = x.view(x.size(0), -1)
        return self.fc(x)  # [B, 10]
`);
  });

  it('moves a still-needed value out of the way before a code block overwrites it', () => {
    const p = projectOf('main', [
      moduleOf(
        'main',
        'Residual',
        [
          ['x', 'Input', { shape: literal('B, 4') }],
          ['code', 'code', {}, { code: { source: 'x = x * 2', inputs: ['x'], outputs: ['x'] } }],
          ['add', 'torch.add'],
          ['output', 'Output'],
        ],
        ['x -> code', 'code -> add.a', 'x -> add.b', 'add -> output'],
      ),
    ]);
    expect(generateCode(p).code).toContain(`        x_1 = x
        x = x * 2
        return x + x_1
`);
  });

  it('binds code block inputs to the names it reads, and stores hyperparameters forward() needs', () => {
    const p = projectOf('main', [
      moduleOf(
        'main',
        'Scaled',
        [
          ['x', 'Input', { shape: literal('B, 4') }],
          ['mul', 'torch.mul', { other: { expr: 'scale * 2' } }],
          ['code', 'code', {}, { code: { source: 'h = h + self.scale', inputs: ['h'], outputs: ['h'] } }],
          ['output', 'Output'],
        ],
        ['x -> mul', 'mul -> code', 'code -> output'],
        { hyperparams: [{ name: 'scale', annotation: 'float', default: { literal: 1 } }] },
      ),
    ]);
    const { code } = generateCode(p);
    expect(code).toContain('    def __init__(self, scale: float = 1.0):\n        super().__init__()\n        self.scale = scale\n');
    expect(code).toContain('        x = x * (self.scale * 2)');
    expect(code).toContain('        h = x\n        h = h + self.scale\n        return h\n');
  });

  it('reports an auto parameter that differs between instances instead of guessing', () => {
    const block = moduleOf(
      'Block',
      'Block',
      [
        ['x', 'Input', { shape: literal('') }],
        ['fc', 'Linear', { in_features: { auto: true }, out_features: literal(8) }],
        ['output', 'Output'],
      ],
      ['x -> fc', 'fc -> output'],
    );
    const main = moduleOf(
      'main',
      'Two',
      [
        ['a', 'Input', { shape: literal('B, 4') }],
        ['b', 'Input', { shape: literal('B, 6') }],
        ['b1', 'module:Block'],
        ['b2', 'module:Block'],
        ['output', 'Output'],
        ['output_1', 'Output'],
      ],
      ['a -> b1', 'b -> b2', 'b1 -> output', 'b2 -> output_1'],
      {},
      { Block: block },
    );
    const { code, todos } = generateCode(projectOf('main', [block, main]));
    expect(todos).toBeGreaterThan(0);
    expect(code).toContain('self.fc = nn.Linear(4, 8)  # TODO(neuroloom): in_features');
  });
});

describe('module editing', () => {
  const pack = (p: LoomProject) => packAsModule(p, 'main', ['fc1', 'act1'], 'Encoder', analyzeModule(p));

  it('packs a selection into a module with inferred inputs and outputs', () => {
    const { project, moduleId, nodeId } = pack(mlp());
    const enc = project.modules[moduleId]!;
    expect(enc.className).toBe('Encoder');
    expect(Object.values(enc.nodes).map((n) => n.name).sort()).toEqual(['act1', 'fc1', 'output', 'x']);
    expect(enc.nodes[enc.interface.inputs[0]!]!.params.shape).toEqual({ literal: 'B, 784' });
    const main = entryModule(project);
    expect(main.nodes[nodeId]).toMatchObject({ kind: 'module', type: moduleId, name: 'encoder1' });
    expect(analyzeModule(project).diagnostics).toEqual([]);
    expect(generateCode(project).code).toContain('        self.encoder1 = Encoder()\n');
    expect(analyzeModule(project).totalParams).toBe(analyzeModule(mlp()).totalParams);
  });

  it('unpacks an instance back to the original graph', () => {
    const packed = pack(mlp());
    const { project } = unpackModule(packed.project, 'main', packed.nodeId);
    const cleaned = deleteModule(project, packed.moduleId);
    expect(diffSignatures(structuralSignature(cleaned), structuralSignature(mlp()))).toEqual({ onlyInA: [], onlyInB: [] });
  });

  it('promotes hyperparameters used by the packed nodes and substitutes them when unpacking', () => {
    const p = resnet();
    const packed = packAsModule(p, 'main', ['stem'], 'Stem', analyzeModule(p));
    const stem = packed.project.modules[packed.moduleId]!;
    expect(stem.hyperparams).toEqual([{ name: 'width', annotation: 'int' }]);
    expect(entryModule(packed.project).nodes[packed.nodeId]!.params).toEqual({ width: { expr: 'width' } });
    expect(analyzeModule(packed.project).diagnostics).toEqual([]);

    // A repeated instance has no single graph to inline.
    expect(() => unpackModule(p, 'main', 'blocks')).toThrow();
    const inlined = unpackModule(packed.project, 'main', packed.nodeId).project;
    expect(entryModule(inlined).nodes[Object.keys(entryModule(inlined).nodes).find((id) => entryModule(inlined).nodes[id]!.name === 'stem')!]!.params.out_channels).toEqual({
      expr: 'width',
    });
  });

  it('makes an independent copy for one instance', () => {
    const p = resnet();
    const { project, moduleId } = makeUnique(p, 'main', 'blocks');
    expect(project.modules[moduleId]!.className).toBe('ResBlock2');
    expect(entryModule(project).nodes.blocks!.type).toBe(moduleId);
    expect(generateCode(project).code).toContain('class ResBlock2(nn.Module):');
  });

  it('renames a hyperparameter in expressions and instances', () => {
    const p = setHyperparams(resnet(), 'ResBlock', [{ name: 'channels', annotation: 'int' }], { ch: 'channels' });
    expect(p.modules.ResBlock!.nodes.conv1!.params.in_channels).toEqual({ expr: 'channels' });
    expect(entryModule(p).nodes.blocks!.params).toEqual({ channels: { expr: 'width' } });
    expect(analyzeModule(p).diagnostics).toEqual([]);
  });

  it('round-trips through .loom', () => {
    const p = resnet();
    expect(generateCode(parseProjectText(serializeProject(p))).code).toBe(generateCode(p).code);
  });
});
