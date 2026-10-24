import {
  CODE_TYPE,
  SCHEMA_VERSION,
  auto,
  emptyModule,
  entryModule,
  expr,
  getSpec,
  initialParams,
  literal,
  outputsOf,
  withDefaults,
  type LoomProject,
  type ModuleDef,
  type NodeInst,
  type NodeSpec,
  type ParamValue,
} from '../src';

type NodeRow = [name: string, type: string, params?: Record<string, ParamValue>, pos?: [number, number]];

const SHORT: Record<string, string> = {
  Input: 'neuroloom.Input',
  Output: 'neuroloom.Output',
  Linear: 'torch.nn.Linear',
  ReLU: 'torch.nn.ReLU',
  relu: 'torch.nn.functional.relu',
};

/**
 * Builds a single-module project. Node ids are the names; edges are written
 * "a -> b" (a.out to b.in). Nodes default to a left-to-right row.
 */
export function build(className: string, rows: NodeRow[], edges: string[]): LoomProject {
  const mod = emptyModule('main', className);
  rows.forEach(([name, short, params = {}, pos], i) => {
    const type = SHORT[short] ?? short;
    const kind = getSpec(type)!.kind;
    mod.nodes[name] = { id: name, kind, type, name, params };
    mod.layout[name] = { x: pos?.[0] ?? i * 220, y: pos?.[1] ?? 0 };
    if (kind === 'input') mod.interface.inputs.push(name);
    if (kind === 'output') mod.interface.outputs.push(name);
  });
  edges.forEach((text, i) => {
    const [a, b] = text.split('->').map((s) => s.trim()) as [string, string];
    mod.edges.push({ id: `e${i}`, source: { node: a, port: 'out' }, target: { node: b, port: 'in' } });
  });
  return { schemaVersion: SCHEMA_VERSION, name: className, entry: 'main', modules: { main: mod } };
}

/** The project with its entry module replaced by `update(entry)`. */
export function editEntry(p: LoomProject, update: (mod: ModuleDef) => ModuleDef): LoomProject {
  return { ...p, modules: { ...p.modules, [p.entry]: update(entryModule(p)) } };
}

const INPUT_NAMES = ['x', 'y', 'z', 'w'];

/** A spec's own example (the round-trip and FLOPs tests): inputs -> the spec's node -> one output per output port. */
export function exampleProject(spec: NodeSpec): LoomProject {
  const example = spec.example!;
  const mod: ModuleDef = emptyModule('main', 'Example');

  example.inputs.forEach((input, i) => {
    const id = `in${i}`;
    mod.nodes[id] = {
      id,
      kind: 'input',
      type: 'neuroloom.Input',
      name: INPUT_NAMES[i]!,
      params: { shape: literal(input.shape), dtype: literal(input.dtype ?? 'float32') },
    };
    mod.layout[id] = { x: 0, y: i * 120 };
    mod.interface.inputs.push(id);
  });

  const params: Record<string, ParamValue> = { ...initialParams(spec) };
  for (const [k, v] of Object.entries(example.params ?? {})) params[k] = v === 'auto' ? { auto: true } : literal(v);
  const name = spec.kind === 'layer' ? `${spec.defaultName}1` : spec.defaultName;
  mod.nodes.node = { id: 'node', kind: spec.kind, type: spec.type, name, params };
  mod.layout.node = { x: 260, y: 0 };

  let next = 0;
  for (const port of spec.inputs) {
    const count = port.variadic ? example.inputs.length - next : next < example.inputs.length ? 1 : 0;
    for (let i = 0; i < count; i++, next++) {
      mod.edges.push({
        id: `e${mod.edges.length}`,
        source: { node: `in${next}`, port: 'out' },
        target: { node: 'node', port: port.variadic ? `${port.name}${i}` : port.name },
      });
    }
  }

  outputsOf(spec, withDefaults(spec, params)).forEach((out, i) => {
    const id = `out${i}`;
    mod.nodes[id] = { id, kind: 'output', type: 'neuroloom.Output', name: i === 0 ? 'output' : `output_${i}`, params: {} };
    mod.layout[id] = { x: 520, y: i * 120 };
    mod.interface.outputs.push(id);
    mod.edges.push({ id: `e${mod.edges.length}`, source: { node: 'node', port: out.name }, target: { node: id, port: 'in' } });
  });

  return { schemaVersion: SCHEMA_VERSION, name: spec.type, entry: 'main', modules: { main: mod } };
}

const lin = (inF: number | 'auto', outF: number, bias?: boolean): Record<string, ParamValue> => ({
  in_features: inF === 'auto' ? auto() : literal(inF),
  out_features: literal(outF),
  ...(bias === undefined ? {} : { bias: literal(bias) }),
});

export const mlp = () =>
  build(
    'MLP',
    [
      ['x', 'Input', { shape: literal('B, 784') }],
      ['fc1', 'Linear', lin('auto', 256)],
      ['act1', 'ReLU'],
      ['fc2', 'Linear', lin('auto', 10)],
      ['output', 'Output'],
    ],
    ['x -> fc1', 'fc1 -> act1', 'act1 -> fc2', 'fc2 -> output'],
  );

export const mlpFunctional = () =>
  build(
    'FunctionalMLP',
    [
      ['x', 'Input', { shape: literal('B, 784') }],
      ['fc1', 'Linear', lin(784, 256)],
      ['relu', 'relu'],
      ['fc2', 'Linear', lin('auto', 10, false)],
      ['output', 'Output'],
    ],
    ['x -> fc1', 'fc1 -> relu', 'relu -> fc2', 'fc2 -> output'],
  );

export const twoHeads = () =>
  build(
    'TwoHeads',
    [
      ['x', 'Input', { shape: literal('B, 32') }, [0, 75]],
      ['fc', 'Linear', lin('auto', 64), [220, 75]],
      ['act', 'ReLU', {}, [440, 75]],
      ['head1', 'Linear', lin('auto', 10), [660, 0]],
      ['head2', 'Linear', lin('auto', 1), [660, 150]],
      ['output', 'Output', {}, [880, 0]],
      ['output_1', 'Output', {}, [880, 150]],
    ],
    ['x -> fc', 'fc -> act', 'act -> head1', 'act -> head2', 'head1 -> output', 'head2 -> output_1'],
  );

export const twoInputs = () =>
  build(
    'TwoInputs',
    [
      ['a', 'Input', { shape: literal('B, 8') }, [0, 0]],
      ['b', 'Input', { shape: literal('B, 16') }, [0, 150]],
      ['fa', 'Linear', lin('auto', 4), [220, 0]],
      ['fb', 'Linear', lin('auto', 4), [220, 150]],
      ['output', 'Output', {}, [440, 0]],
      ['output_1', 'Output', {}, [440, 150]],
    ],
    ['a -> fa', 'b -> fb', 'fa -> output', 'fb -> output_1'],
  );

/** A function call on one branch level with a layer on the other (Inception's torch.cat): statement order must survive reading back. */
export const branches = () =>
  build(
    'Branches',
    [
      ['x', 'Input', { shape: literal('B, 8') }, [0, 75]],
      ['fa', 'Linear', lin('auto', 4), [220, 0]],
      ['relu', 'relu', {}, [440, 0]],
      ['fb', 'Linear', lin('auto', 4), [440, 150]],
      ['output', 'Output', {}, [660, 0]],
      ['output_1', 'Output', {}, [660, 150]],
    ],
    ['x -> fa', 'fa -> relu', 'x -> fb', 'relu -> output', 'fb -> output_1'],
  );

export const ROUND_TRIP_FIXTURES = { mlp, mlpFunctional, twoHeads, twoInputs, branches };

// ---------------------------------------------------------------- multi-module projects

type Row = [name: string, type: string, params?: Record<string, ParamValue>, extra?: Partial<NodeInst>];

/**
 * A module whose node ids are the names. Types: a short name (see SHORT), a
 * NodeSpec type, "module:<ModuleId>" or "code". Edges are "a -> b" or
 * "a.port -> b.port"; module and code nodes default to their first port.
 */
export function moduleOf(
  id: string,
  className: string,
  rows: Row[],
  edges: string[],
  options: Partial<ModuleDef> = {},
  defs: Record<string, ModuleDef> = {},
): ModuleDef {
  const mod: ModuleDef = { ...emptyModule(id, className), ...options };
  rows.forEach(([name, short, params = {}, extra = {}], i) => {
    let node: NodeInst;
    if (short.startsWith('module:')) node = { id: name, kind: 'module', type: short.slice(7), name, params, ...extra };
    else if (short === 'code') node = { id: name, kind: 'code', type: CODE_TYPE, name, params, ...extra };
    else {
      const type = SHORT[short] ?? short;
      node = { id: name, kind: getSpec(type)!.kind, type, name, params, ...extra };
    }
    mod.nodes[name] = node;
    mod.layout[name] = { x: i * 220, y: 0 };
    if (node.kind === 'input') mod.interface.inputs.push(name);
    if (node.kind === 'output') mod.interface.outputs.push(name);
  });
  const port = (node: NodeInst, side: 'source' | 'target'): string => {
    if (node.kind === 'module') return defs[node.type]!.interface[side === 'source' ? 'outputs' : 'inputs'][0]!;
    if (node.kind === 'code') return node.code![side === 'source' ? 'outputs' : 'inputs'][0]!;
    return side === 'source' ? 'out' : (getSpec(node.type)!.inputs[0]?.name ?? 'in');
  };
  edges.forEach((text, i) => {
    const [a, b] = text.split('->').map((s) => s.trim()) as [string, string];
    const [an, ap] = a.split('.') as [string, string | undefined];
    const [bn, bp] = b.split('.') as [string, string | undefined];
    mod.edges.push({
      id: `e${i}`,
      source: { node: an, port: ap ?? port(mod.nodes[an]!, 'source') },
      target: { node: bn, port: bp ?? port(mod.nodes[bn]!, 'target') },
    });
  });
  return mod;
}

export function projectOf(entry: string, modules: ModuleDef[], extra: Partial<LoomProject> = {}): LoomProject {
  return { schemaVersion: SCHEMA_VERSION, name: entry, entry, modules: Object.fromEntries(modules.map((m) => [m.id, m])), ...extra };
}

const conv3 = (ch: ParamValue): Record<string, ParamValue> => ({
  in_channels: ch,
  out_channels: ch,
  kernel_size: literal(3),
  padding: literal(1),
  bias: literal(false),
});

/** A small ResNet: a residual block repeated `depth` times, with hyperparameters throughout. */
export function resnet(): LoomProject {
  const block = moduleOf(
    'ResBlock',
    'ResBlock',
    [
      ['x', 'Input', { shape: literal('') }],
      ['conv1', 'torch.nn.Conv2d', conv3(expr('ch'))],
      ['bn1', 'torch.nn.BatchNorm2d', { num_features: expr('ch') }],
      ['relu', 'relu'],
      ['conv2', 'torch.nn.Conv2d', conv3(expr('ch'))],
      ['bn2', 'torch.nn.BatchNorm2d', { num_features: expr('ch') }],
      ['add', 'torch.add'],
      ['relu_1', 'relu'],
      ['output', 'Output'],
    ],
    ['x -> conv1', 'conv1 -> bn1', 'bn1 -> relu', 'relu -> conv2', 'conv2 -> bn2', 'bn2 -> add.a', 'x -> add.b', 'add -> relu_1', 'relu_1 -> output'],
    { hyperparams: [{ name: 'ch', annotation: 'int' }] },
  );
  const net = moduleOf(
    'main',
    'ResNet',
    [
      ['x', 'Input', { shape: literal('B, 3, 32, 32') }],
      ['stem', 'torch.nn.Conv2d', { in_channels: literal(3), out_channels: expr('width'), kernel_size: literal(3), padding: literal(1) }],
      ['blocks', 'module:ResBlock', { ch: expr('width') }, { repeat: { count: expr('depth') } }],
      ['pool', 'torch.nn.AdaptiveAvgPool2d', { output_size: literal(1) }],
      ['flatten', 'torch.flatten', { start_dim: literal(1) }],
      ['fc', 'torch.nn.Linear', { in_features: auto(), out_features: expr('num_classes') }],
      ['output', 'Output'],
    ],
    ['x -> stem', 'stem -> blocks', 'blocks -> pool', 'pool -> flatten', 'flatten -> fc', 'fc -> output'],
    {
      hyperparams: [
        { name: 'num_classes', annotation: 'int', default: { literal: 10 } },
        { name: 'width', annotation: 'int', default: { literal: 16 } },
        { name: 'depth', annotation: 'int', default: { literal: 2 } },
      ],
    },
    { ResBlock: block },
  );
  return projectOf('main', [block, net]);
}

/** An inline nn.Sequential head whose expressions use the parent's hyperparameter. */
export function sequentialHead(): LoomProject {
  const head = moduleOf(
    'Head',
    'Head',
    [
      ['x', 'Input', { shape: literal('') }],
      ['0', 'Linear', { in_features: auto(), out_features: expr('hidden') }],
      ['1', 'ReLU'],
      ['2', 'Linear', { in_features: expr('hidden'), out_features: literal(10) }],
      ['output', 'Output'],
    ],
    ['x -> 0', '0 -> 1', '1 -> 2', '2 -> output'],
    { sequential: true },
  );
  const main = moduleOf(
    'main',
    'Classifier',
    [
      ['x', 'Input', { shape: literal('B, 64') }],
      ['head', 'module:Head'],
      ['output', 'Output'],
    ],
    ['x -> head', 'head -> output'],
    { hyperparams: [{ name: 'hidden', annotation: 'int', default: { literal: 32 } }] },
    { Head: head },
  );
  return projectOf('main', [head, main]);
}

/** A code block that flattens with x.view(x.size(0), -1), declaring its output shape. */
export function withCodeBlock(): LoomProject {
  return projectOf('main', [
    moduleOf(
      'main',
      'CodeNet',
      [
        ['x', 'Input', { shape: literal('B, 3, 8, 8') }],
        ['conv', 'torch.nn.Conv2d', { in_channels: literal(3), out_channels: literal(4), kernel_size: literal(3), padding: literal(1) }],
        ['code', 'code', {}, { code: { source: 'x = x.view(x.size(0), -1)', inputs: ['x'], outputs: ['x'] }, shapes: { x: 'B, 256' } }],
        ['fc', 'Linear', { in_features: auto(), out_features: literal(10) }],
        ['output', 'Output'],
      ],
      ['x -> conv', 'conv -> code', 'code -> fc', 'fc -> output'],
    ),
  ]);
}
