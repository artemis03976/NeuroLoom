import { SCHEMA_VERSION, auto, getSpec, literal, type LoomProject, type ModuleDef, type ParamValue } from '../src';

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
  const mod: ModuleDef = { id: 'main', className, nodes: {}, edges: [], layout: {} };
  rows.forEach(([name, short, params = {}, pos], i) => {
    const type = SHORT[short] ?? short;
    mod.nodes[name] = { id: name, kind: getSpec(type)!.kind, type, name, params };
    mod.layout[name] = { x: pos?.[0] ?? i * 220, y: pos?.[1] ?? 0 };
  });
  edges.forEach((text, i) => {
    const [a, b] = text.split('->').map((s) => s.trim()) as [string, string];
    mod.edges.push({ id: `e${i}`, source: { node: a, port: 'out' }, target: { node: b, port: 'in' } });
  });
  return { schemaVersion: SCHEMA_VERSION, name: className, entry: 'main', modules: { main: mod } };
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

export const ROUND_TRIP_FIXTURES = { mlp, mlpFunctional, twoHeads, twoInputs };
