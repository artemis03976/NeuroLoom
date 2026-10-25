// Editing rules the editor relies on: names and the forward() interface when nodes
// are added, one edge per input port, no cycles, what removing a node takes with it,
// modules that cannot contain themselves, drill-down paths, and files from a newer
// NeuroLoom.
import { describe, expect, it } from 'vitest';
import {
  SCHEMA_VERSION,
  addModuleNode,
  addNode,
  connect,
  dependsOn,
  emptyModule,
  findPath,
  literal,
  parseProject,
  pruneDanglingEdges,
  removeNodes,
  resolvePath,
  setParam,
  uniqueClassName,
  uniqueModuleId,
  withModule,
  type ModuleDef,
} from '../src';
import { moduleOf, projectOf } from './fixtures';

const linear = { in_features: literal(8), out_features: literal(8) };
const names = (mod: ModuleDef, ids: string[]) => ids.map((id) => mod.nodes[id]!.name);

describe('editing a module', () => {
  it('names new nodes uniquely and appends new inputs and outputs to the interface', () => {
    let mod = emptyModule('main', 'Net');
    const added: string[] = [];
    for (const type of ['neuroloom.Input', 'torch.nn.Linear', 'torch.nn.Linear', 'neuroloom.Input', 'neuroloom.Output']) {
      const result = addNode(mod, type, { x: 0, y: 0 });
      mod = result.mod;
      added.push(result.nodeId);
    }
    const all = names(mod, added);
    expect(new Set(all).size).toBe(all.length);
    expect(mod.interface.inputs).toEqual([added[0], added[3]]);
    expect(mod.interface.outputs).toEqual([added[4]]);
  });

  it('keeps one edge per input port: connecting again replaces the old edge', () => {
    const mod = moduleOf('main', 'Net', [['a', 'Linear', linear], ['b', 'Linear', linear], ['c', 'Linear', linear]], ['a -> c']);
    const next = connect(mod, { node: 'b', port: 'out' }, { node: 'c', port: 'in' })!;
    const into = next.edges.filter((e) => e.target.node === 'c');
    expect(into.map((e) => e.source.node)).toEqual(['b']);
  });

  it('refuses a connection that would close a cycle', () => {
    const mod = moduleOf('main', 'Net', [['a', 'Linear', linear], ['b', 'Linear', linear]], ['a -> b']);
    expect(connect(mod, { node: 'b', port: 'out' }, { node: 'a', port: 'in' })).toBeNull();
    expect(connect(mod, { node: 'a', port: 'out' }, { node: 'a', port: 'in' })).toBeNull();
  });

  it('removes a node with its edges and its place in the interface, and closes gaps in variadic ports', () => {
    const mod = moduleOf(
      'main',
      'Net',
      [
        ['x', 'Input', { shape: literal('B, 8') }],
        ['a', 'Linear', linear],
        ['b', 'Linear', linear],
        ['c', 'Linear', linear],
        ['cat', 'torch.cat'],
        ['output', 'Output'],
      ],
      ['x -> a', 'x -> b', 'x -> c', 'a -> cat.in0', 'b -> cat.in1', 'c -> cat.in2', 'cat -> output'],
    );
    const without = removeNodes(mod, ['b', 'x']);
    expect(without.edges.some((e) => e.source.node === 'b' || e.target.node === 'b' || e.source.node === 'x')).toBe(false);
    expect(without.interface.inputs).toEqual([]);
    const ports = without.edges.filter((e) => e.target.node === 'cat').map((e) => `${e.source.node}->${e.target.port}`);
    expect(ports).toEqual(['a->in0', 'c->in1']);
  });

  it('keeps the defaults of optional forward() arguments through edits and saving', () => {
    const plain = moduleOf(
      'main',
      'Net',
      [['x', 'Input', { shape: literal('B, 8') }], ['mask', 'Input', { shape: literal('') }], ['fc', 'Linear', linear], ['output', 'Output']],
      ['x -> fc', 'fc -> output'],
    );
    let mod: ModuleDef = { ...plain, interface: { ...plain.interface, defaults: { mask: 'None' } } };
    mod = addNode(mod, 'torch.nn.ReLU', { x: 0, y: 0 }).mod;
    expect(mod.interface.defaults).toEqual({ mask: 'None' });
    const reread = parseProject(JSON.parse(JSON.stringify(projectOf('main', [mod]))));
    expect(reread.modules.main!.interface.defaults).toEqual({ mask: 'None' });
    expect(removeNodes(mod, ['mask']).interface.defaults).toEqual({});
  });

  it('sets a parameter, and returns it to its default when set to undefined', () => {
    const mod = moduleOf('main', 'Net', [['a', 'Linear', linear]], []);
    const set = setParam(mod, 'a', 'bias', literal(false));
    expect(set.nodes.a!.params.bias).toEqual(literal(false));
    expect('bias' in setParam(set, 'a', 'bias', undefined).nodes.a!.params).toBe(false);
  });
});

/** Main contains a Block, and the Block contains an Inner. */
function nested() {
  const inner = moduleOf('Inner', 'Inner', [['x', 'Input', { shape: literal('B, 8') }], ['fc', 'Linear', linear], ['output', 'Output']], ['x -> fc', 'fc -> output']);
  const block = moduleOf('Block', 'Block', [['x', 'Input', { shape: literal('B, 8') }], ['inner', 'module:Inner'], ['output', 'Output']], ['x -> inner', 'inner -> output'], {}, { Inner: inner });
  const main = moduleOf('main', 'Main', [['x', 'Input', { shape: literal('B, 8') }], ['block', 'module:Block'], ['output', 'Output']], ['x -> block', 'block -> output'], {}, { Block: block });
  return projectOf('main', [main, block, inner]);
}

describe('modules in a project', () => {
  it('does not let a module be used inside itself or inside a module it contains', () => {
    const p = nested();
    expect(dependsOn(p, 'main', 'Inner')).toBe(true);
    expect(dependsOn(p, 'Inner', 'main')).toBe(false);
    expect(() => addModuleNode(p, 'Inner', 'Block', { x: 0, y: 0 })).toThrow(/inside itself|自身/);
    expect(() => addModuleNode(p, 'Block', 'Block', { x: 0, y: 0 })).toThrow(/inside itself|自身/);
    const { project } = addModuleNode(p, 'main', 'Inner', { x: 0, y: 0 });
    expect(Object.values(project.modules.main!.nodes).filter((n) => n.type === 'Inner')).toHaveLength(1);
  });

  it('finds the drill-down path to a module, and keeps a path only as far as its instances exist', () => {
    const p = nested();
    expect(findPath(p, 'Inner')).toEqual(['block', 'inner']);
    expect(resolvePath(p, ['block', 'inner'])).toEqual({ moduleId: 'Inner', path: ['block', 'inner'] });
    const withoutInner = withModule(p, 'Block', (mod) => removeNodes(mod, ['inner']));
    expect(resolvePath(withoutInner, ['block', 'inner'])).toEqual({ moduleId: 'Block', path: ['block'] });
    expect(findPath(withoutInner, 'Inner')).toBeUndefined();
  });

  it('gives new modules a class name and id nobody uses', () => {
    const p = nested();
    expect(uniqueClassName(p, 'Block')).toBe('Block2');
    expect(uniqueClassName(p, 'Head')).toBe('Head');
    expect(uniqueModuleId(p, 'Block')).toBe('Block_2');
  });

  it('drops edges to a port a module no longer has', () => {
    const p = nested();
    // Block's input is deleted: Main's edge into it has nowhere to go.
    const withoutInput = withModule(p, 'Block', (mod) => removeNodes(mod, ['x']));
    const pruned = pruneDanglingEdges(withoutInput);
    const main = pruned.modules.main!;
    expect(main.edges.some((e) => e.target.node === 'block')).toBe(false);
    expect(main.edges.some((e) => e.source.node === 'block')).toBe(true);
    expect(pruneDanglingEdges(p)).toBe(p);
  });
});

describe('opening files', () => {
  it('refuses a file from a newer NeuroLoom and says so', () => {
    const file = { ...nested(), schemaVersion: '9.9' };
    expect(() => parseProject(file)).toThrow(/newer|更新/);
    expect(parseProject({ ...nested(), schemaVersion: SCHEMA_VERSION }).entry).toBe('main');
  });

  it('refuses a file without an entry module', () => {
    expect(() => parseProject({ ...nested(), entry: 'missing' })).toThrow(/missing/);
  });
});
