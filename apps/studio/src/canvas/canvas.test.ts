import { analyzeProject, emptyModule, literal, renameNode, SCHEMA_VERSION, type LoomProject, type ModuleDef } from '@neuroloom/core';
import { describe, expect, it } from 'vitest';
import { residualChain } from '../lib/bench';
import { nodeFaces } from './face';
import { flowMode, lineage, MAX_DASH_EDGES, MAX_PARTICLE_EDGES } from './flowMode';
import { replaySteps } from './replay';

/** x → a, x → b, (a, b) → add → output, and a module node m after add. */
function diamond(): ModuleDef {
  const mod = emptyModule('main', 'Diamond');
  const node = (id: string, kind: 'input' | 'output' | 'layer' | 'op' | 'module', type: string, y = 0) => {
    mod.nodes[id] = { id, kind, type, name: id, params: {} };
    mod.layout[id] = { x: 0, y };
  };
  node('x', 'input', 'neuroloom.Input');
  node('a', 'layer', 'torch.nn.ReLU', 0);
  node('b', 'layer', 'torch.nn.Tanh', 100);
  node('add', 'op', 'torch.add');
  node('m', 'module', 'Block');
  node('output', 'output', 'neuroloom.Output');
  const edges: [string, string, string][] = [
    ['x', 'a', 'in'],
    ['x', 'b', 'in'],
    ['a', 'add', 'a'],
    ['b', 'add', 'b'],
    ['add', 'm', 'x'],
    ['m', 'output', 'in'],
  ];
  edges.forEach(([s, t, port], i) => mod.edges.push({ id: `e${i}`, source: { node: s, port: 'out' }, target: { node: t, port } }));
  return mod;
}

describe('forward replay', () => {
  it('runs parallel branches in one step, in topological order', () => {
    expect(replaySteps(diamond(), false)).toEqual([['x'], ['a', 'b'], ['add'], ['m'], ['output']]);
  });

  it('gives module nodes their own step when playing inside them', () => {
    const mod = diamond();
    mod.edges.push({ id: 'e9', source: { node: 'x', port: 'out' }, target: { node: 'm', port: 'skip' } });
    mod.nodes.c = { id: 'c', kind: 'layer', type: 'torch.nn.ReLU', name: 'c', params: {} };
    mod.edges.push({ id: 'e10', source: { node: 'add', port: 'out' }, target: { node: 'c', port: 'in' } });
    expect(replaySteps(mod, true)).toEqual([['x'], ['a', 'b'], ['add'], ['c'], ['m'], ['output']]);
  });
});

describe('edge flow', () => {
  it('dashes above the particle budget or when zoomed out, and stands still beyond that', () => {
    expect(flowMode(MAX_PARTICLE_EDGES, 1)).toBe('particles');
    expect(flowMode(MAX_PARTICLE_EDGES + 1, 1)).toBe('dashes');
    expect(flowMode(10, 0.3)).toBe('dashes');
    expect(flowMode(MAX_DASH_EDGES + 1, 1)).toBe('still');
    expect(flowMode(10, 0.2)).toBe('still');
  });

  it('follows a node’s lineage both ways', () => {
    expect([...lineage(diamond(), 'a')].sort()).toEqual(['e0', 'e2', 'e4', 'e5']);
  });
});

describe('node faces', () => {
  it('keep their objects when nothing they show changed', () => {
    const p: LoomProject = residualChain(20);
    const mod = p.modules.main!;
    const first = nodeFaces(p, mod, analyzeProject(p).entry);
    const again = nodeFaces(p, mod, analyzeProject(p).entry, first);
    expect(Object.keys(again).every((id) => again[id] === first[id])).toBe(true);

    const renamed: LoomProject = { ...p, modules: { main: renameNode(mod, 'fc3', 'proj') } };
    const after = nodeFaces(renamed, renamed.modules.main!, analyzeProject(renamed).entry, first);
    const changed = Object.keys(after).filter((id) => after[id] !== first[id]);
    expect(changed).toEqual(['fc3']);
    expect(after.fc3!.name).toBe('proj');
  });

  it('show what the node shows', () => {
    const mod = emptyModule('main', 'One');
    mod.nodes.x = { id: 'x', kind: 'input', type: 'neuroloom.Input', name: 'x', params: { shape: literal('B, 8') } };
    mod.nodes.fc = { id: 'fc', kind: 'layer', type: 'torch.nn.Linear', name: 'fc', params: { in_features: literal(8), out_features: literal(4) } };
    mod.edges.push({ id: 'e', source: { node: 'x', port: 'out' }, target: { node: 'fc', port: 'in' } });
    mod.interface.inputs.push('x');
    const p: LoomProject = { schemaVersion: SCHEMA_VERSION, name: 'One', entry: 'main', modules: { main: mod } };
    const faces = nodeFaces(p, mod, analyzeProject(p).entry);
    expect(faces.fc).toMatchObject({ name: 'fc', type: 'Linear', glyph: 'FC', params: 36, labelled: false, summary: '8 → 4' });
    expect(faces.x!.summary).toBe('[B, 8]');
  });
});
