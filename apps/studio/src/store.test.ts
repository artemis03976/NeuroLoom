import { entryModule, literal, parseProject } from '@neuroloom/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { blankProject, sampleProject } from './lib/sample';
import { useStudio } from './store';
import { useUi } from './ui';

const s = () => useStudio.getState();
const outFeatures = () => entryModule(s().project).nodes.n_fc1!.params.out_features;
const dirty = () => s().doc.saved !== s().project;

beforeEach(() => {
  vi.useRealTimers();
  s().loadProject(sampleProject());
});

describe('undo / redo', () => {
  it('undoes and redoes an edit', () => {
    s().setParam('n_fc1', 'out_features', literal(128));
    expect(outFeatures()).toEqual(literal(128));
    s().undo();
    expect(outFeatures()).toEqual(literal(256));
    s().redo();
    expect(outFeatures()).toEqual(literal(128));
  });

  it('drops the redo branch after a new edit', () => {
    s().setParam('n_fc1', 'out_features', literal(128));
    s().undo();
    s().rename('n_fc1', 'encoder');
    expect(s().history.future).toHaveLength(0);
  });

  it('merges rapid edits to the same field into one step', () => {
    vi.useFakeTimers();
    for (const v of [1, 12, 128]) {
      s().setParam('n_fc1', 'out_features', literal(v));
      vi.advanceTimersByTime(200);
    }
    expect(s().history.past).toHaveLength(1);
    s().undo();
    expect(outFeatures()).toEqual(literal(256));
  });

  it('does not merge edits to different fields or after a pause', () => {
    vi.useFakeTimers();
    s().setParam('n_fc1', 'out_features', literal(128));
    s().setParam('n_fc2', 'out_features', literal(5));
    vi.advanceTimersByTime(5000);
    s().setParam('n_fc2', 'out_features', literal(6));
    expect(s().history.past).toHaveLength(3);
  });

  it('undoes structural edits and keeps derived state in sync', () => {
    s().onNodesChange([{ type: 'remove', id: 'n_act1' }]);
    expect(s().viewAnalysis.diagnostics.some((d) => d.severity === 'error')).toBe(true);
    expect(s().flowNodes.some((n) => n.id === 'n_act1')).toBe(false);
    s().undo();
    expect(s().viewAnalysis.diagnostics).toEqual([]);
    expect(s().flowNodes.some((n) => n.id === 'n_act1')).toBe(true);
    expect(s().codegen.code).toContain('self.act1 = nn.ReLU()');
  });

  it('starts a fresh history when a document is loaded', () => {
    s().setParam('n_fc1', 'out_features', literal(128));
    s().loadProject(blankProject());
    expect(s().history.past).toHaveLength(0);
  });
});

describe('dirty tracking', () => {
  it('is clean after load, dirty after an edit, clean again after undo', () => {
    expect(dirty()).toBe(false);
    s().setParam('n_fc1', 'out_features', literal(128));
    expect(dirty()).toBe(true);
    s().undo();
    expect(dirty()).toBe(false);
  });

  it('is clean once the saved snapshot is recorded', () => {
    s().setParam('n_fc1', 'out_features', literal(128));
    s().markSaved(s().project, '/tmp/mlp.loom', 'mlp');
    expect(dirty()).toBe(false);
    expect(s().doc).toMatchObject({ path: '/tmp/mlp.loom', name: 'mlp' });
  });

  it('treats an imported, never-saved project as dirty', () => {
    s().loadProject(blankProject(), { saved: false });
    expect(dirty()).toBe(true);
  });

  it('keeps a clean document clean through the layout that follows loading', () => {
    s().applyLayout({ n_x: { x: 10, y: 10 } }, false);
    expect(dirty()).toBe(false);
    expect(s().history.past).toHaveLength(0);
    s().applyLayout({ n_x: { x: 50, y: 10 } }, true);
    expect(dirty()).toBe(true);
    expect(s().history.past).toHaveLength(1);
  });
});

describe('a file from a newer NeuroLoom', () => {
  beforeEach(() => s().loadProject(parseProject({ ...sampleProject(), schemaVersion: '9.9' })));

  it('can be looked at but not changed', () => {
    s().selectNode('n_fc1');
    expect(s().flowNodes.find((n) => n.selected)?.id).toBe('n_fc1');
    s().setParam('n_fc1', 'out_features', literal(128));
    s().rename('n_fc1', 'encoder');
    s().applyLayout({ n_x: { x: 50, y: 10 } }, true);
    expect(outFeatures()).toEqual(literal(256));
    expect(entryModule(s().project).nodes.n_fc1!.name).not.toBe('encoder');
    expect(s().history.past).toHaveLength(0);
    expect(dirty()).toBe(false);
    expect(useUi.getState().toasts.at(-1)?.message).toMatch(/newer|更新版本/);
  });

  it('is laid out when it opens, and stays clean', () => {
    s().applyLayout({ n_x: { x: 10, y: 10 } }, false);
    expect(entryModule(s().project).layout.n_x).toEqual({ x: 10, y: 10 });
    expect(dirty()).toBe(false);
  });
});

describe('modules', () => {
  const pack = () => {
    s().selectNodes(['n_fc1', 'n_act1']);
    s().packSelection('Encoder');
    return s().flowNodes.find((n) => n.selected)!.id;
  };

  it('packs the selection into a module instance in one undo step', () => {
    const instance = pack();
    const project = s().project;
    expect(project.modules.Encoder?.className).toBe('Encoder');
    expect(entryModule(project).nodes[instance]).toMatchObject({ kind: 'module', type: 'Encoder' });
    expect(s().viewAnalysis.diagnostics).toEqual([]);
    expect(s().codegen.code).toContain('class Encoder(nn.Module):');
    expect(s().history.past).toHaveLength(1);
    s().undo();
    expect(s().project.modules.Encoder).toBeUndefined();
  });

  it('drills into an instance, shows its shapes in context, and goes back up', () => {
    const instance = pack();
    s().enterModule(instance);
    expect(s().view).toEqual({ moduleId: 'Encoder', path: [instance] });
    const enc = s().project.modules.Encoder!;
    const fc1 = Object.values(enc.nodes).find((n) => n.name === 'fc1')!;
    expect(s().viewAnalysis.nodes[fc1.id]?.outputs?.[0]?.shape).toEqual(['B', 256]);
    expect(s().flowNodes.map((n) => n.id).sort()).toEqual(Object.keys(enc.nodes).sort());

    // Edits apply to the module on the canvas.
    s().setParam(fc1.id, 'out_features', literal(64));
    expect(s().project.modules.Encoder!.nodes[fc1.id]!.params.out_features).toEqual(literal(64));
    s().goUp();
    expect(s().view).toEqual({ moduleId: 'main', path: [] });
    expect(s().viewAnalysis.nodes.n_fc2?.resolved?.in_features).toBe(64);
  });

  it('falls back to the entry when undo removes the module on the canvas', () => {
    const instance = pack();
    s().enterModule(instance);
    s().undo();
    expect(s().view).toEqual({ moduleId: 'main', path: [] });
  });

  it('unpacks an instance back into its parent', () => {
    const instance = pack();
    s().unpack(instance);
    const names = Object.values(entryModule(s().project).nodes).map((n) => n.name).sort();
    expect(names).toEqual(['act1', 'fc1', 'fc2', 'output', 'x']);
    expect(s().viewAnalysis.diagnostics).toEqual([]);
  });

  it('renames a hyperparameter everywhere it is used', () => {
    const instance = pack();
    s().enterModule(instance);
    const enc = s().project.modules.Encoder!;
    const fc1 = Object.values(enc.nodes).find((n) => n.name === 'fc1')!;
    s().setHyperparams([{ name: 'width', annotation: 'int', default: { literal: 256 } }]);
    s().setParam(fc1.id, 'out_features', { expr: 'width' });
    s().setHyperparams([{ name: 'hidden', annotation: 'int', default: { literal: 256 } }], { width: 'hidden' });
    expect(s().project.modules.Encoder!.nodes[fc1.id]!.params.out_features).toEqual({ expr: 'hidden' });
    expect(s().viewAnalysis.diagnostics).toEqual([]);
    expect(s().codegen.code).toContain('self.fc1 = nn.Linear(784, hidden)');
  });
});
