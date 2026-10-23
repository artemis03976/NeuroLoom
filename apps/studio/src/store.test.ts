import { entryModule, literal } from '@neuroloom/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { blankProject, sampleProject } from './lib/sample';
import { useStudio } from './store';

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
    expect(s().analysis.diagnostics.some((d) => d.severity === 'error')).toBe(true);
    expect(s().flowNodes.some((n) => n.id === 'n_act1')).toBe(false);
    s().undo();
    expect(s().analysis.diagnostics).toEqual([]);
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
