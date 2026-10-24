import { analyzeProject } from '@neuroloom/core';
import { Position } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { buildSvg } from './exportImage';
import { sceneOf, type MeasuredNode } from './exportScene';
import { sampleProject } from './sample';

const project = sampleProject();
const analysis = analyzeProject(project).entry;

/** What React Flow would report for a node of this size at its layout position. */
const measuredAt = (id: string, width: number, height: number): MeasuredNode => ({
  measured: { width, height },
  internals: {
    positionAbsolute: project.modules.main!.layout[id]!,
    handleBounds: {
      target: [{ id: 'in', type: 'target', position: Position.Left, x: -5, y: height / 2 - 5, width: 10, height: 10 }],
      source: [{ id: 'out', type: 'source', position: Position.Right, x: width - 5, y: height / 2 - 5, width: 10, height: 10 }],
    },
  },
});

describe('image export: scene of the canvas', () => {
  it('collects what each node shows', () => {
    const scene = sceneOf({ project, moduleId: 'main', analysis, getNode: () => undefined });
    expect(scene.title).toBe('MLP');
    const fc1 = scene.nodes.find((n) => n.id === 'n_fc1')!;
    expect(fc1).toMatchObject({ name: 'fc1', type: 'Linear', glyph: 'FC', category: 'linear', variant: 'layer', params: '201.0K' });
    expect(fc1.summary).toContain('256');
    expect(fc1.rows).toBeUndefined();
    const input = scene.nodes.find((n) => n.id === 'n_x')!;
    expect(input).toMatchObject({ category: 'io', summary: '[B, 784]' });
    // Shapes on edges, as the canvas labels them.
    expect(scene.edges.map((e) => e.shape)).toEqual(['[B, 784]', '[B, 256]', '[B, 256]', '[B, 10]']);
  });

  it('uses React Flow positions, sizes and handles when it has them', () => {
    const scene = sceneOf({ project, moduleId: 'main', analysis, getNode: (id) => measuredAt(id, 200, 80) });
    const fc2 = scene.nodes.find((n) => n.id === 'n_fc2')!;
    expect(fc2).toMatchObject({ x: 780, y: 40, width: 200, height: 80 });
    expect(fc2.handles.map((h) => [h.id, h.type, h.x])).toEqual([
      ['in', 'target', -5],
      ['out', 'source', 195],
    ]);
    const { svg } = buildSvg(scene, { palette: 'print' });
    for (const text of ['>fc1</text>', '>act1</text>', '>output</text>', '>[B, 10]</text>']) expect(svg).toContain(text);
    // Four edges, from the outer edge of each right handle (x + 205) to the next node's left handle (x − 5).
    expect(svg.match(/stroke-width="1.75"/g)).toHaveLength(4);
    expect(svg).toContain('<path d="M205,80 C230,80 230,80 255,80"');
  });

  it('keeps only the selected nodes and the edges between them', () => {
    const scene = sceneOf({ project, moduleId: 'main', analysis, getNode: () => undefined, only: ['n_fc1', 'n_act1', 'n_out'] });
    expect(scene.nodes.map((n) => n.name)).toEqual(['fc1', 'act1', 'output']);
    expect(scene.edges.map((e) => e.id)).toEqual(['e2']);
  });
});
