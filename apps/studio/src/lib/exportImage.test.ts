import { getBezierPath, Position } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { buildSvg, estimateText, fitText, PALETTES, subsetScene, wrapText, type Scene, type SceneNode } from './exportImage';

const handles = (width: number, height: number, ins: string[] = ['in'], outs: string[] = ['out']): SceneNode['handles'] => [
  ...ins.map((id) => ({ id, type: 'target' as const, position: Position.Left, x: -5, y: height / 2 - 5, width: 10, height: 10 })),
  ...outs.map((id) => ({ id, type: 'source' as const, position: Position.Right, x: width - 5, y: height / 2 - 5, width: 10, height: 10 })),
];

const node = (id: string, x: number, y: number, extra: Partial<SceneNode> = {}): SceneNode => ({
  id,
  x,
  y,
  width: 168,
  height: 72,
  category: 'linear',
  variant: 'layer',
  glyph: 'FC',
  name: id,
  type: 'Linear',
  summary: 'k=3 s=1',
  params: '200.9K',
  handles: handles(168, 72),
  ...extra,
});

const scene: Scene = {
  title: 'MLP',
  nodes: [node('a', 0, 0), node('b', 260, 0), node('c', 520, 100, { repeat: '×3' })],
  edges: [
    { id: 'e1', source: 'a', sourceHandle: 'out', target: 'b', targetHandle: 'in', shape: '[B, 256]' },
    { id: 'e2', source: 'b', sourceHandle: 'out', target: 'c', targetHandle: 'in' },
  ],
};

const rootOf = (svg: string) => /<svg [^>]*>/.exec(svg)![0];
const colours = (svg: string) => new Set(svg.match(/#[0-9A-Fa-f]{6}\b/g));

describe('image export: SVG', () => {
  it('is a standalone document fitted to the content with a margin', () => {
    const { svg, width, height } = buildSvg(scene, { palette: 'light' });
    expect(svg.startsWith('<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg"')).toBe(true);
    // Nodes span x -5 (input handle) … 520 + 168 + 8 (repeat stack), y 0 … 100 + 72 + 8, plus a 24 px margin.
    expect(rootOf(svg)).toContain('viewBox="-29 -24 749 228"');
    expect([width, height]).toEqual([749, 228]);
    expect(rootOf(svg)).toContain('width="749" height="228"');
    expect(svg).toContain('<title>MLP</title>');
    expect(svg).toContain(`<rect x="-29" y="-24" width="749" height="228" fill="${PALETTES.light.background}"/>`);
  });

  it('scales only the size attributes', () => {
    const one = buildSvg(scene, { palette: 'light' }).svg;
    const three = buildSvg(scene, { palette: 'light', scale: 3 }).svg;
    expect(rootOf(three)).toContain('width="2247" height="684" viewBox="-29 -24 749 228"');
    expect(three.replace(rootOf(three), '')).toBe(one.replace(rootOf(one), ''));
  });

  it('is deterministic and refers to nothing outside itself', () => {
    const { svg } = buildSvg(scene, { palette: 'dark' });
    expect(buildSvg(scene, { palette: 'dark' }).svg).toBe(svg);
    expect(svg).not.toMatch(/var\(|url\(|<style|class=|<animate|<image|href=|<foreignObject/);
    expect(svg).toContain("font-family=\"'Segoe UI Variable'");
    expect(svg).toContain("font-family=\"'JetBrains Mono'");
  });

  it('draws edges as React Flow bezier paths between the handles', () => {
    const { svg } = buildSvg(scene, { palette: 'light' });
    // Source handle's right edge (0 + 163 + 10, 0 + 31 + 5) to the target handle's left edge (260 − 5, 36).
    const [path, labelX, labelY] = getBezierPath({
      sourceX: 173,
      sourceY: 36,
      sourcePosition: Position.Right,
      targetX: 255,
      targetY: 36,
      targetPosition: Position.Left,
    });
    const rounded = path.replace(/-?\d*\.?\d+/g, (m) => String(Math.round(Number(m) * 100) / 100));
    expect(svg).toContain(`<path d="${rounded}" stroke="${PALETTES.light.flow}" stroke-width="1.75" stroke-opacity="0.85"/>`);
    // Unknown shape: dashed and unlabelled.
    expect(svg).toMatch(/<path d="M433,36 C[^"]+" stroke="#C2C8D9" stroke-width="1.75" stroke-dasharray="5 5"\/>/);
    // The shape label sits centred above the midpoint, like .nl-edge-label.
    expect(svg).toContain(`text-anchor="middle" fill="${PALETTES.light.muted}">[B, 256]</text>`);
    expect(labelX).toBe(214);
    expect(svg).toMatch(new RegExp(`<text x="214" y="[\\d.]+" font-size="10.5"`));
    expect(labelY).toBe(36);
  });

  it('shows what the node shows', () => {
    const { svg } = buildSvg(scene, { palette: 'light' });
    for (const part of ['>a</text>', '>Linear</text>', '>FC</text>', '>k=3 s=1</text>', '>200.9K</text>', '>×3</text>']) expect(svg).toContain(part);
    // Category bar and glyph in the category colour; the repeat as two cards behind the node.
    expect(svg).toContain(`fill="${PALETTES.light.categories.linear}"`);
    expect(svg.match(/<rect x="8.5" y="8.5"/g)).toHaveLength(1);
    expect(svg.match(/<rect x="4.5" y="4.5"/g)).toHaveLength(1);
    // Handles on the node sides.
    expect(svg).toContain('<circle cx="0" cy="36" r="4"');
    expect(svg).toContain('<circle cx="168" cy="36" r="4"');
  });

  it('places port labels on their handles and escapes text', () => {
    const multi = node('m<&>', 0, 0, {
      height: 120,
      rows: [{ input: { label: 'x', handle: 'x' }, output: { label: 'out', handle: 'out' } }, { input: { label: 'mask', handle: 'mask', optional: true } }],
      handles: [
        { id: 'x', type: 'target', position: Position.Left, x: -5, y: 50, width: 10, height: 10 },
        { id: 'out', type: 'source', position: Position.Right, x: 163, y: 50, width: 10, height: 10 },
        { id: 'mask', type: 'target', position: Position.Left, x: -5, y: 70, width: 10, height: 10, optional: true },
      ],
    });
    const { svg } = buildSvg({ title: 'A & B', nodes: [multi], edges: [] }, { palette: 'light' });
    expect(svg).toContain('<title>A &amp; B</title>');
    expect(svg).toContain('>m&lt;&amp;&gt;</text>');
    // Baselines sit 0.35 em under the handle centres (55 and 75).
    expect(svg).toContain(`<text x="13" y="58.68" font-size="10.5" font-family="'JetBrains Mono', 'Cascadia Code', Consolas, ui-monospace, monospace" fill="#6D7692">x</text>`);
    expect(svg).toContain('text-anchor="end" fill="#6D7692">out</text>');
    expect(svg).toMatch(/y="78.68"[^>]*fill-opacity="0.5">mask<\/text>/);
    expect(svg).toMatch(/<circle cx="0" cy="75" r="4"[^>]*stroke-dasharray/);
  });

  it('marks code blocks, modules and errors like the canvas', () => {
    const nodes = [
      node('code', 0, 0, { variant: 'code', category: 'code' }),
      node('mod', 300, 0, { variant: 'module', category: 'module' }),
      node('bad', 600, 0, { width: 258, height: 120, error: 'in_features must be a positive integer, got -3 instead of a size' }),
    ];
    const { svg } = buildSvg({ title: 't', nodes, edges: [] }, { palette: 'light' });
    expect(svg).toContain('stroke="#DCE0EC" stroke-dasharray="4 3"');
    expect(svg).toContain(`fill="${PALETTES.light.categories.module}" fill-opacity="0.08"`);
    expect(svg).toContain(`stroke="${PALETTES.light.error}"`);
    expect(svg).toContain(`fill="${PALETTES.light.error}" fill-opacity="0.12"`);
    expect(svg.match(/fill="#E5484D">[^<]+<\/text>/g)!.length).toBeGreaterThan(1);
  });

  it('uses only the colours of the chosen palette', () => {
    for (const name of ['light', 'dark', 'print'] as const) {
      const p = PALETTES[name];
      const allowed = new Set([p.background, p.surface, p.surface2, p.border, p.borderStrong, p.text, p.muted, p.flow, p.labelBorder, p.error, '#FFFFFF', ...Object.values(p.categories)]);
      const bad = node('bad', 0, 200, { error: 'broken', variant: 'module', category: 'module' });
      const used = colours(buildSvg({ ...scene, nodes: [...scene.nodes, bad] }, { palette: name }).svg);
      expect([...used].filter((c) => !allowed.has(c))).toEqual([]);
    }
    const print = buildSvg(scene, { palette: 'print' }).svg;
    expect(print).toContain('fill="#FFFFFF"/>');
    expect(print).toContain('stroke="#000000" stroke-width="1.75"/>');
    expect(print).not.toContain('stroke-opacity');
  });

  it('exports a selection with only the edges between selected nodes', () => {
    const picked = subsetScene(scene, ['b', 'c']);
    expect(picked.nodes.map((n) => n.id)).toEqual(['b', 'c']);
    expect(picked.edges.map((e) => e.id)).toEqual(['e2']);
    const { svg, width } = buildSvg(picked, { palette: 'light' });
    expect(svg).not.toContain('>a</text>');
    expect(svg).not.toContain('[B, 256]');
    expect(width).toBe(520 + 168 + 8 + 24 - (260 - 5 - 24));
  });

  it('skips edges whose handle is missing, as React Flow does', () => {
    const broken: Scene = { ...scene, edges: [{ id: 'x', source: 'a', sourceHandle: 'nope', target: 'b', targetHandle: 'in', shape: '[B]' }] };
    expect(buildSvg(broken, { palette: 'light' }).svg).toContain('<g fill="none" stroke-linecap="round"></g>');
  });

  it('cuts long text with an ellipsis and wraps errors', () => {
    const font = { size: 10, mono: true };
    expect(fitText('short', 100, font, estimateText)).toBe('short');
    // A node's measured width is rounded to whole pixels: a hair of overflow is kept.
    expect(fitText('abcdefghij', 59, font, estimateText)).toBe('abcdefghij');
    expect(fitText('abcdefghij', 57, font, estimateText)).toBe('abcdefgh…');
    const cut = fitText('a very long summary that does not fit', 60, font, estimateText);
    expect(cut.endsWith('…')).toBe(true);
    expect(estimateText(cut, font)).toBeLessThanOrEqual(60);
    expect(wrapText('one two three four', 50, font, estimateText)).toEqual(['one two', 'three', 'four']);
    expect(wrapText('形状不匹配错误', 30, font, estimateText)).toEqual(['形状不', '匹配错', '误']);
  });
});
