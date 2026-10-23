import { describe, expect, it } from 'vitest';
import { alignPositions, type Box } from './align';

const boxes: Box[] = [
  { id: 'a', x: 0, y: 0, width: 100, height: 40 },
  { id: 'b', x: 300, y: 50, width: 200, height: 60 },
  { id: 'c', x: 120, y: 200, width: 100, height: 40 },
];

describe('align and distribute', () => {
  it('aligns edges and centres to the selection', () => {
    expect(alignPositions(boxes, 'left')).toEqual({ a: { x: 0, y: 0 }, b: { x: 0, y: 50 }, c: { x: 0, y: 200 } });
    expect(alignPositions(boxes, 'right').a).toEqual({ x: 400, y: 0 });
    expect(alignPositions(boxes, 'centerX').b).toEqual({ x: 150, y: 50 });
    expect(alignPositions(boxes, 'bottom').a).toEqual({ x: 0, y: 200 });
    expect(alignPositions(boxes, 'centerY').c).toEqual({ x: 120, y: 100 });
  });

  it('spaces boxes evenly between the outermost two, in their order', () => {
    // Horizontal: 500 px span, 400 px of boxes, so 50 px gaps; order a, c, b.
    expect(alignPositions(boxes, 'distributeX')).toEqual({ a: { x: 0, y: 0 }, c: { x: 150, y: 200 }, b: { x: 300, y: 50 } });
    // Vertical: 240 px span, 140 px of boxes, so 50 px gaps; order a, b, c.
    expect(alignPositions(boxes, 'distributeY')).toEqual({ a: { x: 0, y: 0 }, b: { x: 300, y: 90 }, c: { x: 120, y: 200 } });
  });
});
