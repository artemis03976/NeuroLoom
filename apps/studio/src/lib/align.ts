// Align and distribute selected nodes (PRD F1.2).

export type AlignMode = 'left' | 'centerX' | 'right' | 'top' | 'centerY' | 'bottom' | 'distributeX' | 'distributeY';

export interface Box {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * New positions for the boxes. Aligning lines up an edge or the centre with the
 * selection's extreme or middle; distributing spaces the boxes evenly between the
 * outermost two, keeping their order.
 */
export function alignPositions(boxes: Box[], mode: AlignMode): Record<string, { x: number; y: number }> {
  const out: Record<string, { x: number; y: number }> = {};
  const left = Math.min(...boxes.map((b) => b.x));
  const right = Math.max(...boxes.map((b) => b.x + b.width));
  const top = Math.min(...boxes.map((b) => b.y));
  const bottom = Math.max(...boxes.map((b) => b.y + b.height));
  const at = (b: Box, x: number, y: number) => (out[b.id] = { x: Math.round(x), y: Math.round(y) });

  if (mode === 'distributeX' || mode === 'distributeY') {
    const horizontal = mode === 'distributeX';
    const sorted = [...boxes].sort((a, b) => (horizontal ? a.x - b.x : a.y - b.y));
    const size = (b: Box) => (horizontal ? b.width : b.height);
    const span = (horizontal ? right - left : bottom - top) - sorted.reduce((s, b) => s + size(b), 0);
    const gap = sorted.length > 1 ? span / (sorted.length - 1) : 0;
    let cursor = horizontal ? left : top;
    for (const b of sorted) {
      if (horizontal) at(b, cursor, b.y);
      else at(b, b.x, cursor);
      cursor += size(b) + gap;
    }
    return out;
  }

  for (const b of boxes) {
    switch (mode) {
      case 'left':
        at(b, left, b.y);
        break;
      case 'centerX':
        at(b, (left + right) / 2 - b.width / 2, b.y);
        break;
      case 'right':
        at(b, right - b.width, b.y);
        break;
      case 'top':
        at(b, b.x, top);
        break;
      case 'centerY':
        at(b, b.x, (top + bottom) / 2 - b.height / 2);
        break;
      case 'bottom':
        at(b, b.x, bottom - b.height);
        break;
    }
  }
  return out;
}
