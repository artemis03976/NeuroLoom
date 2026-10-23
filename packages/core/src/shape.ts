/** A dimension is a concrete size or a named symbol such as "B" (batch). */
export type Dim = number | string;

export interface TensorType {
  shape: Dim[];
  dtype: string;
}

const SYMBOL = /^[A-Za-z_][A-Za-z0-9_]*$/;

/** Parses "B, 3, 224, 224" (brackets optional). */
export function parseShape(text: string): { ok: true; shape: Dim[] } | { ok: false; error: string } {
  const body = text.trim().replace(/^[[(]/, '').replace(/[\])]$/, '').trim();
  if (body === '') return { ok: false, error: '形状不能为空' };
  const shape: Dim[] = [];
  for (const raw of body.split(',')) {
    const token = raw.trim();
    if (/^\d+$/.test(token)) {
      const n = Number(token);
      if (n === 0) return { ok: false, error: '维度大小必须大于 0' };
      shape.push(n);
    } else if (SYMBOL.test(token)) {
      shape.push(token);
    } else {
      return { ok: false, error: `无法识别的维度“${token}”` };
    }
  }
  return { ok: true, shape };
}

/** Canonical text form stored in the IR, e.g. "B, 784". */
export const shapeText = (shape: Dim[]): string => shape.join(', ');

/** Display form used on edges and in code comments, e.g. "[B, 784]". */
export const formatShape = (shape: Dim[]): string => `[${shape.join(', ')}]`;

/** Replaces every symbolic dimension with a concrete value. */
export function concretize(shape: Dim[], value: number): number[] {
  return shape.map((d) => (typeof d === 'number' ? d : value));
}
