import { msg } from './i18n';

/**
 * A dimension is a concrete size, a named symbol such as "B" (batch), or a
 * product / sum of those such as "B*T" or "C+64". "?" means unknown.
 */
export type Dim = number | string;

export interface TensorType {
  shape: Dim[];
  dtype: string;
}

export const UNKNOWN = '?';
const SYMBOL = /^[A-Za-z_][A-Za-z0-9_]*$/;

/** Parses "B, 3, 224, 224" (brackets optional). */
export function parseShape(text: string): { ok: true; shape: Dim[] } | { ok: false; error: string } {
  const body = text.trim().replace(/^[[(]/, '').replace(/[\])]$/, '').trim();
  if (body === '') return { ok: false, error: msg('形状不能为空', 'the shape cannot be empty') };
  const shape: Dim[] = [];
  for (const raw of body.split(',')) {
    const token = raw.trim();
    if (/^\d+$/.test(token)) {
      const n = Number(token);
      if (n === 0) return { ok: false, error: msg('维度大小必须大于 0', 'dimension sizes must be greater than 0') };
      shape.push(n);
    } else if (SYMBOL.test(token)) {
      shape.push(token);
    } else if (token.split('*').every((f) => SYMBOL.test(f.trim()) || /^[1-9]\d*$/.test(f.trim()))) {
      // A product such as B*T or B*4, in canonical form.
      shape.push(dimProduct(token.split('*').map((f) => (/^\d+$/.test(f.trim()) ? Number(f.trim()) : f.trim()))));
    } else {
      return { ok: false, error: msg(`无法识别的维度“${token}”`, `unrecognized dimension "${token}"`) };
    }
  }
  return { ok: true, shape };
}

/** Canonical text form stored in the IR, e.g. "B, 784". */
export const shapeText = (shape: Dim[]): string => shape.join(', ');

/** Display form used on edges and in code comments, e.g. "[B, 784]". */
export const formatShape = (shape: Dim[]): string => `[${shape.join(', ')}]`;

// ---------------------------------------------------------------- symbolic dims

/** coef × sym₁ × sym₂ …; null for sums and unknowns. */
interface Monomial {
  coef: number;
  syms: string[];
}

function toMonomial(d: Dim): Monomial | null {
  if (typeof d === 'number') return { coef: d, syms: [] };
  let coef = 1;
  const syms: string[] = [];
  for (const factor of d.split('*')) {
    const f = factor.trim();
    if (/^\d+$/.test(f)) coef *= Number(f);
    else if (SYMBOL.test(f)) syms.push(f);
    else return null;
  }
  return { coef, syms: syms.sort() };
}

function fromMonomial(m: Monomial): Dim {
  if (m.syms.length === 0) return m.coef;
  const syms = m.syms.join('*');
  return m.coef === 1 ? syms : `${syms}*${m.coef}`;
}

export function dimMul(a: Dim, b: Dim): Dim {
  const x = toMonomial(a);
  const y = toMonomial(b);
  return x && y ? fromMonomial({ coef: x.coef * y.coef, syms: [...x.syms, ...y.syms].sort() }) : UNKNOWN;
}

export const dimProduct = (dims: Dim[]): Dim => dims.reduce<Dim>((acc, d) => dimMul(acc, d), 1);

/** a / b when it divides exactly, otherwise undefined. */
export function dimDiv(a: Dim, b: Dim): Dim | undefined {
  const x = toMonomial(a);
  const y = toMonomial(b);
  if (!x || !y || y.coef === 0 || x.coef % y.coef !== 0) return undefined;
  const syms = [...x.syms];
  for (const s of y.syms) {
    const i = syms.indexOf(s);
    if (i < 0) return undefined;
    syms.splice(i, 1);
  }
  return fromMonomial({ coef: x.coef / y.coef, syms });
}

export function dimAdd(a: Dim, b: Dim): Dim {
  if (typeof a === 'number' && typeof b === 'number') return a + b;
  if (a === UNKNOWN || b === UNKNOWN) return UNKNOWN;
  const x = toMonomial(a);
  const y = toMonomial(b);
  if (x && y && x.syms.join('*') === y.syms.join('*')) return fromMonomial({ coef: x.coef + y.coef, syms: x.syms });
  return `${a}+${b}`;
}

/** true / false when certain, undefined when it depends on symbol values. */
export function dimEq(a: Dim, b: Dim): boolean | undefined {
  if (typeof a === 'number' && typeof b === 'number') return a === b;
  if (a === UNKNOWN || b === UNKNOWN) return undefined;
  if (typeof a === 'string' && typeof b === 'string' && a === b) return true;
  return undefined;
}

/** Normalizes a possibly negative dimension index; undefined when out of range. */
export function axis(dim: number, rank: number): number | undefined {
  const i = dim < 0 ? dim + rank : dim;
  return i >= 0 && i < rank ? i : undefined;
}

/** NumPy-style broadcasting. Returns an error message for incompatible shapes. */
export function broadcast(shapes: Dim[][]): Dim[] | string {
  const rank = Math.max(...shapes.map((s) => s.length));
  const out: Dim[] = [];
  for (let i = 1; i <= rank; i++) {
    const dims = shapes.map((s) => s.at(-i)).filter((d): d is Dim => d !== undefined && d !== 1);
    if (dims.length === 0) {
      out.unshift(1);
      continue;
    }
    const numbers = [...new Set(dims.filter((d) => typeof d === 'number'))];
    if (numbers.length > 1) {
      return msg(`无法广播：倒数第 ${i} 维为 ${numbers.join(' 和 ')}`, `cannot broadcast: dimension -${i} has sizes ${numbers.join(' and ')}`);
    }
    out.unshift(numbers[0] ?? dims[0]!);
  }
  return out;
}

/** Evaluates a dimension with every symbol set to `value` (NaN when unknown). */
export function evalDim(d: Dim, value: number): number {
  if (typeof d === 'number') return d;
  return d.split('+').reduce((sum, term) => {
    const product = term.split('*').reduce((p, f) => {
      const t = f.trim();
      if (/^\d+$/.test(t)) return p * Number(t);
      return SYMBOL.test(t) ? p * value : NaN;
    }, 1);
    return sum + product;
  }, 0);
}

/**
 * A dimension from its sizes in two runs where the symbols had different values
 * (`first`, `second`): equal sizes are constants, others a product of symbols
 * times a constant (B, B*T, B*4). Undefined when no such product fits (sums).
 */
export function dimFromRuns(a: number, b: number, first: Record<string, number>, second: Record<string, number>): Dim | undefined {
  if (a === b) return a;
  const symbols = Object.keys(first);
  for (let mask = 1; mask < 1 << symbols.length; mask++) {
    const picked = symbols.filter((_, i) => mask & (1 << i));
    const pa = picked.reduce((p, s) => p * first[s]!, 1);
    const pb = picked.reduce((p, s) => p * (second[s] ?? 1), 1);
    if (a % pa === 0 && b % pb === 0 && a / pa === b / pb) return dimProduct([...picked, a / pa]);
  }
  return undefined;
}

/** Replaces every symbolic dimension with a concrete value. */
export function concretize(shape: Dim[], value: number): number[] {
  return shape.map((d) => evalDim(d, value));
}
