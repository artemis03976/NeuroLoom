// Text <-> value for the inspector's parameter fields, by ParamSpec type.
import { checkParamValue, type Literal, type ParamSpec } from '@neuroloom/core';

/** How a value is shown in its text field ("3, 5" for a tuple). */
export function formatParam(value: Literal | undefined): string {
  if (value === undefined || value === null) return '';
  if (Array.isArray(value)) return value.join(', ');
  return String(value);
}

const INT = /^-?\d+$/;

/** Parses field text; returns the value or an error message. */
export function parseParam(text: string, p: ParamSpec): { ok: true; value: Literal } | { ok: false; error: string } {
  const t = text.trim();
  let value: Literal;
  switch (p.type) {
    case 'int':
      if (!INT.test(t)) return { ok: false, error: '请输入整数' };
      value = Number(t);
      break;
    case 'float': {
      const n = Number(t);
      if (t === '' || !Number.isFinite(n)) return { ok: false, error: '请输入数字' };
      value = n;
      break;
    }
    case 'ints': {
      if (p.options?.includes(t)) {
        value = t;
        break;
      }
      const parts = t.replace(/[()[\]]/g, '').split(',').map((s) => s.trim()).filter(Boolean);
      if (parts.length === 0 || !parts.every((s) => INT.test(s))) {
        return { ok: false, error: p.options ? `请输入整数、整数元组或 ${p.options.join(' / ')}` : '请输入整数或整数元组，如 3, 5' };
      }
      value = parts.length === 1 ? Number(parts[0]) : parts.map(Number);
      break;
    }
    case 'dims': {
      const parts = t.replace(/[()[\]]/g, '').split(',').map((s) => s.trim()).filter(Boolean);
      if (parts.length === 0 || !parts.every((s) => INT.test(s))) return { ok: false, error: '请输入整数列表，如 0, 2, 1' };
      value = parts.map(Number);
      break;
    }
    default:
      value = text;
  }
  const error = checkParamValue(p, value);
  return error ? { ok: false, error } : { ok: true, value };
}

/** A sensible non-null value when "None" is switched off. */
export function fallbackValue(p: ParamSpec): Literal {
  if (p.initial !== undefined && p.initial !== 'auto' && p.initial !== null) return p.initial;
  if (p.default !== undefined && p.default !== null) return p.default;
  if (p.type === 'bool') return false;
  if (p.type === 'enum') return p.options?.[0] ?? '';
  return p.type === 'float' ? 1.0 : 1;
}
