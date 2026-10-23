import { msg } from '../i18n';
import { isAuto, literalOf, type Literal } from '../ir';
import { dimEq, type Dim, type TensorType } from '../shape';
import type { Bilingual, InferResult, ParamSpec, ParamType, Params, Resolved } from './types';

export const L = (en: string, zh: string): Bilingual => ({ en, zh });

type Extra = Partial<Omit<ParamSpec, 'name' | 'type' | 'label'>>;
const param =
  (type: ParamType) =>
  (name: string, zh: string, extra: Extra = {}): ParamSpec => ({ name, type, label: L(name, zh), ...extra });

/** Parameter builders: P.int('out_features', '输出特征数', { initial: 64 }). */
export const P = {
  int: param('int'),
  float: param('float'),
  bool: param('bool'),
  enum: param('enum'),
  ints: param('ints'),
  dims: param('dims'),
  shape: param('shape'),
};

export const inplaceParam = () => P.bool('inplace', '原地操作', { default: false });

export const IN = [{ name: 'in' }];
export const OUT = [{ name: 'out' }];

export const ok = (outputs: TensorType[], resolved: Resolved): InferResult => ({ ok: true, outputs, resolved });
export const fail = (error: string): InferResult => ({ ok: false, error });

export const lit = (params: Params, name: string): Literal | undefined => literalOf(params[name]);
export const num = (params: Params, name: string): number => literalOf(params[name]) as number;
export const flag = (params: Params, name: string): boolean => literalOf(params[name]) === true;

/** An 'ints' value expanded to `arity` numbers. */
export function tuple(value: Literal | undefined, arity: number): number[] {
  return Array.isArray(value) ? (value as number[]) : Array<number>(arity).fill(value as number);
}

/** All literal params, with auto params set from `auto`. */
export function resolveParams(params: Params, auto: Resolved = {}): Resolved {
  const out: Resolved = {};
  for (const [k, v] of Object.entries(params)) out[k] = isAuto(v) ? (auto[k] ?? null) : (literalOf(v) ?? null);
  return out;
}

/**
 * A feature size that can be inferred: `auto` takes the actual dimension,
 * a literal must agree with it. Returns the size, or an error message.
 * `what` names the dimension and must already be localized with msg().
 */
export function feature(params: Params, name: string, actual: Dim | undefined, what: string): number | string {
  if (actual === undefined) return msg(`缺少${what}`, `${what} is missing`);
  if (isAuto(params[name])) {
    return typeof actual === 'number'
      ? actual
      : msg(`无法从符号维度 ${actual} 推断 ${name}，请填写具体数值`, `cannot infer ${name} from the symbolic dimension ${actual}; enter a concrete value`);
  }
  const value = literalOf(params[name]) as number;
  return dimEq(actual, value) === false ? msg(`${what}为 ${actual}，但 ${name} = ${value}`, `${what} is ${actual}, but ${name} = ${value}`) : value;
}

/** `what` is a code name (a class or port name), the same in every locale. */
export function rankError(x: TensorType, ranks: number[], what: string): string | undefined {
  if (ranks.includes(x.shape.length)) return undefined;
  return msg(
    `${what}需要 ${ranks.join(' 或 ')} 维输入，收到 ${x.shape.length} 维`,
    `${what} expects ${ranks.map((r) => `${r}-D`).join(' or ')} input, got ${x.shape.length}-D`,
  );
}

/** Shows a tuple compactly: 3, or 3×5. */
export const showTuple = (v: Literal | undefined): string =>
  Array.isArray(v) ? (new Set(v).size === 1 ? String(v[0]) : v.join('×')) : String(v);
