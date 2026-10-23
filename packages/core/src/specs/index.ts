import { literalOf } from '../ir';
import { ACTIVATION_SPECS } from './activation';
import { ATTENTION_SPECS } from './attention';
import { CONV_SPECS } from './conv';
import { DROPOUT_SPECS } from './dropout';
import { IO_SPECS } from './io';
import { LINEAR_SPECS } from './linear';
import { MATH_SPECS } from './math';
import { NORM_SPECS } from './norm';
import { POOL_SPECS } from './pool';
import { RECURRENT_SPECS } from './recurrent';
import { RESIZE_SPECS } from './resize';
import { TENSOR_SPECS } from './tensor';
import type { NodeSpec, OutputSpec, Params } from './types';

export * from './types';

export const BUILTIN_SPECS: NodeSpec[] = [
  ...IO_SPECS,
  ...LINEAR_SPECS,
  ...CONV_SPECS,
  ...RESIZE_SPECS,
  ...POOL_SPECS,
  ...NORM_SPECS,
  ...ACTIVATION_SPECS,
  ...DROPOUT_SPECS,
  ...RECURRENT_SPECS,
  ...ATTENTION_SPECS,
  ...TENSOR_SPECS,
  ...MATH_SPECS,
];

const registry = new Map(BUILTIN_SPECS.map((s) => [s.type, s]));
if (registry.size !== BUILTIN_SPECS.length) throw new Error('Duplicate NodeSpec type');

export function getSpec(type: string | undefined): NodeSpec | undefined {
  return type === undefined ? undefined : registry.get(type);
}

export function allSpecs(): NodeSpec[] {
  return [...registry.values()];
}

/** Parameter values for a freshly created node. */
export function initialParams(spec: NodeSpec): Params {
  const params: Params = {};
  for (const p of spec.params) {
    if (p.initial === 'auto') params[p.name] = { auto: true };
    else if (p.initial !== undefined) params[p.name] = { literal: p.initial };
    else if (p.default !== undefined) params[p.name] = { literal: p.default };
  }
  return params;
}

const MAX_DYNAMIC_OUTPUTS = 64;

/** The node's output ports; for dynamic outputs this depends on the parameters. */
export function outputsOf(spec: NodeSpec, params: Params): OutputSpec[] {
  if (!spec.dynamicOutputs) return spec.outputs;
  const { param, prefix } = spec.dynamicOutputs;
  const n = Number(literalOf(params[param]) ?? 0);
  const count = Number.isInteger(n) ? Math.max(0, Math.min(n, MAX_DYNAMIC_OUTPUTS)) : 0;
  return Array.from({ length: count }, (_, i) => ({ name: `${prefix}${i}`, path: [i] }));
}

/** For a variadic input "in", port "in3" has index 3. */
export function variadicIndex(port: string, base: string): number | undefined {
  if (!port.startsWith(base)) return undefined;
  const rest = port.slice(base.length);
  return /^\d+$/.test(rest) ? Number(rest) : undefined;
}
