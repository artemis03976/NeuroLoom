// The part of every NodeSpec the Python importer needs, exported as
// python/neuroloom/specs.json (`npm run specs`). A test keeps the file in sync.
import type { Literal } from './ir';
import { BUILTIN_SPECS, type NodeSpec, type ParamType, type PythonCall } from './specs';
import { SCHEMA_VERSION } from './ir';

export interface ManifestParam {
  name: string;
  type: ParamType;
  /** Absent when the parameter is required. */
  default?: Literal;
  nullable?: boolean;
  arity?: number;
  spread?: boolean;
}

export interface ManifestEntry {
  type: string;
  kind: NodeSpec['kind'];
  /** Base for generated node names ("fc", "relu"). */
  defaultName: string;
  /** Layers: the torch.nn class, e.g. "torch.nn.Conv2d". */
  layerClass?: string;
  params: ManifestParam[];
  inputs: NodeSpec['inputs'];
  outputs: NodeSpec['outputs'];
  dynamicOutputs?: NodeSpec['dynamicOutputs'];
  python?: PythonCall;
}

export interface Manifest {
  schemaVersion: string;
  specs: ManifestEntry[];
}

export function exportManifest(): Manifest {
  const specs = BUILTIN_SPECS.filter((s) => s.kind === 'layer' || s.kind === 'op').map((s): ManifestEntry => {
    const entry: ManifestEntry = {
      type: s.type,
      kind: s.kind,
      defaultName: s.defaultName,
      params: s.params.map((p) => {
        const out: ManifestParam = { name: p.name, type: p.type };
        if (p.default !== undefined) out.default = p.default;
        if (p.nullable) out.nullable = true;
        if (p.arity !== undefined) out.arity = p.arity;
        if (p.spread) out.spread = true;
        return out;
      }),
      inputs: s.inputs,
      outputs: s.outputs,
    };
    if (s.codegen.ctor) entry.layerClass = s.codegen.ctor.replace(/^nn\./, 'torch.nn.');
    if (s.dynamicOutputs) entry.dynamicOutputs = s.dynamicOutputs;
    if (s.python) entry.python = s.python;
    return entry;
  });
  return { schemaVersion: SCHEMA_VERSION, specs };
}

export const manifestText = (): string => `${JSON.stringify(exportManifest(), null, 2)}\n`;
