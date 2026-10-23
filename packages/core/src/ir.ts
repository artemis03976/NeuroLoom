// NeuroLoom IR: the single source of truth for a project (PRD section 16.1).
// M0 implements the subset needed for flat, single-module graphs; hyperparameters,
// module instances, repeat blocks and code blocks arrive in later milestones.

export const SCHEMA_VERSION = '0.1';

export type NodeId = string;
export type ModuleId = string;

export type Literal = number | string | boolean | null;

/** A parameter is either a literal or `auto`, meaning "infer it from the input shape". */
export type ParamValue = { literal: Literal } | { auto: true };

export type NodeKind = 'layer' | 'op' | 'input' | 'output';

export interface NodeInst {
  /** Stable id, decoupled from the variable name. */
  id: NodeId;
  kind: NodeKind;
  /** NodeSpec type, e.g. "torch.nn.Linear". */
  type: string;
  /** Python identifier used in the generated code, e.g. "fc1". */
  name: string;
  params: Record<string, ParamValue>;
}

export interface PortRef {
  node: NodeId;
  port: string;
}

export interface Edge {
  id: string;
  source: PortRef;
  target: PortRef;
}

export interface ModuleDef {
  id: ModuleId;
  className: string;
  nodes: Record<NodeId, NodeInst>;
  edges: Edge[];
  /** Canvas positions. Kept apart from semantics so moving a node is not a semantic change. */
  layout: Record<NodeId, { x: number; y: number }>;
}

export interface LoomProject {
  schemaVersion: typeof SCHEMA_VERSION;
  name: string;
  entry: ModuleId;
  modules: Record<ModuleId, ModuleDef>;
}

export const literal = (value: Literal): ParamValue => ({ literal: value });
export const auto = (): ParamValue => ({ auto: true });

export function literalOf(value: ParamValue | undefined): Literal | undefined {
  return value && 'literal' in value ? value.literal : undefined;
}

export function isAuto(value: ParamValue | undefined): boolean {
  return !!value && 'auto' in value;
}

export function entryModule(project: LoomProject): ModuleDef {
  const mod = project.modules[project.entry];
  if (!mod) throw new Error(`Entry module "${project.entry}" does not exist`);
  return mod;
}
