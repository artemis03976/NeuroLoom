// NeuroLoom IR: the single source of truth for a project (PRD section 16.1).
// A project is a set of module definitions (one per nn.Module class); one of
// them is the entry model. Each definition is a dataflow graph whose nodes are
// layers, ops, instances of other modules, code blocks, inputs and outputs.

export const SCHEMA_VERSION = '0.3';

export type NodeId = string;
export type ModuleId = string;

/** Parameter values; arrays hold tuples such as kernel_size (3, 5). */
export type Literal = number | string | boolean | null | Literal[];

/**
 * A parameter is a literal, `auto` ("infer it from the input shape"), or a
 * Python expression over the module's hyperparameters and constants, such as
 * "d_model * 4". Expressions are kept as written and exported verbatim.
 */
export type ParamValue = { literal: Literal } | { auto: true } | { expr: string };

export type NodeKind = 'layer' | 'op' | 'module' | 'code' | 'input' | 'output';

/** Type of every code block node. */
export const CODE_TYPE = 'neuroloom.Code';

/**
 * A layer or module instance repeated N times in series (PRD F4.4): each copy's
 * output feeds the next copy's first input; other inputs go to every copy.
 * Exported as nn.ModuleList plus a for loop.
 */
export interface Repeat {
  count: ParamValue;
  /** Parameters that differ for the first copy, e.g. stride=2 at the start of a ResNet stage. */
  first?: Record<string, ParamValue>;
}

/** Python statements kept verbatim (PRD F6.3): what the importer could not turn into nodes. */
export interface CodeBlock {
  /** Dedented statements. */
  source: string;
  /** Variables the statements read; one input port each. */
  inputs: string[];
  /** Variables the statements define for later use; one output port each. */
  outputs: string[];
}

export interface NodeInst {
  /** Stable id, decoupled from the variable name. */
  id: NodeId;
  kind: NodeKind;
  /** NodeSpec type such as "torch.nn.Linear"; the ModuleId for module nodes; CODE_TYPE for code blocks. */
  type: string;
  /** Python identifier used in the generated code, e.g. "fc1". */
  name: string;
  /** Layer and op parameters; for module nodes, hyperparameter values by name. */
  params: Record<string, ParamValue>;
  repeat?: Repeat;
  code?: CodeBlock;
  /**
   * Declared output shapes ("B, 512") for outputs that inference cannot see
   * into: code blocks and opaque modules. Keyed by output port.
   */
  shapes?: Record<string, string>;
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

/** A constructor argument of a module (PRD 9.3), e.g. `out_ch: int = 64`. */
export interface HyperParam {
  name: string;
  /** Python annotation such as "int"; absent means none. */
  annotation?: string;
  /** Absent: a required argument. */
  default?: { literal: Literal } | { expr: string };
  /** Declared after `*` in __init__: callers pass it by name. */
  keywordOnly?: boolean;
}

/** A plain attribute set in __init__ (`self.scale = head_dim ** -0.5`), usable by name in expressions. */
export interface Constant {
  name: string;
  expr: string;
}

export interface ModuleDef {
  id: ModuleId;
  /** Class name; for a third-party class, the name the preamble imports (may be dotted). */
  className: string;
  hyperparams: HyperParam[];
  /** Name of a `**kwargs` that __init__ accepts but never reads; written back so callers passing extra keywords still work. */
  kwargs?: string;
  constants?: Constant[];
  nodes: Record<NodeId, NodeInst>;
  edges: Edge[];
  /** Canvas positions. Kept apart from semantics so moving a node is not a semantic change. */
  layout: Record<NodeId, { x: number; y: number }>;
  /**
   * forward() argument order (input nodes) and return order (output nodes).
   * Explicit, because neither canvas position nor JSON key order may change a model's interface.
   * For module nodes these are also the ports: the input and output node ids.
   */
  interface: {
    inputs: NodeId[];
    outputs: NodeId[];
    /**
     * Defaults of optional forward() arguments, as Python source (`targets=None` is
     * { [targets]: "None" }). Callers may leave these ports unconnected, and an entry
     * input with a default needs no example shape.
     */
    defaults?: Record<NodeId, string>;
  };
  /**
   * Exported inline as nn.Sequential(...) of its nodes in chain order instead
   * of as a class. Has no hyperparameters: its expressions are in the scope of
   * the module that instantiates it.
   */
  sequential?: boolean;
  /**
   * A class kept as Python source instead of a graph (import level L3), or,
   * without source, a third-party class imported by the preamble. Only the
   * interface nodes are meaningful.
   */
  opaque?: {
    source?: string;
    /**
     * Instances pass constructor arguments as written: hyperparameters "#0",
     * "#1", … are positional, the rest keywords. Always so without source.
     */
    verbatimArgs?: boolean;
  };
  /** Statements appended to __init__ verbatim (weight initialization, buffers). */
  initCode?: string;
  /** Other class members (helper methods, class attributes), verbatim and dedented. */
  members?: string;
}

/** How exported code is written (PRD F5.3). Every field is optional; absent means the default. */
export interface ExportOptions {
  /** `# [B, 64, 56, 56]` after forward() lines. Default true. */
  shapeComments?: boolean;
  /** Annotations in signatures (`x: torch.Tensor`, `-> torch.Tensor`, `width: int`). Default true. */
  typeHints?: boolean;
  /** How many calls may nest in one line (`F.relu(self.bn1(self.conv1(x)))` is 2); 0 writes one step per line. Default 0. */
  nesting?: number;
  /** Repeats with one input and one output as an nn.Sequential called once, instead of nn.ModuleList + for. Default false. */
  sequentialRepeats?: boolean;
  /** Class docstrings with the parameter count. Default false. */
  docstrings?: boolean;
  /** A comment naming the project at the top of each file. Default false. */
  header?: boolean;
  /** `if __name__ == "__main__":` that runs the model once on example inputs and checks the output shapes. Default false. */
  selfTest?: boolean;
  /** One file per class plus __init__.py, instead of a single file. Default false. */
  splitFiles?: boolean;
}

export interface LoomProject {
  schemaVersion: typeof SCHEMA_VERSION;
  name: string;
  entry: ModuleId;
  modules: Record<ModuleId, ModuleDef>;
  /** Top-of-file Python that code blocks and opaque classes need (imports, helpers), verbatim. */
  preamble?: string;
  exportOptions?: ExportOptions;
}

export const literal = (value: Literal): ParamValue => ({ literal: value });
export const auto = (): ParamValue => ({ auto: true });
export const expr = (text: string): ParamValue => ({ expr: text });

export function literalOf(value: ParamValue | undefined): Literal | undefined {
  return value && 'literal' in value ? value.literal : undefined;
}

export function isAuto(value: ParamValue | undefined): boolean {
  return !!value && 'auto' in value;
}

export function exprOf(value: ParamValue | undefined): string | undefined {
  return value && 'expr' in value ? value.expr : undefined;
}

export function entryModule(project: LoomProject): ModuleDef {
  const mod = project.modules[project.entry];
  if (!mod) throw new Error(`Entry module "${project.entry}" does not exist`);
  return mod;
}

/** A module definition with no nodes yet. */
export function emptyModule(id: ModuleId, className: string): ModuleDef {
  return { id, className, hyperparams: [], nodes: {}, edges: [], layout: {}, interface: { inputs: [], outputs: [] } };
}
