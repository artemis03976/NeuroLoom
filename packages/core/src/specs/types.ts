// NodeSpec: the declarative definition of a node type (PRD section 16.2).
// One spec drives the palette, the inspector form, shape inference, parameter
// counting, code generation, the Python importer (via the exported manifest)
// and its own round-trip test (via `example`).
import type { Literal, NodeKind, ParamValue } from '../ir';
import type { TensorType } from '../shape';

export type Category =
  | 'io'
  | 'linear'
  | 'conv'
  | 'pool'
  | 'norm'
  | 'activation'
  | 'dropout'
  | 'recurrent'
  | 'attention'
  | 'tensor'
  | 'math'
  | 'module'
  | 'code';

export interface Bilingual {
  en: string;
  zh: string;
}

export type ParamType =
  /** Integer. */
  | 'int'
  | 'float'
  | 'bool'
  /** One of `options`. */
  | 'enum'
  /** Input-node shape text such as "B, 3, 224, 224". */
  | 'shape'
  /** An int or a tuple of `arity` ints (kernel_size); may also be one of `options` (padding="same"). */
  | 'ints'
  /** A variable-length list of ints, negative allowed (permute dims, reshape shape). */
  | 'dims';

export interface ParamSpec {
  name: string;
  type: ParamType;
  /** Absent means required. */
  default?: Literal;
  /** Value used when a node is first created from the palette. */
  initial?: Literal | 'auto';
  allowAuto?: boolean;
  /** None is a valid value. */
  nullable?: boolean;
  min?: number;
  /** For 'ints': the tuple length (2 for Conv2d). */
  arity?: number;
  /** For 'enum', and string alternatives for 'ints'. */
  options?: string[];
  /** Ops: always written as a keyword (F.softmax(x, dim=-1)). */
  keyword?: boolean;
  /** Ops: written positionally even when optional (torch.flatten(x, 1)). */
  positional?: boolean;
  /** Ops: list values are spread as separate arguments (x.permute(0, 2, 1)). */
  spread?: boolean;
  label: Bilingual;
}

export interface PortSpec {
  name: string;
  /** May stay unconnected. */
  optional?: boolean;
  /** Accepts any number of edges, stored as ports name0, name1, … (torch.cat). */
  variadic?: boolean;
}

export interface OutputSpec {
  name: string;
  /** Index path into the Python return value: [1, 0] is result[1][0]. Absent = the whole result. */
  path?: number[];
}

export type Resolved = Record<string, Literal>;
export type Params = Record<string, ParamValue>;
/** Input tensors in port order; variadic ports contribute every connected tensor. Optional ports may be undefined. */
export type Inputs = (TensorType | undefined)[];

export type InferResult = { ok: true; outputs: TensorType[]; resolved: Resolved } | { ok: false; error: string };

export interface CodegenSpec {
  /** Layers: constructor such as "nn.Conv2d". */
  ctor?: string;
  /** Ops: function such as "torch.cat" or "F.relu". */
  fn?: string;
  /** Ops: tensor method such as "flatten" (called on the first input). */
  method?: string;
  /** Ops: binary operator such as "+" or "@". */
  operator?: string;
  /** How a tuple result is unpacked, e.g. "{out}, ({h_n}, {c_n})". Default: one name per output, flat. */
  unpack?: string;
  /**
   * A tensor the module holds (nn.Parameter, a buffer): created in __init__ as
   * `self.x = nn.Parameter(...)` ("parameter") or `self.register_buffer("x", ...)`
   * ("buffer"), and read in forward() as `self.x`.
   */
  state?: 'parameter' | 'buffer';
}

/** Python argument order of an op, so the importer can map call arguments back to ports and params. */
export type PythonArg =
  | { name: string; port: string; scalarParam?: string }
  | { name: string; ports: string }
  | { name: string; param: string };

export interface PythonCall {
  /** Qualified functions, e.g. "torch.flatten", "operator.add". */
  functions?: string[];
  /** Tensor methods, e.g. "flatten" (the receiver is the first argument). */
  methods?: string[];
  args: PythonArg[];
}

export interface Example {
  inputs: { shape: string; dtype?: string }[];
  params?: Record<string, Literal | 'auto'>;
}

export interface NodeSpec {
  type: string;
  kind: NodeKind;
  category: Category;
  displayName: Bilingual;
  summary: Bilingual;
  docUrl?: string;
  params: ParamSpec[];
  inputs: PortSpec[];
  outputs: OutputSpec[];
  /** Outputs whose number comes from a parameter (torch.chunk): ports prefix0 … prefixN-1. */
  dynamicOutputs?: { param: string; prefix: string };
  /** Base for generated names: "fc" gives fc1, fc2 for layers; ops follow torch.fx naming (cat, cat_1). */
  defaultName: string;
  infer(inputs: Inputs, params: Params): InferResult;
  paramCount?(resolved: Resolved): number;
  /**
   * Forward FLOPs of one call (PRD F8.2), counted the way
   * torch.utils.flop_counter.FlopCounterMode counts them: a multiply-add is 2
   * FLOPs, and element-wise ops, normalization and activations count 0.
   * Shapes are concrete (symbolic dimensions already given values). Absent: 0.
   */
  flops?(resolved: Resolved, inputs: number[][], outputs: number[][]): number;
  /** How the output shape follows from the input (PRD F2.4), e.g. "H_out = ⌊(H + 2p − d(k − 1) − 1) / s⌋ + 1". */
  shapeDoc?: Bilingual;
  /** One-line summary shown on the node body. */
  describe?(resolved: Resolved): string;
  codegen: CodegenSpec;
  /** Ops only; layers are recognized by their class (codegen.ctor). */
  python?: PythonCall;
  /** A small graph around this node, used by the round-trip tests. Required for everything but io. */
  example?: Example;
}
