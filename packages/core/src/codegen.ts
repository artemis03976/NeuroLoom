// IR -> PyTorch source (PRD section 10). Output is deterministic: the same graph
// always produces byte-identical code, and every value has one canonical spelling
// so that code -> graph -> code is a fixed point.
import { analyzeModule, analyzeProject, sequentialChain, usedByCode, variadicEdges, withDefaults, type Analysis, type ProjectAnalysis } from './analyze';
import { asOperand, renameInExpr, subtractOne } from './expr';
import { consumers, incomingByPort, interfaceOrder, portKey } from './graph';
import { msg } from './i18n';
import { exprOf, literalOf, type ExportOptions, type HyperParam, type Literal, type LoomProject, type ModuleDef, type ModuleId, type NodeId, type NodeInst, type ParamValue } from './ir';
import { defaultSymbolValue, shapeValue } from './profile';
import { dependencyOrder, portsOf, snakeCase } from './project';
import { formatShape, type TensorType } from './shape';
import { getSpec, type NodeSpec, type ParamSpec, type Resolved } from './specs';

export interface CodegenOptions {
  /** Records the shape of every code-block and opaque-module output when the code runs (see `PROBE_PRELUDE`). Ignores the export options. */
  probe?: boolean;
  /** Export options instead of the project's own. */
  export?: ExportOptions;
}

export interface CodegenResult {
  code: string;
  /** Per module: 1-based line numbers per node, its constructor line and its forward lines. */
  lineMap: Record<ModuleId, Record<NodeId, number[]>>;
  /** Per module: the line of its `class` statement. */
  classLines: Record<ModuleId, number>;
  /** Number of TODO(neuroloom) markers written for unfinished parts of the graph. */
  todos: number;
}

const I1 = '    ';
const I2 = I1 + I1;

/** Defines `_nl_probe`, which instrumented code calls with every value inference cannot see. */
export const PROBE_PRELUDE = `_NL_SHAPES = {}


def _nl_probe(module, node, port, value):
    if isinstance(value, torch.Tensor):
        _NL_SHAPES.setdefault(f"{module}/{node}/{port}", [list(value.shape), str(value.dtype).replace("torch.", "")])
    return value`;

/** Canonical form of a parameter value: uniform tuples collapse to a scalar (kernel_size (3, 3) -> 3). */
export function canonicalValue(value: Literal | undefined, p: ParamSpec): Literal | undefined {
  if (p.type === 'ints' && Array.isArray(value) && value.length > 0) {
    if (value.length === 1 && p.arity === undefined) return value[0]!;
    if (p.arity !== undefined && value.every((v) => v === value[0])) return value[0]!;
  }
  return value;
}

export function sameValue(a: Literal | undefined, b: Literal | undefined, p: ParamSpec): boolean {
  return JSON.stringify(canonicalValue(a, p)) === JSON.stringify(canonicalValue(b, p));
}

function pyNumber(value: number, float: boolean): string {
  if (float && Number.isInteger(value)) return `${value}.0`;
  return String(value);
}

export function pyLiteral(value: Literal | undefined, p?: ParamSpec, float = p?.type === 'float'): string {
  const v = p ? canonicalValue(value, p) : value;
  if (v === undefined || v === null) return 'None';
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (typeof v === 'number') return pyNumber(v, float);
  if (typeof v === 'string') return JSON.stringify(v);
  const items = v.map((item) => pyLiteral(item));
  return items.length === 1 ? `(${items[0]},)` : `(${items.join(', ')})`;
}

/** Simple assignment targets in a block of statements; code blocks may set more names than they export. */
function assignedNames(source: string): string[] {
  const names = new Set<string>();
  for (const line of source.split('\n')) {
    const m = /^\s*(?:for\s+(.+?)\s+in\b|([A-Za-z_][\w\s,()*]*?)\s*(?:[-+*/%@&|^]|\/\/|\*\*|<<|>>)?=(?!=))/.exec(line);
    const targets = m?.[1] ?? m?.[2];
    for (const name of targets?.match(/[A-Za-z_]\w*/g) ?? []) names.add(name);
  }
  return [...names];
}

/** Indents every non-empty line. */
const indent = (text: string, prefix: string) => text.split('\n').map((l) => (l.trim() ? prefix + l : ''));

/**
 * Builds an op call from the spec's Python signature. Arguments are positional
 * until one is skipped or written as a keyword; after that, keywords only.
 */
function opCall(
  spec: NodeSpec,
  value: (p: ParamSpec) => { literal: Literal | undefined } | { expr: string },
  portExpr: (port: string) => string | undefined,
  variadicExprs: (base: string) => string[],
): string {
  const py = spec.python!;
  const paramOf = (name: string) => spec.params.find((p) => p.name === name)!;
  const render = (p: ParamSpec) => {
    const v = value(p);
    return 'expr' in v ? v.expr : pyLiteral(v.literal, p);
  };
  const parts: string[] = [];
  let receiver: string | undefined;
  let keywordsOnly = false;
  const push = (name: string, text: string, keyword: boolean) => {
    if (keyword || keywordsOnly) {
      parts.push(`${name}=${text}`);
      keywordsOnly = true;
    } else parts.push(text);
  };

  py.args.forEach((arg, i) => {
    if ('ports' in arg) {
      push(arg.name, `[${variadicExprs(arg.ports).join(', ')}]`, false);
    } else if ('port' in arg) {
      const text = portExpr(arg.port);
      if (text !== undefined) {
        if (spec.codegen.method && i === 0) receiver = text;
        else push(arg.name, text, false);
      } else if (arg.scalarParam) {
        push(arg.name, render(paramOf(arg.scalarParam)), false);
      } else {
        keywordsOnly = true;
      }
    } else {
      const p = paramOf(arg.param);
      const v = value(p);
      if ('literal' in v && p.default !== undefined && sameValue(v.literal, p.default, p)) {
        keywordsOnly = true;
        return;
      }
      if ('literal' in v && p.spread && Array.isArray(v.literal) && !keywordsOnly) {
        parts.push(...v.literal.map((item) => pyLiteral(item)));
        return;
      }
      push(arg.name, render(p), p.keyword === true || (p.default !== undefined && !p.positional));
    }
  });

  // Operators take exactly two operands: a tensor and a tensor or constant.
  if (spec.codegen.operator) return `${parts[0]} ${spec.codegen.operator} ${asOperand(parts[1] ?? '', spec.codegen.operator)}`;
  if (spec.codegen.method) return `${receiver}.${spec.codegen.method}(${parts.join(', ')})`;
  return `${spec.codegen.fn}(${parts.join(', ')})`;
}

/** "(1, 1, d)" -> "1, 1, d" and "(d,)" -> "d"; undefined when the text is not a parenthesized tuple. */
function tupleItems(text: string): string | undefined {
  if (!text.startsWith('(') || !text.endsWith(')')) return undefined;
  let depth = 0;
  let comma = false;
  for (let i = 1; i < text.length - 1; i++) {
    const c = text[i]!;
    if ('([{'.includes(c)) depth++;
    else if (')]}'.includes(c)) depth--;
    if (depth < 0) return undefined; // "(a) + (b)" closes the first parenthesis early
    if (c === ',' && depth === 0) comma = true;
  }
  return comma ? text.slice(1, -1).replace(/,\s*$/, '') : undefined;
}

/** A sequential module whose children are named 0, 1, …, as nn.Sequential(a, b) names them. */
const numberedChain = (mod: ModuleDef) => sequentialChain(mod).every((n, i) => n.name === String(i));

/** A loop variable for a repeat: "blocks" -> "block". */
function singular(name: string): string {
  return name.length > 1 && name.endsWith('s') && !name.endsWith('ss') ? name.slice(0, -1) : `${name}_i`;
}

class ClassWriter {
  private readonly hyperparams: Set<string>;
  private readonly constants: Set<string>;

  constructor(
    private readonly gen: Generator,
    private readonly mod: ModuleDef,
    private readonly analyses: Analysis[],
  ) {
    this.hyperparams = new Set(mod.hyperparams.map((h) => h.name));
    this.constants = new Set((mod.constants ?? []).map((c) => c.name));
  }

  private get a0(): Analysis {
    return this.analyses[0]!;
  }

  /** An expression as written in __init__ (constants are attributes there) or forward (everything is). */
  expr(text: string, where: 'init' | 'forward'): string {
    return renameInExpr(text, (n) => (this.constants.has(n) || (where === 'forward' && this.hyperparams.has(n)) ? `self.${n}` : undefined));
  }

  /** Hyperparameters forward() needs, so __init__ keeps them as attributes. */
  private storedHyperparams(): string[] {
    const used = new Set<string>();
    for (const node of Object.values(this.mod.nodes)) {
      if (node.kind === 'op') {
        for (const v of Object.values(node.params)) {
          const text = exprOf(v);
          if (text) renameInExpr(text, (n) => void used.add(n));
        }
      }
    }
    const verbatim = [this.mod.initCode, this.mod.members, ...Object.values(this.mod.nodes).map((n) => n.code?.source)].join('\n');
    for (const m of verbatim.matchAll(/\bself\.([A-Za-z_]\w*)/g)) used.add(m[1]!);
    return this.mod.hyperparams.map((h) => h.name).filter((n) => used.has(n) && !this.constants.has(n));
  }

  /** A hyperparameter value or default: a literal (floats per annotation) or an expression. */
  private value(v: ParamValue | { literal: Literal } | { expr: string }, where: 'init' | 'forward', float = false): string {
    if ('expr' in v) return this.expr(v.expr, where);
    if ('literal' in v) return pyLiteral(v.literal, undefined, float);
    return 'None';
  }

  /**
   * Parameter values for a node's constructor, agreed by every context the
   * module is analyzed in. An `auto` parameter that resolves differently in
   * different instances cannot be written once; it is reported instead.
   */
  private resolvedFor(nodeId: NodeId, which: 'resolved' | 'resolvedFirst' = 'resolved'): { resolved?: Resolved; conflicts: string[] } {
    const node = this.mod.nodes[nodeId]!;
    const all = this.analyses.map((a) => a.nodes[nodeId]?.[which] ?? (which === 'resolvedFirst' ? a.nodes[nodeId]?.resolved : undefined)).filter((r): r is Resolved => !!r);
    const resolved = all[0];
    const conflicts: string[] = [];
    for (const [name, v] of Object.entries(node.params)) {
      if (!('auto' in v) || !resolved) continue;
      const values = [...new Set(all.map((r) => JSON.stringify(r[name] ?? null)))];
      if (values.length > 1) conflicts.push(msg(`${name} 在不同实例中推断为 ${values.join(' / ')}，请改用超参数表达式`, `${name} is inferred as ${values.join(' / ')} in different instances; use a hyperparameter expression`));
    }
    return { resolved, conflicts };
  }

  /** `nn.Conv2d(3, 16, 3, padding=1)`: required params positionally, optional ones as keywords when not default. */
  layerCtor(spec: NodeSpec, node: NodeInst, resolved: Resolved | undefined, overrides: Record<string, ParamValue> = {}): { text: string; missing: string[] } {
    const args: string[] = [];
    const missing: string[] = [];
    const params = withDefaults(spec, { ...node.params, ...overrides });
    for (const p of spec.params) {
      const raw = params[p.name];
      const text = exprOf(raw);
      if (text !== undefined) {
        args.push(p.default === undefined ? this.expr(text, 'init') : `${p.name}=${this.expr(text, 'init')}`);
        continue;
      }
      const value = resolved?.[p.name] ?? literalOf(raw);
      if (p.default === undefined) {
        if (value === undefined || value === null) {
          if (value === undefined || !p.nullable) missing.push(p.name);
        }
        args.push(pyLiteral(value, p));
      } else if (value !== undefined && !sameValue(value, p.default, p)) {
        args.push(`${p.name}=${pyLiteral(value, p)}`);
      }
    }
    return { text: `${spec.codegen.ctor}(${args.join(', ')})`, missing };
  }

  /** `Block(64, stride=2)` for a module instance, or the inline nn.Sequential(...) of a sequential module. */
  moduleCtor(node: NodeInst, overrides: Record<string, ParamValue> = {}, child?: Analysis): { text: string; missing: string[] } {
    const def = this.gen.project.modules[node.type];
    if (!def) return { text: 'None', missing: [node.type] };
    if (def.sequential) {
      const missing: string[] = [];
      const inner = new ClassWriter(this.gen, { ...def, hyperparams: this.mod.hyperparams, constants: this.mod.constants }, child ? [child] : this.gen.analysesOf(def.id));
      const chain = sequentialChain(def);
      const items = chain.map((n) => {
        const r = inner.ctorOf(n, child);
        missing.push(...r.missing);
        return r.text;
      });
      // Named children (features.conv0) keep their names, and so their state_dict keys.
      if (!numberedChain(def)) return { text: `nn.Sequential(OrderedDict([${items.map((text, i) => `(${JSON.stringify(chain[i]!.name)}, ${text})`).join(', ')}]))`, missing };
      return { text: `nn.Sequential(${items.join(', ')})`, missing };
    }
    const verbatim = !!def.opaque && (!def.opaque.source || !!def.opaque.verbatimArgs);
    const given = { ...node.params, ...overrides };
    const args: string[] = [];
    const missing: string[] = [];
    for (const hp of def.hyperparams) {
      const v = given[hp.name];
      const float = hp.annotation === 'float';
      if (hp.name.startsWith('#')) {
        if (v) args.push(this.value(v, 'init', float));
      } else if (verbatim) {
        if (v) args.push(`${hp.name}=${this.value(v, 'init', float)}`);
      } else if (hp.keywordOnly && !hp.default) {
        if (!v) missing.push(hp.name);
        args.push(`${hp.name}=${v ? this.value(v, 'init', float) : 'None'}`);
      } else if (!hp.default) {
        if (!v) missing.push(hp.name);
        args.push(v ? this.value(v, 'init', float) : 'None');
      } else if (v && JSON.stringify(v) !== JSON.stringify(hp.default)) {
        args.push(`${hp.name}=${this.value(v, 'init', float)}`);
      }
    }
    return { text: `${def.className}(${args.join(', ')})`, missing };
  }

  /** Constructor expression of a layer or module node (no repeat). */
  ctorOf(node: NodeInst, context?: Analysis): { text: string; missing: string[] } {
    if (node.kind === 'module') return this.moduleCtor(node, {}, context?.nodes[node.id]?.children?.[0] ?? this.a0.nodes[node.id]?.children?.[0]);
    const spec = getSpec(node.type)!;
    const resolved = context ? context.nodes[node.id]?.resolved : this.resolvedFor(node.id).resolved;
    return this.layerCtor(spec, node, resolved);
  }

  /** Whether a call of `node` can take an argument written inline (export option "nesting"). */
  private takesInline(node: NodeInst, incoming: Map<string, unknown>, errorsOf: Map<NodeId, string>): boolean {
    if (node.kind !== 'layer' && node.kind !== 'module' && node.kind !== 'op') return false;
    if ((node.repeat && !this.calledOnce(node)) || errorsOf.has(node.id)) return false;
    const ports = portsOf(this.gen.project, node);
    return !!ports && ports.inputs.every((p) => !p.variadic && (p.optional || incoming.has(portKey(node.id, p.name))));
  }

  /** "ResBlock: 73,984 parameters." over every context the class is used in. */
  private docstring(): string {
    const counts = [...new Set(this.analyses.map((a) => a.totalParams))].sort((a, b) => a - b);
    const n = (v: number) => v.toLocaleString('en-US');
    const amount = counts.length === 1 ? `${n(counts[0]!)} parameters` : `${n(counts[0]!)} to ${n(counts.at(-1)!)} parameters, depending on its arguments`;
    return `${this.mod.className}: ${this.analyses.some((a) => a.partial) ? 'at least ' : ''}${amount}.`;
  }

  /** A repeat written as an nn.Sequential and called once (export option), rather than looped over. */
  private calledOnce(node: NodeInst): boolean {
    if (!node.repeat || !this.gen.opts.sequentialRepeats) return false;
    const ports = portsOf(this.gen.project, node);
    return ports?.inputs.length === 1 && ports.outputs.length === 1;
  }

  /** `self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))`, or `self.register_buffer("index", torch.zeros(49, dtype=torch.int64))`. */
  private stateLine(node: NodeInst, spec: NodeSpec): string {
    const params = withDefaults(spec, node.params);
    const resolved = this.resolvedFor(node.id).resolved;
    const value = (name: string) => resolved?.[name] ?? literalOf(params[name]);
    const shapeExpr = exprOf(params.shape);
    // torch.zeros(1, 1, d) rather than torch.zeros((1, 1, d)): a tuple expression is written as its items.
    const shapeText = shapeExpr !== undefined ? this.expr(shapeExpr, 'init') : undefined;
    const shape = shapeText !== undefined ? (tupleItems(shapeText) ?? shapeText) : ((value('shape') as number[] | undefined) ?? []).join(', ');
    const dtype = String(value('dtype') ?? 'float32');
    const tensor = `torch.${String(value('init') ?? 'zeros')}(${shape}${dtype !== 'float32' ? `${shape ? ', ' : ''}dtype=torch.${dtype}` : ''})`;
    if (spec.codegen.state === 'buffer') return `${I2}self.register_buffer(${JSON.stringify(node.name)}, ${tensor}${value('persistent') === false ? ', persistent=False' : ''})`;
    return `${I2}self.${node.name} = nn.Parameter(${tensor}${value('requires_grad') === false ? ', requires_grad=False' : ''})`;
  }

  /** The `self.name = ...` line of a layer or module node. */
  private initLine(node: NodeInst): { line: string; problems: string[] } {
    const stateSpec = node.kind === 'layer' && !node.repeat ? getSpec(node.type) : undefined;
    if (stateSpec?.codegen.state) return { line: this.stateLine(node, stateSpec), problems: [] };
    const problems: string[] = [];
    let text: string;
    if (node.repeat) {
      const { count, first } = node.repeat;
      const countText = 'expr' in count ? this.expr(count.expr, 'init') : String(literalOf(count) ?? 1);
      let rest: { text: string; missing: string[] };
      let head: { text: string; missing: string[] } | undefined;
      if (node.kind === 'module') {
        const children = this.a0.nodes[node.id]?.children;
        rest = this.moduleCtor(node, {}, children?.[1] ?? children?.[0]);
        if (first) head = this.moduleCtor(node, first, children?.[0]);
      } else {
        const spec = getSpec(node.type)!;
        const r = this.resolvedFor(node.id);
        const f = this.resolvedFor(node.id, 'resolvedFirst');
        problems.push(...r.conflicts);
        rest = this.layerCtor(spec, node, r.resolved);
        const firstCtor = this.layerCtor(spec, node, f.resolved, first);
        if (first || firstCtor.text !== rest.text) head = firstCtor;
      }
      problems.push(...rest.missing.map((m) => msg(`无法确定 ${m}`, `cannot determine ${m}`)));
      const copies = head
        ? `[${head.text}] + [${rest.text} for _ in range(${'expr' in count ? subtractOne(countText) : String(Number(countText) - 1)})]`
        : `[${rest.text} for _ in range(${countText})]`;
      // Code kept as written may call the stack (`x = self.blocks(x)`), which needs nn.Sequential.
      if (usedByCode(this.mod).has(node.name) || this.calledOnce(node)) text = head ? `nn.Sequential(*(${copies}))` : `nn.Sequential(*${copies})`;
      else text = `nn.ModuleList(${copies})`;
    } else {
      if (node.kind === 'layer') problems.push(...this.resolvedFor(node.id).conflicts);
      const r = this.ctorOf(node);
      problems.push(...r.missing.map((m) => msg(`无法确定 ${m}`, `cannot determine ${m}`)));
      text = r.text;
    }
    return { line: `${I2}self.${node.name} = ${text}`, problems };
  }

  write(): void {
    const { gen, mod } = this;
    const a0 = this.a0;
    const order = a0.order.map((id) => mod.nodes[id]!).filter((n) => n);
    gen.classStart(mod.id);
    gen.emit(`class ${mod.className}(nn.Module):`);
    if (gen.opts.docstrings) gen.emit(`${I1}"""${this.docstring()}"""`);

    const hints = gen.opts.typeHints !== false;
    const parameter = (hp: HyperParam) => {
      const annotation = hints ? hp.annotation : undefined;
      const head = annotation ? `${hp.name}: ${annotation}` : hp.name;
      if (!hp.default) return head;
      const value = this.value(hp.default, 'init', hp.annotation === 'float');
      return annotation ? `${head} = ${value}` : `${head}=${value}`;
    };
    // Keyword-only hyperparameters follow a bare `*`; an unused **kwargs goes last.
    const keywordOnly = mod.hyperparams.filter((hp) => hp.keywordOnly);
    const signature = [
      'self',
      ...mod.hyperparams.filter((hp) => !hp.keywordOnly).map(parameter),
      ...(keywordOnly.length > 0 ? ['*', ...keywordOnly.map(parameter)] : []),
      ...(mod.kwargs ? [`**${mod.kwargs}`] : []),
    ];
    gen.emit(`${I1}def __init__(${signature.join(', ')}):`);
    gen.emit(`${I2}super().__init__()`);
    for (const name of this.storedHyperparams()) gen.emit(`${I2}self.${name} = ${name}`);
    for (const c of mod.constants ?? []) gen.emit(`${I2}self.${c.name} = ${this.expr(c.expr, 'init')}`);
    // __init__ order must survive a round trip, but the importers place nodes outside
    // the data flow differently. Parameters and buffers (read wherever an expression
    // happens to read them) come first, in the order forward() uses them; submodules
    // without edges (unused, or called only from code kept as written) come last.
    const position = new Map(order.map((n, i) => [n.id, i]));
    const firstUse = (node: NodeInst) =>
      Math.min(...mod.edges.filter((e) => e.source.node === node.id).map((e) => position.get(e.target.node) ?? Infinity));
    const isolated = (node: NodeInst) => !mod.edges.some((e) => e.source.node === node.id || e.target.node === node.id);
    const isState = (node: NodeInst) => node.kind === 'layer' && !node.repeat && !!getSpec(node.type)?.codegen.state;
    const inits = [
      ...order.filter((n) => isState(n) && !isolated(n)).sort((a, b) => firstUse(a) - firstUse(b)),
      ...order.filter((n) => !isState(n) && !isolated(n)),
      ...order.filter(isolated),
    ];
    for (const node of inits) {
      if (node.kind !== 'layer' && node.kind !== 'module') continue;
      const { line, problems } = this.initLine(node);
      gen.emit(problems.length > 0 ? `${line}  # TODO(neuroloom): ${problems.join('; ')}` : line, mod.id, node.id);
      gen.todos += problems.length > 0 ? 1 : 0;
    }
    if (mod.initCode?.trim()) for (const line of indent(mod.initCode.trimEnd(), I2)) gen.emit(line);
    gen.emit('');
    this.writeForward(order);
    if (mod.members?.trim()) {
      gen.emit('');
      for (const line of indent(mod.members.trimEnd(), I1)) gen.emit(line);
    }
  }

  private writeForward(ordered: NodeInst[]): void {
    const { gen, mod } = this;
    const analysis = this.a0;
    const incoming = incomingByPort(mod);

    // Signature and return order are part of the model's interface, so they follow
    // the declared interface and never canvas position.
    const inputs = interfaceOrder(mod, 'input');
    const outputs = interfaceOrder(mod, 'output');
    const returnType =
      outputs.length === 0 ? 'None' : outputs.length === 1 ? 'torch.Tensor' : `tuple[${outputs.map(() => 'torch.Tensor').join(', ')}]`;
    const hints = gen.opts.typeHints !== false;
    // An optional argument keeps its default as written (`targets=None`, `return_h=False`).
    const argument = (n: NodeInst) => {
      const fallback = mod.interface.defaults?.[n.id];
      if (fallback !== undefined) return `${n.name}=${fallback}`;
      return hints ? `${n.name}: torch.Tensor` : n.name;
    };
    const signature = ['self', ...inputs.map(argument)].join(', ');
    gen.emit(hints ? `${I1}def forward(${signature}) -> ${returnType}:` : `${I1}def forward(${signature}):`);
    const bodyStart = gen.lines.length;

    // Variable naming: a value consumed exactly once hands its variable name on
    // (x = self.fc1(x)); a value used several times keeps its own name. A name is
    // never reassigned while another value in it is still needed.
    const varOf = new Map<string, string>();
    const holder = new Map<string, string>();
    const remaining = new Map<string, number>();
    for (const e of mod.edges) {
      const key = portKey(e.source.node, e.source.port);
      remaining.set(key, (remaining.get(key) ?? 0) + 1);
    }
    const taken = new Set([...Object.values(mod.nodes).map((n) => n.name), ...inputs.map((n) => n.name)]);
    const assign = (key: string, name: string) => {
      varOf.set(key, name);
      holder.set(name, key);
      taken.add(name);
    };
    const fresh = (base: string) => {
      for (let i = 1; ; i++) if (!taken.has(`${base}_${i}`)) return `${base}_${i}`;
    };
    /** Moves a still-needed value out of `name` before something else is assigned to it. */
    const ensureFree = (name: string, except?: string) => {
      const held = holder.get(name);
      if (!held || held === except || (remaining.get(held) ?? 0) === 0 || varOf.get(held) !== name) return;
      const moved = fresh(name);
      gen.emit(`${I2}${moved} = ${name}`);
      assign(held, moved);
    };
    const consume = (nodeId: NodeId) => {
      for (const e of mod.edges) {
        if (e.target.node !== nodeId) continue;
        const key = portKey(e.source.node, e.source.port);
        remaining.set(key, (remaining.get(key) ?? 1) - 1);
      }
    };
    // Export option "nesting": a value used once, by the very next statement, is
    // written into that call instead of a line of its own (execution order stays).
    const nesting = gen.opts.nesting ?? 0;
    const inline = new Map<string, { text: string; depth: number; target: string; nodeIds: NodeId[] }>();
    /** Inline values the current node's call takes; their nodes share its lines. */
    let pulled = { keys: [] as string[], nodeIds: [] as NodeId[], depth: -1 };
    const sourceVar = (nodeId: NodeId, port: string): string | undefined => {
      const e = incoming.get(portKey(nodeId, port));
      if (!e) return undefined;
      const key = portKey(e.source.node, e.source.port);
      const held = inline.get(key);
      if (!held) return varOf.get(key);
      if (!pulled.keys.includes(key)) pulled = { keys: [...pulled.keys, key], nodeIds: [...pulled.nodeIds, ...held.nodeIds], depth: Math.max(pulled.depth, held.depth) };
      return held.text;
    };
    const emitting = ordered.filter((n) => n.kind !== 'input' && n.kind !== 'output');
    const errorsOf = new Map<NodeId, string>();
    for (const d of analysis.diagnostics) {
      if (d.severity === 'error' && d.nodeId && !errorsOf.has(d.nodeId)) errorsOf.set(d.nodeId, d.message);
    }
    const probe = (node: NodeInst, port: string, variable: string) => {
      if (gen.options.probe) gen.emit(`${I2}_nl_probe(${JSON.stringify(mod.id)}, ${JSON.stringify(node.id)}, ${JSON.stringify(port)}, ${variable})`);
    };
    const opaqueModule = (node: NodeInst) => node.kind === 'module' && !!gen.project.modules[node.type]?.opaque;
    const calledFromCode = usedByCode(mod);
    let returned = false;

    for (const node of ordered) {
      for (const key of pulled.keys) inline.delete(key);
      pulled = { keys: [], nodeIds: [], depth: -1 };
      if (node.kind === 'input') {
        assign(portKey(node.id, 'out'), node.name);
        continue;
      }
      if (node.kind === 'output') continue;
      // A parameter or buffer is read where it is used: `x + self.pos_embedding`.
      if (node.kind === 'layer' && getSpec(node.type)?.codegen.state) {
        varOf.set(portKey(node.id, 'out'), `self.${node.name}`);
        continue;
      }
      const ports = portsOf(gen.project, node);
      if (!ports) continue;

      if (node.kind === 'code') {
        this.writeCode(node, { incoming, varOf, holder, ensureFree, consume, assign, taken });
        for (const name of node.code?.outputs ?? []) probe(node, name, name);
        continue;
      }

      const variadicExprs = (base: string) =>
        variadicEdges(mod, node.id, base).map((e) => varOf.get(portKey(e.source.node, e.source.port)) ?? 'None');
      const connected = ports.inputs.filter((p) => !p.optional);
      const hasEdges = mod.edges.some((e) => e.target.node === node.id);
      // Called only from code kept as written, or not at all: built in __init__, no call here.
      const unused = !hasEdges && !mod.edges.some((e) => e.source.node === node.id);
      if ((node.kind === 'layer' || node.kind === 'module') && !hasEdges && (calledFromCode.has(node.name) || unused)) continue;
      const missingInput = connected.some((p) => (p.variadic ? variadicExprs(p.name).length === 0 : sourceVar(node.id, p.name) === undefined));
      if (missingInput) {
        gen.emit(`${I2}# TODO(neuroloom): ${msg(`${node.name} 的输入未连接`, `${node.name} has unconnected inputs`)}`, mod.id, node.id);
        gen.todos++;
        continue;
      }

      const info = analysis.nodes[node.id];
      const shape = info?.outputs?.[0]?.shape;
      const problem = errorsOf.get(node.id);
      let comment = shape && gen.opts.shapeComments !== false ? `  # ${formatShape(shape)}` : '';
      if (problem) {
        comment = `  # TODO(neuroloom): ${problem}`;
        gen.todos++;
      }

      // Positional arguments until an optional input is skipped, then keywords.
      const callArgs = () => {
        const args: string[] = [];
        let keywordsOnly = false;
        for (const p of ports.inputs) {
          const v = sourceVar(node.id, p.name);
          if (v === undefined) keywordsOnly = true;
          else args.push(keywordsOnly ? `${p.label}=${v}` : v);
        }
        return args;
      };

      if (node.repeat && (node.kind === 'layer' || node.kind === 'module') && !this.calledOnce(node)) {
        const first = mod.edges.find((e) => e.target.node === node.id && e.target.port === ports.inputs[0]?.name);
        const firstKey = first && portKey(first.source.node, first.source.port);
        const firstVar = firstKey ? varOf.get(firstKey)! : 'None';
        const args = callArgs();
        consume(node.id);
        let target = firstVar;
        if (!firstKey || (remaining.get(firstKey) ?? 0) > 0 || !holder.has(firstVar)) {
          target = node.name;
          ensureFree(target);
          gen.emit(`${I2}${target} = ${firstVar}`, mod.id, node.id);
        }
        let loop = singular(node.name);
        if (taken.has(loop)) loop = fresh(loop);
        taken.add(loop);
        gen.emit(`${I2}for ${loop} in self.${node.name}:`, mod.id, node.id);
        gen.emit(`${I2}${I1}${target} = ${loop}(${[target, ...args.slice(1)].join(', ')})${comment}`, mod.id, node.id);
        assign(portKey(node.id, ports.outputs[0]!.name), target);
        continue;
      }

      let call: string;
      if (node.kind === 'layer' || node.kind === 'module') {
        call = `self.${node.name}(${callArgs().join(', ')})`;
      } else {
        const spec = getSpec(node.type)!;
        const params = withDefaults(spec, node.params);
        const resolved = info?.resolved;
        call = opCall(
          spec,
          (p) => {
            const text = exprOf(params[p.name]);
            return text !== undefined ? { expr: this.expr(text, 'forward') } : { literal: resolved?.[p.name] ?? literalOf(params[p.name]) };
          },
          (port) => sourceVar(node.id, port),
          variadicExprs,
        );
      }
      const tensorInputs = ports.inputs
        .flatMap((p) => (p.variadic ? variadicEdges(mod, node.id, p.name) : [incoming.get(portKey(node.id, p.name))]))
        .filter((e) => e);
      consume(node.id);

      const outs = ports.outputs;
      if (outs.length !== 1) {
        // Tuple results: one name per used output, "_" for the rest.
        const names = outs.map((o) => (consumers(mod, node.id, o.name).length > 0 ? `${node.name}_${o.label}` : '_'));
        const spec = getSpec(node.type);
        for (const n of names) if (n !== '_') ensureFree(n);
        const lhs = spec?.codegen.unpack ? outs.reduce((text, o, i) => text.replace(`{${o.name}}`, names[i]!), spec.codegen.unpack) : names.join(', ');
        gen.emit(names.every((n) => n === '_') ? `${I2}${call}${comment}` : `${I2}${lhs} = ${call}${comment}`, mod.id, node.id, ...pulled.nodeIds);
        outs.forEach((o, i) => {
          if (names[i] === '_') return;
          assign(portKey(node.id, o.name), names[i]!);
          if (opaqueModule(node)) probe(node, o.name, names[i]!);
        });
        continue;
      }

      const outPort = outs[0]!.name;
      const uses = consumers(mod, node.id, outPort);
      if (outputs.length === 1 && uses.length === 1 && uses[0]!.target.node === outputs[0]!.id && !(gen.options.probe && opaqueModule(node))) {
        gen.emit(`${I2}return ${call}${comment}`, mod.id, node.id, ...pulled.nodeIds, outputs[0]!.id);
        returned = true;
        continue;
      }
      let target = node.name;
      const only = tensorInputs.length === 1 ? tensorInputs[0] : undefined;
      if (only && consumers(mod, only.source.node, only.source.port).length === 1) {
        const key = portKey(only.source.node, only.source.port);
        const from = inline.get(key)?.target ?? varOf.get(key)!;
        // `self.layer_scale` is read, never assigned.
        if (!from.startsWith('self.')) target = from;
      }

      const consumer = uses.length === 1 ? mod.nodes[uses[0]!.target.node] : undefined;
      const depth = pulled.depth + 1;
      if (consumer && nesting > depth && !problem && consumer.id === emitting[emitting.indexOf(node) + 1]?.id && this.takesInline(consumer, incoming, errorsOf)) {
        // An operator expression keeps its own parentheses as an operand or a receiver.
        const consumerSpec = consumer.kind === 'op' ? getSpec(consumer.type) : undefined;
        const wrap = node.kind === 'op' && !!getSpec(node.type)?.codegen.operator && (!!consumerSpec?.codegen.operator || !!consumerSpec?.codegen.method);
        inline.set(portKey(node.id, outPort), { text: wrap ? `(${call})` : call, depth, target, nodeIds: [node.id, ...pulled.nodeIds] });
        continue;
      }
      ensureFree(target);
      gen.emit(`${I2}${target} = ${call}${comment}`, mod.id, node.id, ...pulled.nodeIds);
      assign(portKey(node.id, outPort), target);
      if (opaqueModule(node)) probe(node, outPort, target);
    }

    if (!returned && outputs.length > 0) {
      const values = outputs.map((o) => sourceVar(o.id, 'in'));
      if (values.some((v) => v === undefined)) {
        gen.emit(`${I2}return None  # TODO(neuroloom): ${msg('有输出节点未连接', 'an output is not connected')}`, mod.id, ...outputs.map((o) => o.id));
        gen.todos++;
      } else {
        gen.emit(`${I2}return ${values.join(', ')}`, mod.id, ...outputs.map((o) => o.id));
      }
    } else if (gen.lines.length === bodyStart) {
      gen.emit(`${I2}pass`);
    }
  }

  /** A code block: inputs bound to the names it reads, then its statements verbatim. */
  private writeCode(
    node: NodeInst,
    s: {
      incoming: Map<string, { source: { node: NodeId; port: string } }>;
      varOf: Map<string, string>;
      holder: Map<string, string>;
      ensureFree: (name: string, except?: string) => void;
      consume: (nodeId: NodeId) => void;
      assign: (key: string, name: string) => void;
      taken: Set<string>;
    },
  ): void {
    const { gen, mod } = this;
    const code = node.code!;
    for (const name of code.inputs) {
      const e = s.incoming.get(portKey(node.id, name));
      if (!e) continue;
      const key = portKey(e.source.node, e.source.port);
      const v = s.varOf.get(key);
      if (v === undefined || v === name) continue;
      s.ensureFree(name, key);
      gen.emit(`${I2}${name} = ${v}`, mod.id, node.id);
      s.holder.set(name, key);
      s.taken.add(name);
    }
    s.consume(node.id);
    for (const name of new Set([...code.outputs, ...assignedNames(code.source)])) s.ensureFree(name);
    for (const line of indent(code.source.trimEnd(), I2)) gen.emit(line, mod.id, node.id);
    for (const name of code.outputs) s.assign(portKey(node.id, name), name);
  }
}

class Generator {
  readonly lines: string[] = [];
  readonly lineMap: Record<ModuleId, Record<NodeId, number[]>> = {};
  readonly classLines: Record<ModuleId, number> = {};
  /** Per written class: its lines, [start, end) as indexes into `lines`. */
  readonly ranges: Record<ModuleId, [number, number]> = {};
  /** Index of the first line of the self-test, when there is one. */
  selfTestStart?: number;
  todos = 0;
  /** The export options in effect; instrumented code ignores them. */
  readonly opts: ExportOptions;

  constructor(
    readonly project: LoomProject,
    private readonly analysis: ProjectAnalysis,
    readonly options: CodegenOptions,
  ) {
    this.opts = options.probe ? {} : (options.export ?? project.exportOptions ?? {});
  }

  emit(line: string, moduleId?: ModuleId, ...nodeIds: NodeId[]): void {
    this.lines.push(line);
    if (!moduleId) return;
    const map = (this.lineMap[moduleId] ??= {});
    for (const id of nodeIds) (map[id] ??= []).push(this.lines.length);
  }

  classStart(moduleId: ModuleId): void {
    this.classLines[moduleId] = this.lines.length + 1;
  }

  analysesOf(moduleId: ModuleId): Analysis[] {
    const list = this.analysis.byModule[moduleId];
    return list && list.length > 0 ? list : [analyzeModule(this.project, moduleId)];
  }

  run(): void {
    const { project } = this;
    const modules = dependencyOrder(project).map((id) => project.modules[id]!);
    const verbatim = [project.preamble, ...modules.flatMap((m) => [m.opaque?.source, m.initCode, m.members, ...Object.values(m.nodes).map((n) => n.code?.source)])].join('\n');
    const usesF =
      /\bF\./.test(verbatim) ||
      modules.some((m) => Object.values(m.nodes).some((n) => n.kind === 'op' && getSpec(n.type)?.codegen.fn?.startsWith('F.')));

    if (this.opts.header) {
      this.emit(`# ${project.name || project.modules[project.entry]!.className}: generated by NeuroLoom from a graph.`);
      this.emit('');
    }
    // isort order: the standard library first; then plain imports, then from-imports.
    if (/\bOrderedDict\b/.test(verbatim) || modules.some((m) => m.sequential && !numberedChain(m))) {
      this.emit('from collections import OrderedDict');
      this.emit('');
    }
    this.emit('import torch');
    if (usesF) this.emit('import torch.nn.functional as F');
    this.emit('from torch import nn');
    if (project.preamble?.trim()) {
      this.emit('');
      for (const line of project.preamble.trimEnd().split('\n')) this.emit(line);
    }
    if (this.options.probe) {
      this.emit('');
      this.emit('');
      for (const line of PROBE_PRELUDE.split('\n')) this.emit(line);
    }
    for (const mod of modules) {
      if (mod.sequential || (mod.opaque && !mod.opaque.source)) continue;
      this.emit('');
      this.emit('');
      const start = this.lines.length;
      if (mod.opaque) {
        this.classStart(mod.id);
        for (const line of mod.opaque.source!.trimEnd().split('\n')) this.emit(line);
      } else {
        new ClassWriter(this, mod, this.analysesOf(mod.id)).write();
      }
      this.ranges[mod.id] = [start, this.lines.length];
    }
    if (this.opts.selfTest) this.writeSelfTest();
  }

  /**
   * `if __name__ == "__main__":` that builds the entry model, runs it once on
   * inputs of the declared shapes (B = 1, other symbols 128) and checks every
   * output shape inference knows.
   */
  private writeSelfTest(): void {
    const { project } = this;
    const entry = project.modules[project.entry]!;
    const analysis = this.analysis.entry;
    const inputs = interfaceOrder(entry, 'input');
    const outputs = interfaceOrder(entry, 'output');
    const values = (shape: TensorType['shape']) => Object.fromEntries(shape.filter((d): d is string => typeof d === 'string').flatMap((d) => d.split(/[+*]/)).map((s) => [s.trim(), defaultSymbolValue(s.trim())]));
    const sizesOf = (t: TensorType | undefined) => (t ? shapeValue(t.shape, values(t.shape)) : undefined);
    const taken = new Set(inputs.map((n) => n.name));
    const model = taken.has('model') ? 'model_' : 'model';
    const results = outputs.length === 1 ? ['out'] : outputs.map((o) => (taken.has(o.name) || o.name === model ? `${o.name}_` : o.name));

    this.selfTestStart = this.lines.length;
    this.emit('');
    this.emit('');
    this.emit('if __name__ == "__main__":');
    this.emit(`${I1}${model} = ${entry.className}()`);
    // An optional argument without an example shape is left at its default; later ones go by name.
    const passed = inputs.filter((n) => entry.interface.defaults?.[n.id] === undefined || analysis.nodes[n.id]?.outputs?.[0]);
    const args: string[] = [];
    for (const n of inputs) {
      if (!passed.includes(n)) continue;
      args.push(args.length < inputs.indexOf(n) ? `${n.name}=${n.name}` : n.name);
    }
    for (const node of passed) {
      const t = analysis.nodes[node.id]?.outputs?.[0];
      const sizes = sizesOf(t);
      const dtype = t?.dtype ?? 'float32';
      const shape = (sizes ?? [1]).join(', ');
      const todo = sizes ? '' : `  # TODO(neuroloom): ${msg(`声明 ${node.name} 的形状`, `declare the shape of ${node.name}`)}`;
      if (!sizes) this.todos++;
      if (dtype.startsWith('float') || dtype === 'bfloat16') this.emit(`${I1}${node.name} = torch.randn(${shape}${dtype === 'float32' ? '' : `, dtype=torch.${dtype}`})${todo}`);
      else this.emit(`${I1}${node.name} = torch.zeros(${shape}, dtype=torch.${dtype})${todo}`);
    }
    if (outputs.length === 0) {
      this.emit(`${I1}${model}(${args.join(', ')})`);
      return;
    }
    this.emit(`${I1}${results.join(', ')} = ${model}(${args.join(', ')})`);
    let checked = false;
    outputs.forEach((o, i) => {
      const sizes = sizesOf(analysis.outputs[i]);
      if (!sizes) return;
      checked = true;
      this.emit(`${I1}assert tuple(${results[i]}.shape) == ${sizes.length === 1 ? `(${sizes[0]},)` : `(${sizes.join(', ')})`}, ${results[i]}.shape`);
    });
    if (!checked) this.emit(`${I1}print(${results.map((r) => `tuple(${r}.shape)`).join(', ')})`);
  }
}

export function generateCode(project: LoomProject, analysis: ProjectAnalysis = analyzeProject(project), options: CodegenOptions = {}): CodegenResult {
  const gen = new Generator(project, analysis, options);
  gen.run();
  return { code: `${gen.lines.join('\n')}\n`, lineMap: gen.lineMap, classLines: gen.classLines, todos: gen.todos };
}

export interface GeneratedFile {
  name: string;
  text: string;
}

/**
 * The code as a Python package (export option "one file per class"): a module
 * per class, named after it in snake_case, importing the classes it mentions
 * from its siblings, and an __init__.py that exports the entry model. The
 * self-test, if any, goes with the entry class (run it with `python -m`).
 */
export function generateFiles(project: LoomProject, analysis: ProjectAnalysis = analyzeProject(project), options: CodegenOptions = {}): GeneratedFile[] {
  const gen = new Generator(project, analysis, options);
  gen.run();
  const ids = Object.keys(gen.ranges);
  const className = (id: ModuleId) => project.modules[id]!.className;
  const fileOf = new Map<ModuleId, string>();
  const taken = new Set(['__init__']);
  for (const id of ids) {
    const base = snakeCase(className(id)) || 'model';
    let name = base;
    for (let i = 2; taken.has(name); i++) name = `${base}_${i}`;
    taken.add(name);
    fileOf.set(id, name);
  }
  const header = gen.opts.header ? [gen.lines[0]!, ''] : [];
  const preamble = project.preamble?.trim() ? ['', ...project.preamble.trimEnd().split('\n')] : [];
  const files = ids.map((id): GeneratedFile => {
    const [start, end] = gen.ranges[id]!;
    const body = gen.lines.slice(start, end);
    const text = body.join('\n');
    const siblings = ids
      .filter((other) => other !== id && new RegExp(`\\b${className(other)}\\b`).test(text))
      .map((other) => `from .${fileOf.get(other)} import ${className(other)}`)
      .sort();
    const imports = [
      ...(/\bOrderedDict\b/.test(text) ? ['from collections import OrderedDict', ''] : []),
      'import torch',
      ...(/\bF\./.test(text) ? ['import torch.nn.functional as F'] : []),
      'from torch import nn',
      ...preamble,
    ];
    const selfTest = id === project.entry && gen.selfTestStart !== undefined ? gen.lines.slice(gen.selfTestStart) : [];
    const lines = [...header, ...imports, ...(siblings.length > 0 ? ['', ...siblings] : []), '', '', ...body, ...selfTest];
    return { name: `${fileOf.get(id)}.py`, text: `${lines.join('\n')}\n` };
  });
  const entry = className(project.entry);
  const init = [...header, `from .${fileOf.get(project.entry)} import ${entry}`, '', `__all__ = ["${entry}"]`];
  return [{ name: '__init__.py', text: `${init.join('\n')}\n` }, ...files];
}
