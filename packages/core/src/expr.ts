// Python expressions in parameters (PRD F2.3), hyperparameter defaults and
// repeat counts: "d_model * 4", "in_ch if first else out_ch", "max(1, c // 16)".
// Expressions are stored as written; this module parses a safe subset of Python
// to evaluate them, list the names they use and rewrite those names.
import { msg } from './i18n';
import type { Literal } from './ir';

export type Expr =
  | { t: 'num'; value: number; raw: string }
  | { t: 'str'; value: string; raw: string }
  | { t: 'const'; value: boolean | null }
  | { t: 'name'; id: string; start: number; end: number }
  | { t: 'unary'; op: '-' | '+' | '~' | 'not'; operand: Expr }
  | { t: 'bin'; op: string; left: Expr; right: Expr }
  | { t: 'bool'; op: 'and' | 'or'; values: Expr[] }
  | { t: 'cmp'; ops: string[]; operands: Expr[] }
  | { t: 'ifexp'; test: Expr; body: Expr; orelse: Expr }
  | { t: 'call'; func: Expr; args: Expr[]; kwargs: [string, Expr][] }
  | { t: 'attr'; value: Expr; attr: string }
  | { t: 'sub'; value: Expr; index: Expr }
  | { t: 'tuple'; items: Expr[] }
  | { t: 'list'; items: Expr[] };

// ---------------------------------------------------------------- tokens

type Token =
  | { k: 'num'; raw: string; value: number; pos: number }
  | { k: 'str'; raw: string; value: string; pos: number }
  | { k: 'name'; raw: string; pos: number }
  | { k: 'op'; raw: string; pos: number }
  | { k: 'end'; raw: ''; pos: number };

const KEYWORDS = new Set(['and', 'or', 'not', 'if', 'else', 'in', 'is', 'True', 'False', 'None', 'lambda']);
const OPERATORS = ['**', '//', '==', '!=', '<=', '>=', '<<', '>>', '+', '-', '*', '/', '%', '@', '<', '>', '(', ')', '[', ']', ',', '.', ':', '=', '~', '&', '|', '^'];

class ExprError extends Error {}

function tokenize(text: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < text.length) {
    const c = text[i]!;
    if (/\s/.test(c)) {
      i++;
      continue;
    }
    const rest = text.slice(i);
    const number = /^(0[xX][0-9a-fA-F_]+|0[oO][0-7_]+|0[bB][01_]+|(\d[\d_]*\.?[\d_]*|\.\d[\d_]*)([eE][+-]?\d[\d_]*)?)/.exec(rest);
    if (number && (/\d/.test(c) || (c === '.' && /\d/.test(text[i + 1] ?? '')))) {
      const raw = number[0];
      if (/^[jJ]/.test(text[i + raw.length] ?? '')) throw new ExprError(msg('不支持复数', 'complex numbers are not supported'));
      const clean = raw.replace(/_/g, '');
      const value = /^0[xXoObB]/.test(clean) ? Number(clean) : Number.parseFloat(clean);
      tokens.push({ k: 'num', raw, value, pos: i });
      i += raw.length;
      continue;
    }
    if (c === '"' || c === "'") {
      let j = i + 1;
      let value = '';
      while (j < text.length && text[j] !== c) {
        if (text[j] === '\\') {
          const e = text[j + 1] ?? '';
          value += e === 'n' ? '\n' : e === 't' ? '\t' : e;
          j += 2;
        } else value += text[j++];
      }
      if (j >= text.length) throw new ExprError(msg('字符串没有结束', 'unterminated string'));
      tokens.push({ k: 'str', raw: text.slice(i, j + 1), value, pos: i });
      i = j + 1;
      continue;
    }
    const name = /^[A-Za-z_][A-Za-z0-9_]*/.exec(rest);
    if (name) {
      tokens.push({ k: 'name', raw: name[0], pos: i });
      i += name[0].length;
      continue;
    }
    const op = OPERATORS.find((o) => rest.startsWith(o));
    if (!op) throw new ExprError(msg(`无法识别的字符“${c}”`, `unexpected character "${c}"`));
    tokens.push({ k: 'op', raw: op, pos: i });
    i += op.length;
  }
  tokens.push({ k: 'end', raw: '', pos: text.length });
  return tokens;
}

// ---------------------------------------------------------------- parser

class Parser {
  private i = 0;
  constructor(private readonly tokens: Token[]) {}

  private peek(): Token {
    return this.tokens[this.i]!;
  }
  private next(): Token {
    return this.tokens[this.i++]!;
  }
  private is(raw: string): boolean {
    const t = this.peek();
    return (t.k === 'op' || t.k === 'name') && t.raw === raw;
  }
  private eat(raw: string): boolean {
    if (!this.is(raw)) return false;
    this.i++;
    return true;
  }
  private expect(raw: string): void {
    if (!this.eat(raw)) throw new ExprError(msg(`此处应为“${raw}”`, `expected "${raw}"`));
  }

  parseAll(): Expr {
    const e = this.tupleOrTest();
    if (this.peek().k !== 'end') throw new ExprError(msg(`多余的内容“${this.peek().raw}”`, `unexpected "${this.peek().raw}"`));
    return e;
  }

  /** A bare tuple is allowed at the top level: "3, 3". */
  private tupleOrTest(): Expr {
    const first = this.test();
    if (!this.is(',')) return first;
    const items = [first];
    while (this.eat(',')) {
      if (this.peek().k === 'end' || this.is(')')) break;
      items.push(this.test());
    }
    return { t: 'tuple', items };
  }

  private test(): Expr {
    if (this.is('lambda')) throw new ExprError(msg('不支持 lambda', 'lambda is not supported'));
    const body = this.or();
    if (!this.eat('if')) return body;
    const test = this.or();
    this.expect('else');
    return { t: 'ifexp', test, body, orelse: this.test() };
  }

  private or(): Expr {
    const values = [this.and()];
    while (this.eat('or')) values.push(this.and());
    return values.length === 1 ? values[0]! : { t: 'bool', op: 'or', values };
  }

  private and(): Expr {
    const values = [this.not()];
    while (this.eat('and')) values.push(this.not());
    return values.length === 1 ? values[0]! : { t: 'bool', op: 'and', values };
  }

  private not(): Expr {
    if (this.eat('not')) return { t: 'unary', op: 'not', operand: this.not() };
    return this.comparison();
  }

  private comparison(): Expr {
    const operands = [this.bitOr()];
    const ops: string[] = [];
    for (;;) {
      let op: string | undefined;
      if (['<', '>', '==', '>=', '<=', '!='].some((o) => this.is(o))) op = this.next().raw;
      else if (this.eat('in')) op = 'in';
      else if (this.is('not') && this.tokens[this.i + 1]?.raw === 'in') {
        this.i += 2;
        op = 'not in';
      } else if (this.eat('is')) op = this.eat('not') ? 'is not' : 'is';
      if (!op) break;
      ops.push(op);
      operands.push(this.bitOr());
    }
    return ops.length === 0 ? operands[0]! : { t: 'cmp', ops, operands };
  }

  private binary(next: () => Expr, ops: string[]): Expr {
    let left = next();
    for (;;) {
      const t = this.peek();
      if (t.k !== 'op' || !ops.includes(t.raw)) return left;
      this.i++;
      left = { t: 'bin', op: t.raw, left, right: next() };
    }
  }

  private bitOr = (): Expr => this.binary(this.bitXor, ['|']);
  private bitXor = (): Expr => this.binary(this.bitAnd, ['^']);
  private bitAnd = (): Expr => this.binary(this.shift, ['&']);
  private shift = (): Expr => this.binary(this.arith, ['<<', '>>']);
  private arith = (): Expr => this.binary(this.term, ['+', '-']);
  private term = (): Expr => this.binary(this.factor, ['*', '/', '//', '%', '@']);

  private factor = (): Expr => {
    for (const op of ['-', '+', '~'] as const) {
      if (this.eat(op)) return { t: 'unary', op, operand: this.factor() };
    }
    return this.power();
  };

  private power(): Expr {
    const base = this.trailers(this.atom());
    // ** binds tighter than a unary minus on its left but takes one on its right: -2 ** -1.
    if (this.eat('**')) return { t: 'bin', op: '**', left: base, right: this.factor() };
    return base;
  }

  private trailers(value: Expr): Expr {
    for (;;) {
      if (this.eat('(')) {
        const args: Expr[] = [];
        const kwargs: [string, Expr][] = [];
        while (!this.is(')')) {
          const t = this.peek();
          if (t.k === 'name' && this.tokens[this.i + 1]?.raw === '=') {
            this.i += 2;
            kwargs.push([t.raw, this.test()]);
          } else {
            if (kwargs.length > 0) throw new ExprError(msg('位置参数不能在关键字参数之后', 'positional argument after keyword argument'));
            args.push(this.test());
          }
          if (!this.eat(',')) break;
        }
        this.expect(')');
        value = { t: 'call', func: value, args, kwargs };
      } else if (this.eat('.')) {
        const t = this.next();
        if (t.k !== 'name') throw new ExprError(msg('“.”后应为属性名', 'expected an attribute name after "."'));
        value = { t: 'attr', value, attr: t.raw };
      } else if (this.eat('[')) {
        const index = this.tupleOrTest();
        this.expect(']');
        value = { t: 'sub', value, index };
      } else return value;
    }
  }

  private atom(): Expr {
    const t = this.next();
    if (t.k === 'num') return { t: 'num', value: t.value, raw: t.raw };
    if (t.k === 'str') {
      let value = t.value;
      let raw = t.raw;
      // Adjacent string literals concatenate.
      while (this.peek().k === 'str') {
        const s = this.next() as Extract<Token, { k: 'str' }>;
        value += s.value;
        raw += ` ${s.raw}`;
      }
      return { t: 'str', value, raw };
    }
    if (t.k === 'name') {
      if (t.raw === 'True' || t.raw === 'False') return { t: 'const', value: t.raw === 'True' };
      if (t.raw === 'None') return { t: 'const', value: null };
      if (KEYWORDS.has(t.raw)) throw new ExprError(msg(`此处不能出现“${t.raw}”`, `unexpected "${t.raw}"`));
      return { t: 'name', id: t.raw, start: t.pos, end: t.pos + t.raw.length };
    }
    if (t.k === 'op' && (t.raw === '(' || t.raw === '[')) {
      const close = t.raw === '(' ? ')' : ']';
      const items: Expr[] = [];
      let trailingComma = false;
      while (!this.is(close)) {
        items.push(this.test());
        trailingComma = this.eat(',');
        if (!trailingComma) break;
      }
      this.expect(close);
      if (t.raw === '[') return { t: 'list', items };
      return items.length === 1 && !trailingComma ? items[0]! : { t: 'tuple', items };
    }
    throw new ExprError(t.k === 'end' ? msg('表达式不完整', 'incomplete expression') : msg(`此处不能出现“${t.raw}”`, `unexpected "${t.raw}"`));
  }
}

export type ParseResult = { ok: true; expr: Expr } | { ok: false; error: string };

export function parseExpr(text: string): ParseResult {
  try {
    if (text.trim() === '') return { ok: false, error: msg('表达式为空', 'empty expression') };
    return { ok: true, expr: new Parser(tokenize(text)).parseAll() };
  } catch (e) {
    if (e instanceof ExprError) return { ok: false, error: e.message };
    throw e;
  }
}

// ---------------------------------------------------------------- printing

// Precedence levels as in Python's ast.unparse.
const P = { TUPLE: 1, TEST: 2, OR: 3, AND: 4, NOT: 5, CMP: 6, BOR: 7, BXOR: 8, BAND: 9, SHIFT: 10, ARITH: 11, TERM: 12, FACTOR: 13, POWER: 14, ATOM: 16 };
const BIN_PREC: Record<string, number> = {
  '|': P.BOR,
  '^': P.BXOR,
  '&': P.BAND,
  '<<': P.SHIFT,
  '>>': P.SHIFT,
  '+': P.ARITH,
  '-': P.ARITH,
  '*': P.TERM,
  '/': P.TERM,
  '//': P.TERM,
  '%': P.TERM,
  '@': P.TERM,
  '**': P.POWER,
};

function precedenceOf(e: Expr): number {
  switch (e.t) {
    case 'ifexp':
      return P.TEST;
    case 'bool':
      return e.op === 'or' ? P.OR : P.AND;
    case 'unary':
      return e.op === 'not' ? P.NOT : P.FACTOR;
    case 'cmp':
      return P.CMP;
    case 'bin':
      return BIN_PREC[e.op]!;
    default:
      return P.ATOM;
  }
}

/** Python source for an expression, with the fewest parentheses that keep its meaning. */
export function printExpr(e: Expr, context = P.TEST): string {
  const wrap = (text: string) => (precedenceOf(e) < context ? `(${text})` : text);
  switch (e.t) {
    case 'num':
    case 'str':
      return e.raw;
    case 'const':
      return e.value === null ? 'None' : e.value ? 'True' : 'False';
    case 'name':
      return e.id;
    case 'unary':
      return wrap(e.op === 'not' ? `not ${printExpr(e.operand, P.NOT)}` : `${e.op}${printExpr(e.operand, P.FACTOR)}`);
    case 'bin': {
      const p = BIN_PREC[e.op]!;
      const right = e.op === '**' ? printExpr(e.right, P.FACTOR) : printExpr(e.right, p + 1);
      const left = e.op === '**' ? printExpr(e.left, p + 1) : printExpr(e.left, p);
      return wrap(`${left} ${e.op} ${right}`);
    }
    case 'bool': {
      const p = e.op === 'or' ? P.OR : P.AND;
      return wrap(e.values.map((v) => printExpr(v, p + 1)).join(` ${e.op} `));
    }
    case 'cmp':
      return wrap(e.operands.map((o, i) => (i === 0 ? '' : ` ${e.ops[i - 1]} `) + printExpr(o, P.CMP + 1)).join(''));
    case 'ifexp':
      return wrap(`${printExpr(e.body, P.TEST + 1)} if ${printExpr(e.test, P.TEST + 1)} else ${printExpr(e.orelse, P.TEST)}`);
    case 'call':
      return `${printExpr(e.func, P.ATOM)}(${[...e.args.map((a) => printExpr(a)), ...e.kwargs.map(([k, v]) => `${k}=${printExpr(v)}`)].join(', ')})`;
    case 'attr':
      return `${printExpr(e.value, P.ATOM)}.${e.attr}`;
    case 'sub':
      return `${printExpr(e.value, P.ATOM)}[${e.index.t === 'tuple' && e.index.items.length > 0 ? e.index.items.map((i) => printExpr(i)).join(', ') : printExpr(e.index)}]`;
    case 'tuple':
      return e.items.length === 1 ? `(${printExpr(e.items[0]!)},)` : `(${e.items.map((i) => printExpr(i)).join(', ')})`;
    case 'list':
      return `[${e.items.map((i) => printExpr(i)).join(', ')}]`;
  }
}

/**
 * The expression as the right operand of a binary operator: unchanged, or in
 * parentheses when it binds more loosely (`x * (a - b)`).
 */
export function asOperand(text: string, operator: string): string {
  const parsed = parseExpr(text);
  const p = BIN_PREC[operator];
  if (!parsed.ok || p === undefined) return text;
  return precedenceOf(parsed.expr) < p + 1 ? `(${text})` : text;
}

/** `text - 1`, parenthesized only when needed. */
export function subtractOne(text: string): string {
  const parsed = parseExpr(text);
  if (!parsed.ok) return `(${text}) - 1`;
  return printExpr({ t: 'bin', op: '-', left: parsed.expr, right: { t: 'num', value: 1, raw: '1' } });
}

// ---------------------------------------------------------------- names

const BUILTINS = new Set(['int', 'float', 'abs', 'min', 'max', 'round', 'len', 'bool', 'sum', 'tuple', 'list', 'math']);

function visitNames(e: Expr, visit: (n: Extract<Expr, { t: 'name' }>) => void): void {
  switch (e.t) {
    case 'name':
      return visit(e);
    case 'unary':
      return visitNames(e.operand, visit);
    case 'bin':
      visitNames(e.left, visit);
      return visitNames(e.right, visit);
    case 'bool':
    case 'cmp':
      for (const v of e.t === 'bool' ? e.values : e.operands) visitNames(v, visit);
      return;
    case 'ifexp':
      visitNames(e.test, visit);
      visitNames(e.body, visit);
      return visitNames(e.orelse, visit);
    case 'call':
      visitNames(e.func, visit);
      for (const a of e.args) visitNames(a, visit);
      for (const [, v] of e.kwargs) visitNames(v, visit);
      return;
    case 'attr':
      return visitNames(e.value, visit);
    case 'sub':
      visitNames(e.value, visit);
      return visitNames(e.index, visit);
    case 'tuple':
    case 'list':
      for (const i of e.items) visitNames(i, visit);
      return;
    default:
      return;
  }
}

/** Names an expression refers to, excluding the builtins it may call; empty when it does not parse. */
export function freeNames(text: string): string[] {
  const parsed = parseExpr(text);
  if (!parsed.ok) return [];
  const names = new Set<string>();
  visitNames(parsed.expr, (n) => {
    if (!BUILTINS.has(n.id)) names.add(n.id);
  });
  return [...names];
}

/**
 * Replaces names in the text, keeping everything else exactly as written:
 * `d_model * 4` with d_model -> self.d_model gives `self.d_model * 4`.
 */
export function renameInExpr(text: string, rename: (name: string) => string | undefined): string {
  const parsed = parseExpr(text);
  if (!parsed.ok) return text;
  const spans: { start: number; end: number; text: string }[] = [];
  visitNames(parsed.expr, (n) => {
    const replacement = BUILTINS.has(n.id) ? undefined : rename(n.id);
    if (replacement !== undefined && replacement !== n.id) spans.push({ start: n.start, end: n.end, text: replacement });
  });
  let out = text;
  for (const s of spans.sort((a, b) => b.start - a.start)) out = out.slice(0, s.start) + s.text + out.slice(s.end);
  return out;
}

/**
 * Replaces names with other expressions, re-printing with the parentheses the
 * result needs: `in_ch * 2` with in_ch -> "a + b" gives `(a + b) * 2`.
 */
export function substituteExpr(text: string, values: Record<string, string>): string {
  const parsed = parseExpr(text);
  if (!parsed.ok) return text;
  const replace = (e: Expr): Expr => {
    switch (e.t) {
      case 'name': {
        const v = values[e.id];
        if (v === undefined) return e;
        const sub = parseExpr(v);
        return sub.ok ? sub.expr : e;
      }
      case 'unary':
        return { ...e, operand: replace(e.operand) };
      case 'bin':
        return { ...e, left: replace(e.left), right: replace(e.right) };
      case 'bool':
        return { ...e, values: e.values.map(replace) };
      case 'cmp':
        return { ...e, operands: e.operands.map(replace) };
      case 'ifexp':
        return { ...e, test: replace(e.test), body: replace(e.body), orelse: replace(e.orelse) };
      case 'call':
        return { ...e, func: e.func, args: e.args.map(replace), kwargs: e.kwargs.map(([k, v]) => [k, replace(v)] as [string, Expr]) };
      case 'attr':
        return { ...e, value: replace(e.value) };
      case 'sub':
        return { ...e, value: replace(e.value), index: replace(e.index) };
      case 'tuple':
      case 'list':
        return { ...e, items: e.items.map(replace) };
      default:
        return e;
    }
  };
  return printExpr(replace(parsed.expr));
}

/**
 * The value when the text is a plain literal ("64", "(3, 3)", "'gelu'", "None"),
 * otherwise undefined: then it is an expression.
 */
export function asLiteral(text: string): Literal | undefined {
  const parsed = parseExpr(text);
  if (!parsed.ok) return undefined;
  const plain = (e: Expr): boolean =>
    e.t === 'num' ||
    e.t === 'str' ||
    e.t === 'const' ||
    (e.t === 'unary' && (e.op === '-' || e.op === '+') && e.operand.t === 'num') ||
    ((e.t === 'tuple' || e.t === 'list') && e.items.every(plain));
  if (!plain(parsed.expr)) return undefined;
  const r = evalExpr(text, {});
  return r.ok ? r.value : undefined;
}

/** Python source for a literal value. */
export function literalSource(value: Literal): string {
  if (value === null) return 'None';
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value === 'number') return String(value);
  if (typeof value === 'string') return JSON.stringify(value);
  const items = value.map(literalSource);
  return items.length === 1 ? `(${items[0]},)` : `(${items.join(', ')})`;
}

// ---------------------------------------------------------------- evaluation

/** Values of names in scope; `undefined` marks a name whose value is not known. */
export type Env = Record<string, Literal | undefined>;

export type EvalResult = { ok: true; value: Literal } | { ok: false; error: string; unknown?: boolean };

class Unknown extends Error {}

const truthy = (v: Literal): boolean => (Array.isArray(v) ? v.length > 0 : v !== null && v !== false && v !== 0 && v !== '');
const isNum = (v: Literal): v is number => typeof v === 'number';
const asNum = (v: Literal): number => {
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (typeof v !== 'number') throw new ExprError(msg(`需要数字，得到 ${literalSource(v)}`, `expected a number, got ${literalSource(v)}`));
  return v;
};
const equal = (a: Literal, b: Literal): boolean => JSON.stringify(a) === JSON.stringify(b);

/** Python's round(): half to even. */
function pyRound(x: number, digits = 0): number {
  const m = 10 ** digits;
  const v = x * m;
  const r = Math.round(v);
  return (Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r) / m;
}

const MATH: Record<string, Literal | ((...a: number[]) => number)> = {
  pi: Math.PI,
  e: Math.E,
  inf: Number.POSITIVE_INFINITY,
  sqrt: Math.sqrt,
  ceil: Math.ceil,
  floor: Math.floor,
  log: (x, base) => (base === undefined ? Math.log(x) : Math.log(x) / Math.log(base)),
  log2: Math.log2,
  log10: Math.log10,
  exp: Math.exp,
  gcd: (a, b) => {
    let x = Math.abs(a);
    let y = Math.abs(b);
    while (y) [x, y] = [y, x % y];
    return x;
  },
};

function evaluate(e: Expr, env: Env): Literal {
  switch (e.t) {
    case 'num':
    case 'str':
    case 'const':
      return e.value;
    case 'name': {
      if (!(e.id in env)) throw new ExprError(msg(`未定义的名称 ${e.id}`, `name ${e.id} is not defined`));
      const v = env[e.id];
      if (v === undefined) throw new Unknown();
      return v;
    }
    case 'unary': {
      const v = evaluate(e.operand, env);
      if (e.op === 'not') return !truthy(v);
      if (e.op === '~') return ~asNum(v);
      return e.op === '-' ? -asNum(v) : asNum(v);
    }
    case 'bool': {
      let v: Literal = null;
      for (const item of e.values) {
        v = evaluate(item, env);
        if (e.op === 'and' ? !truthy(v) : truthy(v)) return v;
      }
      return v;
    }
    case 'cmp': {
      let left = evaluate(e.operands[0]!, env);
      for (let i = 0; i < e.ops.length; i++) {
        const right = evaluate(e.operands[i + 1]!, env);
        if (!compare(e.ops[i]!, left, right)) return false;
        left = right;
      }
      return true;
    }
    case 'ifexp':
      return truthy(evaluate(e.test, env)) ? evaluate(e.body, env) : evaluate(e.orelse, env);
    case 'bin':
      return binary(e.op, evaluate(e.left, env), evaluate(e.right, env));
    case 'tuple':
    case 'list':
      return e.items.map((i) => evaluate(i, env));
    case 'sub': {
      const v = evaluate(e.value, env);
      const i = evaluate(e.index, env);
      if (!Array.isArray(v) || !isNum(i) || !Number.isInteger(i)) throw new ExprError(msg('只支持对元组按整数下标取值', 'only integer indexing of tuples is supported'));
      const item = v[i < 0 ? v.length + i : i];
      if (item === undefined) throw new ExprError(msg('下标越界', 'index out of range'));
      return item;
    }
    case 'attr': {
      if (e.value.t === 'name' && e.value.id === 'math' && !('math' in env)) {
        const v = MATH[e.attr];
        if (v === undefined || typeof v === 'function') throw new ExprError(msg(`不支持 math.${e.attr}`, `math.${e.attr} is not supported`));
        return v;
      }
      throw new ExprError(msg('表达式中不支持属性访问', 'attribute access is not supported in expressions'));
    }
    case 'call':
      return call(e, env);
  }
}

function compare(op: string, a: Literal, b: Literal): boolean {
  switch (op) {
    case '==':
      return equal(a, b);
    case '!=':
      return !equal(a, b);
    case 'is':
      return a === b;
    case 'is not':
      return a !== b;
    case 'in':
    case 'not in': {
      const found = Array.isArray(b) ? b.some((x) => equal(x, a)) : typeof b === 'string' && typeof a === 'string' && b.includes(a);
      return op === 'in' ? found : !found;
    }
  }
  const x = typeof a === 'string' && typeof b === 'string' ? a : asNum(a);
  const y = typeof a === 'string' && typeof b === 'string' ? b : asNum(b);
  return op === '<' ? x < y : op === '>' ? x > y : op === '<=' ? x <= y : x >= y;
}

function binary(op: string, a: Literal, b: Literal): Literal {
  if (op === '+' && typeof a === 'string' && typeof b === 'string') return a + b;
  if (op === '+' && Array.isArray(a) && Array.isArray(b)) return [...a, ...b];
  if (op === '*' && Array.isArray(a) && isNum(b)) return Array.from({ length: Math.max(0, b) }, () => a).flat();
  const x = asNum(a);
  const y = asNum(b);
  const nonzero = () => {
    if (y === 0) throw new ExprError(msg('除数为零', 'division by zero'));
  };
  switch (op) {
    case '+':
      return x + y;
    case '-':
      return x - y;
    case '*':
      return x * y;
    case '/':
      nonzero();
      return x / y;
    case '//':
      nonzero();
      return Math.floor(x / y);
    case '%':
      nonzero();
      return ((x % y) + y) % y;
    case '**':
      return x ** y;
    case '<<':
      return x << y;
    case '>>':
      return x >> y;
    case '&':
      return x & y;
    case '|':
      return x | y;
    case '^':
      return x ^ y;
  }
  throw new ExprError(msg(`不支持运算符 ${op}`, `operator ${op} is not supported`));
}

function call(e: Extract<Expr, { t: 'call' }>, env: Env): Literal {
  const args = e.args.map((a) => evaluate(a, env));
  if (e.kwargs.length > 0) throw new ExprError(msg('函数调用不支持关键字参数', 'keyword arguments are not supported in calls'));
  if (e.func.t === 'attr' && e.func.value.t === 'name' && e.func.value.id === 'math' && !('math' in env)) {
    const f = MATH[e.func.attr];
    if (typeof f !== 'function') throw new ExprError(msg(`不支持 math.${e.func.attr}`, `math.${e.func.attr} is not supported`));
    return f(...args.map(asNum));
  }
  if (e.func.t !== 'name' || e.func.id in env) throw new ExprError(msg('表达式中只能调用内置函数', 'only builtin functions can be called in expressions'));
  const items = (v: Literal | undefined): Literal[] => {
    if (!Array.isArray(v)) throw new ExprError(msg('需要元组或列表', 'expected a tuple or list'));
    return v;
  };
  const values = args.length === 1 && Array.isArray(args[0]) ? args[0] : args;
  switch (e.func.id) {
    case 'int': {
      const v = args[0];
      if (typeof v === 'string') return Number.parseInt(v, 10);
      return Math.trunc(asNum(v ?? 0));
    }
    case 'float':
      return typeof args[0] === 'string' ? Number.parseFloat(args[0]) : asNum(args[0] ?? 0);
    case 'bool':
      return truthy(args[0] ?? false);
    case 'abs':
      return Math.abs(asNum(args[0] ?? 0));
    case 'min':
      return Math.min(...values.map(asNum));
    case 'max':
      return Math.max(...values.map(asNum));
    case 'sum':
      return items(args[0]).reduce<number>((s, v) => s + asNum(v), 0);
    case 'round':
      return pyRound(asNum(args[0] ?? 0), args[1] === undefined ? 0 : asNum(args[1]));
    case 'len':
      return typeof args[0] === 'string' ? args[0].length : items(args[0]).length;
    case 'tuple':
    case 'list':
      return [...items(args[0] ?? [])];
  }
  throw new ExprError(msg(`不支持函数 ${e.func.id}`, `function ${e.func.id} is not supported`));
}

export function evalExpr(text: string, env: Env): EvalResult {
  const parsed = parseExpr(text);
  if (!parsed.ok) return parsed;
  try {
    const value = evaluate(parsed.expr, env);
    if (typeof value === 'number' && !Number.isFinite(value) && value !== Number.POSITIVE_INFINITY) {
      return { ok: false, error: msg('结果不是有限的数', 'the result is not a finite number') };
    }
    return { ok: true, value };
  } catch (e) {
    if (e instanceof Unknown) return { ok: false, error: msg('依赖未知的值', 'depends on an unknown value'), unknown: true };
    if (e instanceof ExprError) return { ok: false, error: e.message };
    throw e;
  }
}
