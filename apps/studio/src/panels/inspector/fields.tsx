// Inspector building blocks: sections and the fields that edit parameter values.
import { asLiteral, literalSource, parseExpr, type Literal, type ParamSpec, type ParamValue } from '@neuroloom/core';
import { useState, type ReactNode } from 'react';
import { tr } from '../../i18n';
import { formatParam, parseParam } from '../../lib/paramText';

/** Input styling without a width, for fields that share a row. */
export const fieldBase =
  'rounded-md border border-line bg-bg px-2 py-1 font-mono text-[12.5px] outline-none focus:border-primary focus:ring-2 focus:ring-primary-soft disabled:opacity-60';
export const field = `w-full ${fieldBase}`;

export function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="border-b border-line px-4 py-3">
      <h3 className="mb-2 flex items-center justify-between text-[11px] font-semibold tracking-wide text-muted">
        <span>{title}</span>
        {aside}
      </h3>
      {children}
    </section>
  );
}

/** How a value is written in a value field: Python source for literals, the text for expressions. */
export function valueText(value: ParamValue | undefined): string {
  if (!value) return '';
  if ('expr' in value) return value.expr;
  if ('literal' in value) return literalSource(value.literal);
  return '';
}

/** Field text -> a literal or an expression; empty text means "not set". */
export function parseValue(text: string): { ok: true; value: ParamValue | undefined } | { ok: false; error: string } {
  const t = text.trim();
  if (t === '') return { ok: true, value: undefined };
  const parsed = parseExpr(t);
  if (!parsed.ok) return { ok: false, error: parsed.error };
  const lit = asLiteral(t);
  return { ok: true, value: lit !== undefined ? { literal: lit } : { expr: t } };
}

/**
 * A free value: a Python literal (64, 0.1, "gelu", (3, 3)) or an expression over
 * the module's hyperparameters (d_model * 4). Commits as you type when it parses.
 */
export function ValueField({
  value,
  onCommit,
  placeholder,
  hint,
}: {
  value: ParamValue | undefined;
  onCommit: (v: ParamValue | undefined) => void;
  placeholder?: string;
  /** The evaluated value, shown under expressions. */
  hint?: string;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const text = draft ?? valueText(value);
  const parsed = parseValue(text);
  return (
    <>
      <input
        className={`${field} ${!parsed.ok ? 'border-error' : ''}`}
        value={text}
        placeholder={placeholder}
        spellCheck={false}
        onChange={(e) => {
          setDraft(e.target.value);
          const next = parseValue(e.target.value);
          if (next.ok) onCommit(next.value);
        }}
        onBlur={() => setDraft(null)}
      />
      {!parsed.ok && draft !== null && <p className="mt-1 text-[11px] text-error">{parsed.error}</p>}
      {parsed.ok && hint !== undefined && value && 'expr' in value && <p className="mt-0.5 font-mono text-[11px] text-muted">= {hint}</p>}
    </>
  );
}

/**
 * A typed parameter field that commits only values that parse; keeps the raw text
 * while typing. Text starting with "=" is an expression over hyperparameters.
 */
export function TextParam({
  spec,
  value,
  onCommit,
  disabled,
}: {
  spec: ParamSpec;
  value: Literal | undefined;
  onCommit: (v: ParamValue) => void;
  disabled?: boolean;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const text = draft ?? formatParam(value);
  const parse = (t: string): { ok: true; value: ParamValue } | { ok: false; error: string } => {
    if (t.trimStart().startsWith('=')) {
      const e = parseExpr(t.trim().slice(1));
      return e.ok ? { ok: true, value: { expr: t.trim().slice(1).trim() } } : e;
    }
    const r = parseParam(t, spec);
    return r.ok ? { ok: true, value: { literal: r.value } } : r;
  };
  const parsed = parse(text);
  return (
    <>
      <input
        className={`${field} ${!parsed.ok && !disabled ? 'border-error' : ''}`}
        value={text}
        disabled={disabled}
        spellCheck={false}
        title={tr('以 = 开头输入表达式，如 =d_model * 4', 'Start with = for an expression, e.g. =d_model * 4')}
        onChange={(e) => {
          setDraft(e.target.value);
          const next = parse(e.target.value);
          if (next.ok) onCommit(next.value);
        }}
        onBlur={() => setDraft(null)}
      />
      {!parsed.ok && !disabled && draft !== null && <p className="mt-1 text-[11px] text-error">{parsed.error}</p>}
    </>
  );
}

/** An expression field, for parameters currently given as an expression. */
export function ExprParam({ text, resolved, onCommit }: { text: string; resolved?: Literal; onCommit: (v: ParamValue) => void }) {
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? text;
  const parsed = parseExpr(shown);
  return (
    <>
      <div className="flex items-center gap-1">
        <span className="font-mono text-[12px] text-primary">=</span>
        <input
          className={`${field} ${!parsed.ok ? 'border-error' : ''}`}
          value={shown}
          spellCheck={false}
          onChange={(e) => {
            setDraft(e.target.value);
            if (parseExpr(e.target.value).ok) onCommit({ expr: e.target.value.trim() });
          }}
          onBlur={() => setDraft(null)}
        />
      </div>
      {!parsed.ok && draft !== null && <p className="mt-1 text-[11px] text-error">{parsed.error}</p>}
      {parsed.ok && resolved !== undefined && <p className="mt-0.5 font-mono text-[11px] text-muted">= {literalSource(resolved)}</p>}
    </>
  );
}

/** A comma-separated list of Python identifiers (code block variables). */
export function NamesField({ names, onCommit }: { names: string[]; onCommit: (names: string[]) => void }) {
  const [draft, setDraft] = useState<string | null>(null);
  const text = draft ?? names.join(', ');
  const parts = text.split(',').map((s) => s.trim()).filter(Boolean);
  const valid = parts.every((p) => /^[A-Za-z_]\w*$/.test(p)) && new Set(parts).size === parts.length;
  return (
    <>
      <input
        className={`${field} ${!valid ? 'border-error' : ''}`}
        value={text}
        spellCheck={false}
        onChange={(e) => {
          setDraft(e.target.value);
          const next = e.target.value.split(',').map((s) => s.trim()).filter(Boolean);
          if (next.every((p) => /^[A-Za-z_]\w*$/.test(p)) && new Set(next).size === next.length) onCommit(next);
        }}
        onBlur={() => setDraft(null)}
      />
      {!valid && <p className="mt-1 text-[11px] text-error">{tr('请输入用逗号分隔的变量名', 'Enter variable names separated by commas')}</p>}
    </>
  );
}

/** A multi-line text area that commits as you type (code, preamble). */
export function CodeArea({ value, onCommit, rows = 6, placeholder }: { value: string; onCommit: (v: string) => void; rows?: number; placeholder?: string }) {
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <textarea
      className={`${field} resize-y leading-relaxed`}
      rows={rows}
      value={draft ?? value}
      placeholder={placeholder}
      spellCheck={false}
      onChange={(e) => {
        setDraft(e.target.value);
        onCommit(e.target.value);
      }}
      onBlur={() => setDraft(null)}
      onKeyDown={(e) => {
        // Tab indents instead of leaving the field.
        if (e.key !== 'Tab') return;
        e.preventDefault();
        const el = e.currentTarget;
        const { selectionStart: s, selectionEnd: end } = el;
        const next = `${el.value.slice(0, s)}    ${el.value.slice(end)}`;
        setDraft(next);
        onCommit(next);
        requestAnimationFrame(() => el.setSelectionRange(s + 4, s + 4));
      }}
    />
  );
}
