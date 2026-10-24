// Standalone vector image of the canvas (PRD F7.3). The editor's state is turned
// into a plain-data Scene (see exportScene.ts); this module lays each node out the
// way styles.css draws it and serializes everything to one SVG with concrete
// colours, inline fonts and no references to the page, so the result is the same
// in a browser, a vector editor or a PDF. Animations, the background pattern,
// controls, the minimap and selection highlights never appear.
import { getBezierPath, Position } from '@xyflow/react';

export type PaletteName = 'light' | 'dark' | 'print';

export interface Palette {
  background: string;
  surface: string;
  surface2: string;
  border: string;
  borderStrong: string;
  text: string;
  muted: string;
  /** Edges whose tensor shape is known. */
  flow: string;
  flowOpacity: number;
  /** Outline of the shape labels on edges. */
  labelBorder: string;
  error: string;
  categories: Record<string, string>;
}

// Mirrors brand/tokens.css (generated from brand/palette.json). The category
// colours are the same in every theme.
const CATEGORIES: Record<string, string> = {
  io: '#8C95B0',
  linear: '#6576FF',
  conv: '#3D8BFF',
  pool: '#31B8E0',
  norm: '#20B5A0',
  activation: '#FF8A3D',
  dropout: '#A3AABF',
  recurrent: '#EE5A92',
  attention: '#B867F5',
  tensor: '#4CBF6B',
  math: '#A6C93A',
  module: '#8B7DFF',
  code: '#737B90',
};

export const PALETTES: Record<PaletteName, Palette> = {
  light: {
    background: '#F6F7FB',
    surface: '#FFFFFF',
    surface2: '#EDEFF6',
    border: '#DCE0EC',
    borderStrong: '#C2C8D9',
    text: '#191E33',
    muted: '#6D7692',
    flow: '#0AA3B5',
    flowOpacity: 0.85,
    labelBorder: '#DCE0EC',
    error: '#E5484D',
    categories: CATEGORIES,
  },
  dark: {
    background: '#0B0E1C',
    surface: '#191E33',
    surface2: '#282E45',
    border: '#282E45',
    borderStrong: '#3D445D',
    text: '#EDEFF6',
    muted: '#969EB6',
    flow: '#3FE0F0',
    flowOpacity: 0.85,
    labelBorder: '#282E45',
    error: '#E5484D',
    categories: CATEGORIES,
  },
  // For papers: white paper, black lines and text; category accents stay.
  print: {
    background: '#FFFFFF',
    surface: '#FFFFFF',
    surface2: '#EEEEEE',
    border: '#000000',
    borderStrong: '#000000',
    text: '#000000',
    muted: '#444444',
    flow: '#000000',
    flowOpacity: 1,
    labelBorder: '#9E9E9E',
    error: '#C62828',
    categories: CATEGORIES,
  },
};

export const SANS_FONT = "'Segoe UI Variable', 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif";
export const MONO_FONT = "'JetBrains Mono', 'Cascadia Code', Consolas, ui-monospace, monospace";

export interface FontSpec {
  size: number;
  mono?: boolean;
  weight?: number;
}

/** Width of a line of text in px; the editor passes one backed by a canvas. */
export type MeasureText = (text: string, font: FontSpec) => number;

export interface SceneHandle {
  id: string;
  type: 'source' | 'target';
  position: Position;
  /** Bounds relative to the node's top-left corner, as React Flow measured them. */
  x: number;
  y: number;
  width: number;
  height: number;
  /** A spare variadic input, drawn dashed. */
  optional?: boolean;
}

export interface ScenePortLabel {
  label: string;
  /** The id of the handle on this row, which places the row. */
  handle?: string;
  optional?: boolean;
}

/** One node as the canvas draws it, in flow coordinates. */
export interface SceneNode {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  /** Category key (io, conv, …, module, code) for the accent colour. */
  category: string;
  variant: 'layer' | 'module' | 'code';
  glyph: string;
  name: string;
  type: string;
  /** Repeat badge text, e.g. "×4". */
  repeat?: string;
  /** Port label rows, for nodes with several inputs or outputs. */
  rows?: { input?: ScenePortLabel; output?: ScenePortLabel }[];
  summary?: string;
  /** Formatted parameter count. */
  params?: string;
  error?: string;
  handles: SceneHandle[];
}

export interface SceneEdge {
  id: string;
  source: string;
  sourceHandle?: string | null;
  target: string;
  targetHandle?: string | null;
  /** The tensor shape; edges without one are drawn dashed and unlabelled. */
  shape?: string;
}

export interface Scene {
  title: string;
  nodes: SceneNode[];
  edges: SceneEdge[];
}

export interface SvgOptions {
  palette: PaletteName;
  /** Multiplies the width and height attributes (for rasterizing); the drawing is unchanged. */
  scale?: number;
  /** Space around the content, in px. */
  margin?: number;
  measure?: MeasureText;
}

export interface SvgImage {
  svg: string;
  /** Size of the drawing in CSS px, before `scale`. */
  width: number;
  height: number;
}

// ---------------------------------------------------------------- layout (styles.css)

const BORDER = 1;
const RADIUS = 10;
const BAR = 3;
const PAD_X = 10;
const GLYPH = 20;
const HEAD_TOP = BORDER + BAR + 8;
const NAME_LINE = 15; // 12.5px × 1.2
const TYPE_LINE = 16.5; // 11px, line-height: normal (Segoe UI Variable)
const HEAD_CONTENT = Math.max(GLYPH, NAME_LINE + TYPE_LINE);
/** Where the port rows start, below the header. */
export const HEAD_BOTTOM = HEAD_TOP + HEAD_CONTENT + 6;
export const PORT_ROW = 20;
const TEXT_X = BORDER + PAD_X + GLYPH + 8;
const BODY_LINE = 16.1; // 11.5px × 1.4
const BADGE_H = 10.5 * 1.4 + 2;
const BODY_PAD_BOTTOM = 8;
const ERROR_LINE = 15.4; // 11px × 1.4
const ERROR_MAX = 240;
const CODE_LINE_MAX = 260;
const LABEL_H = 10.5 * 1.5 + 2 + 2;
const LABEL_GAP = 6;
const REPEAT_OFFSET = 8;

// ---------------------------------------------------------------- text

/** A rough width when no canvas is at hand (tests): monospace 0.6 em, wide CJK glyphs 1 em. */
export const estimateText: MeasureText = (text, font) => {
  let em = 0;
  for (const ch of text) {
    const code = ch.codePointAt(0)!;
    if (code >= 0x2e80) em += 1;
    else if (font.mono) em += 0.6;
    else if (/[A-Z]/.test(ch)) em += 0.65;
    else if (/[ijlI.,:;'!| ]/.test(ch)) em += 0.3;
    else em += 0.55;
  }
  return em * font.size * (font.weight && font.weight >= 600 ? 1.05 : 1);
};

/**
 * Node boxes are as wide as their content, and React Flow measures them in whole
 * pixels: text that fitted on the canvas may be a fraction of a pixel too wide here.
 */
const FIT_SLACK = 2;

/** Cuts text to fit `max` px, ending in an ellipsis like `text-overflow: ellipsis`. */
export function fitText(text: string, max: number, font: FontSpec, measure: MeasureText): string {
  if (measure(text, font) <= max + FIT_SLACK) return text;
  const chars = [...text];
  let lo = 0;
  let hi = chars.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (measure(`${chars.slice(0, mid).join('')}…`, font) <= max) lo = mid;
    else hi = mid - 1;
  }
  return `${chars.slice(0, lo).join('').trimEnd()}…`;
}

/** Greedy line breaking at spaces, and between any two CJK characters. */
export function wrapText(text: string, max: number, font: FontSpec, measure: MeasureText): string[] {
  const tokens = text.match(/[⺀-￿]|[^\s⺀-￿]+|\s+/g) ?? [];
  const lines: string[] = [];
  let line = '';
  for (const token of tokens) {
    const next = line + token;
    if (line.trim() && measure(next.trimEnd(), font) > max) {
      lines.push(line.trimEnd());
      line = /^\s+$/.test(token) ? '' : token;
    } else {
      line = next;
    }
  }
  if (line.trim()) lines.push(line.trimEnd());
  return lines;
}

const escapeXml = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[c]!);

/** Numbers with at most two decimals, and no "-0". */
const num = (v: number): string => {
  const r = Math.round(v * 100) / 100;
  return String(r === 0 ? 0 : r);
};

const roundPath = (d: string) => d.replace(/-?\d*\.?\d+(?:e[-+]?\d+)?/gi, (m) => num(Number(m)));

/** Baseline for text vertically centred on `center` (cap height is ~0.7 em). */
const baseline = (center: number, size: number) => center + size * 0.35;

interface TextOptions {
  size: number;
  fill: string;
  mono?: boolean;
  weight?: number;
  anchor?: 'start' | 'middle' | 'end';
  opacity?: number;
}

function text(x: number, center: number, content: string, o: TextOptions): string {
  const attrs = [
    `x="${num(x)}"`,
    `y="${num(baseline(center, o.size))}"`,
    `font-size="${num(o.size)}"`,
    o.mono ? `font-family="${MONO_FONT}"` : '',
    o.weight ? `font-weight="${o.weight}"` : '',
    o.anchor && o.anchor !== 'start' ? `text-anchor="${o.anchor}"` : '',
    `fill="${o.fill}"`,
    o.opacity !== undefined ? `fill-opacity="${o.opacity}"` : '',
  ].filter(Boolean);
  return `<text ${attrs.join(' ')}>${escapeXml(content)}</text>`;
}

const rect = (x: number, y: number, w: number, h: number, r: number, attrs: string) =>
  `<rect x="${num(x)}" y="${num(y)}" width="${num(w)}" height="${num(h)}" rx="${num(r)}" ${attrs}/>`;

// ---------------------------------------------------------------- geometry

interface Box {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

const grow = (b: Box, x: number, y: number) => {
  b.minX = Math.min(b.minX, x);
  b.minY = Math.min(b.minY, y);
  b.maxX = Math.max(b.maxX, x);
  b.maxY = Math.max(b.maxY, y);
};

/** The point React Flow attaches an edge to (getHandlePosition in @xyflow/system). */
export function handlePoint(node: SceneNode, handle: SceneHandle): { x: number; y: number } {
  const x = node.x + handle.x;
  const y = node.y + handle.y;
  switch (handle.position) {
    case Position.Top:
      return { x: x + handle.width / 2, y };
    case Position.Right:
      return { x: x + handle.width, y: y + handle.height / 2 };
    case Position.Bottom:
      return { x: x + handle.width / 2, y: y + handle.height };
    default:
      return { x, y: y + handle.height / 2 };
  }
}

const findHandle = (node: SceneNode, type: SceneHandle['type'], id?: string | null) => {
  const candidates = node.handles.filter((h) => h.type === type);
  return id ? candidates.find((h) => h.id === id) : candidates[0];
};

/** Samples a cubic "M x,y C x,y x,y x,y" path for its bounds. */
function bezierBounds(d: string, box: Box): void {
  const n = (d.match(/-?\d*\.?\d+(?:e[-+]?\d+)?/gi) ?? []).map(Number);
  if (n.length < 8) return;
  const [x0, y0, x1, y1, x2, y2, x3, y3] = n as [number, number, number, number, number, number, number, number];
  for (let i = 0; i <= 32; i++) {
    const t = i / 32;
    const u = 1 - t;
    const a = u * u * u;
    const b = 3 * u * u * t;
    const c = 3 * u * t * t;
    const e = t * t * t;
    grow(box, a * x0 + b * x1 + c * x2 + e * x3, a * y0 + b * y1 + c * y2 + e * y3);
  }
}

/** The nodes with these ids and the edges between them. */
export function subsetScene(scene: Scene, ids: Iterable<string>): Scene {
  const keep = new Set(ids);
  return {
    ...scene,
    nodes: scene.nodes.filter((n) => keep.has(n.id)),
    edges: scene.edges.filter((e) => keep.has(e.source) && keep.has(e.target)),
  };
}

// ---------------------------------------------------------------- drawing

/** Header tint of a module instance: the node's top, to the bottom of the header. */
const headPath = (w: number) =>
  `M${BORDER},${num(HEAD_BOTTOM)} L${BORDER},${BORDER + RADIUS} A${RADIUS} ${RADIUS} 0 0 1 ${BORDER + RADIUS},${BORDER} ` +
  `L${num(w - BORDER - RADIUS)},${BORDER} A${RADIUS} ${RADIUS} 0 0 1 ${num(w - BORDER)},${BORDER + RADIUS} L${num(w - BORDER)},${num(HEAD_BOTTOM)} Z`;

/**
 * The category bar: a 3px top border on a header with 10px corners, which tapers
 * down the curve of each corner (inner edge an ellipse of 10 × 7).
 */
const barPath = (w: number) => {
  const l = BORDER;
  const r = w - BORDER;
  const inner = RADIUS - BAR;
  return (
    `M${l},${l + RADIUS} A${RADIUS} ${RADIUS} 0 0 1 ${l + RADIUS},${l} L${num(r - RADIUS)},${l} A${RADIUS} ${RADIUS} 0 0 1 ${num(r)},${l + RADIUS} ` +
    `A${RADIUS} ${inner} 0 0 0 ${num(r - RADIUS)},${l + BAR} L${l + RADIUS},${l + BAR} A${RADIUS} ${inner} 0 0 0 ${l},${l + RADIUS} Z`
  );
};

function drawNode(node: SceneNode, p: Palette, measure: MeasureText): string {
  const { width: w, height: h } = node;
  const cat = p.categories[node.category] ?? p.categories.code ?? p.muted;
  const out: string[] = [`<g transform="translate(${num(node.x)},${num(node.y)})">`];

  // A repeat is a stack of cards.
  if (node.repeat) {
    out.push(rect(REPEAT_OFFSET + 0.5, REPEAT_OFFSET + 0.5, w - 1, h - 1, RADIUS - 0.5, `fill="${p.surface}" stroke="${p.border}"`));
    out.push(rect(REPEAT_OFFSET / 2 + 0.5, REPEAT_OFFSET / 2 + 0.5, w - 1, h - 1, RADIUS - 0.5, `fill="${p.surface}" stroke="${p.borderStrong}"`));
  }
  const stroke = node.error ? p.error : p.border;
  out.push(rect(0.5, 0.5, w - 1, h - 1, RADIUS - 0.5, `fill="${p.surface}" stroke="${stroke}"${node.variant === 'code' ? ' stroke-dasharray="4 3"' : ''}`));
  if (node.variant === 'module') out.push(`<path d="${headPath(w)}" fill="${cat}" fill-opacity="0.08"/>`);
  out.push(`<path d="${barPath(w)}" fill="${cat}"/>`);

  // Header: glyph, name over type, repeat badge.
  const glyphY = HEAD_TOP + (HEAD_CONTENT - GLYPH) / 2;
  const glyphX = BORDER + PAD_X;
  out.push(rect(glyphX, glyphY, GLYPH, GLYPH, 6, `fill="${cat}"`));
  out.push(text(glyphX + GLYPH / 2, glyphY + GLYPH / 2, node.glyph, { size: 10, mono: true, weight: 700, anchor: 'middle', fill: '#FFFFFF' }));
  let headRight = w - BORDER - PAD_X;
  if (node.repeat) {
    const font = { size: 11, mono: true, weight: 700 };
    const bw = measure(node.repeat, font) + 14;
    const bh = 11 * 1.5 + 2;
    const center = HEAD_TOP + HEAD_CONTENT / 2;
    out.push(rect(headRight - bw, center - bh / 2, bw, bh, bh / 2, `fill="${cat}"`));
    out.push(text(headRight - bw / 2, center, node.repeat, { ...font, anchor: 'middle', fill: '#FFFFFF' }));
    headRight -= bw + 8;
  }
  const nameFont = { size: 12.5, mono: true, weight: 600 };
  out.push(text(TEXT_X, HEAD_TOP + NAME_LINE / 2, fitText(node.name, headRight - TEXT_X, nameFont, measure), { ...nameFont, fill: p.text }));
  out.push(text(TEXT_X, HEAD_TOP + NAME_LINE + TYPE_LINE / 2, fitText(node.type, headRight - TEXT_X, { size: 11 }, measure), { size: 11, fill: p.muted }));

  // Port rows line up with their handles.
  const rowCenter = (i: number) => {
    const row = node.rows?.[i];
    const hd =
      (row?.input?.handle !== undefined && node.handles.find((x) => x.type === 'target' && x.id === row.input!.handle)) ||
      (row?.output?.handle !== undefined && node.handles.find((x) => x.type === 'source' && x.id === row.output!.handle));
    return hd ? hd.y + hd.height / 2 : HEAD_BOTTOM + PORT_ROW * i + PORT_ROW / 2;
  };
  const portFont = { size: 10.5, mono: true };
  node.rows?.forEach((row, i) => {
    const center = rowCenter(i);
    if (row.input) {
      out.push(text(BORDER + 12, center, row.input.label, { ...portFont, fill: p.muted, opacity: row.input.optional ? 0.5 : undefined }));
    }
    if (row.output) out.push(text(w - BORDER - 12, center, row.output.label, { ...portFont, fill: p.muted, anchor: 'end' }));
  });

  // Error and summary rows sit at the bottom of the measured box.
  let bottom = h - BORDER;
  if (node.error) {
    const boxW = Math.min(ERROR_MAX, w - 2 * BORDER - 16);
    const font = { size: 11 };
    const lines = wrapText(node.error, boxW - 16, font, measure);
    const boxH = lines.length * ERROR_LINE + 8;
    const top = bottom - 8 - boxH;
    out.push(rect(BORDER + 8, top, boxW, boxH, 6, `fill="${p.error}" fill-opacity="0.12"`));
    lines.forEach((line, i) => out.push(text(BORDER + 16, top + 4 + ERROR_LINE * i + ERROR_LINE / 2, line, { ...font, fill: p.error })));
    bottom = top;
  }
  if (node.summary || node.params) {
    const contentH = node.params ? Math.max(BODY_LINE, BADGE_H) : BODY_LINE;
    const center = bottom - BODY_PAD_BOTTOM - contentH / 2;
    const right = Math.min(w, node.variant === 'code' ? CODE_LINE_MAX : Infinity) - BORDER - PAD_X;
    let summaryRight = right;
    if (node.params) {
      const font = { size: 10.5, mono: true };
      const bw = measure(node.params, font) + 12;
      out.push(rect(right - bw, center - BADGE_H / 2, bw, BADGE_H, BADGE_H / 2, `fill="${p.surface2}"`));
      out.push(text(right - bw / 2, center, node.params, { ...font, anchor: 'middle', fill: p.muted }));
      summaryRight -= bw + 8;
    }
    if (node.summary) {
      const font = { size: 11.5, mono: true };
      const fitted = fitText(node.summary, summaryRight - TEXT_X, font, measure);
      out.push(text(TEXT_X, center, fitted, { ...font, fill: node.variant === 'code' ? p.text : p.muted }));
    }
  }

  // Handles last: they sit on top of the border.
  for (const hd of node.handles) {
    const r = Math.min(hd.width, hd.height) / 2 - 1;
    out.push(
      `<circle cx="${num(hd.x + hd.width / 2)}" cy="${num(hd.y + hd.height / 2)}" r="${num(r)}" fill="${p.surface}" stroke="${p.borderStrong}" stroke-width="2"${hd.optional ? ' stroke-dasharray="2 1.5"' : ''}/>`,
    );
  }
  out.push('</g>');
  return out.join('');
}

/** Serializes the scene to a standalone SVG document fitted to its content. */
export function buildSvg(scene: Scene, options: SvgOptions): SvgImage {
  const p = PALETTES[options.palette];
  const measure = options.measure ?? estimateText;
  const margin = options.margin ?? 24;
  const scale = options.scale ?? 1;
  const box: Box = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  const byId = new Map(scene.nodes.map((n) => [n.id, n]));

  for (const n of scene.nodes) {
    const extra = n.repeat ? REPEAT_OFFSET : 0;
    grow(box, n.x, n.y);
    grow(box, n.x + n.width + extra, n.y + n.height + extra);
    for (const hd of n.handles) {
      grow(box, n.x + hd.x, n.y + hd.y);
      grow(box, n.x + hd.x + hd.width, n.y + hd.y + hd.height);
    }
  }

  const edges: string[] = [];
  const labels: string[] = [];
  const labelFont = { size: 10.5, mono: true };
  for (const e of scene.edges) {
    const source = byId.get(e.source);
    const target = byId.get(e.target);
    const sh = source && findHandle(source, 'source', e.sourceHandle);
    const th = target && findHandle(target, 'target', e.targetHandle);
    // React Flow does not draw an edge whose handle is missing either.
    if (!source || !target || !sh || !th) continue;
    const s = handlePoint(source, sh);
    const t = handlePoint(target, th);
    const [rawPath, labelX, labelY] = getBezierPath({
      sourceX: s.x,
      sourceY: s.y,
      sourcePosition: sh.position,
      targetX: t.x,
      targetY: t.y,
      targetPosition: th.position,
    });
    const d = roundPath(rawPath);
    bezierBounds(d, box);
    const live = e.shape !== undefined;
    edges.push(
      live
        ? `<path d="${d}" stroke="${p.flow}" stroke-width="1.75"${p.flowOpacity < 1 ? ` stroke-opacity="${p.flowOpacity}"` : ''}/>`
        : `<path d="${d}" stroke="${p.borderStrong}" stroke-width="1.75" stroke-dasharray="5 5"/>`,
    );
    if (live) {
      const lw = measure(e.shape!, labelFont) + 14;
      const bottom = labelY - LABEL_GAP;
      const x = labelX - lw / 2;
      const y = bottom - LABEL_H;
      grow(box, x, y);
      grow(box, x + lw, bottom);
      labels.push(rect(x + 0.5, y + 0.5, lw - 1, LABEL_H - 1, 6, `fill="${p.surface}" stroke="${p.labelBorder}"`));
      labels.push(text(labelX, y + LABEL_H / 2, e.shape!, { ...labelFont, anchor: 'middle', fill: p.muted }));
    }
  }

  if (!Number.isFinite(box.minX)) Object.assign(box, { minX: 0, minY: 0, maxX: 0, maxY: 0 });
  const minX = Math.floor(box.minX - margin);
  const minY = Math.floor(box.minY - margin);
  const width = Math.ceil(box.maxX + margin) - minX;
  const height = Math.ceil(box.maxY + margin) - minY;

  const svg = [
    '<?xml version="1.0" encoding="UTF-8"?>\n',
    `<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="${num(width * scale)}" height="${num(height * scale)}" viewBox="${minX} ${minY} ${width} ${height}" font-family="${SANS_FONT}">`,
    `<title>${escapeXml(scene.title)}</title>`,
    `<rect x="${minX}" y="${minY}" width="${width}" height="${height}" fill="${p.background}"/>`,
    `<g fill="none" stroke-linecap="round">${edges.join('')}</g>`,
    `<g>${labels.join('')}</g>`,
    `<g>${scene.nodes.map((n) => drawNode(n, p, measure)).join('')}</g>`,
    '</svg>\n',
  ].join('');
  return { svg, width, height };
}
