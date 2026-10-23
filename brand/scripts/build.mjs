// Generates every brand asset from palette.json: logo SVGs, outlined wordmark
// lockups, PNG app icons, favicon.ico, tokens.css and the preview page.
// Run with `npm run brand` from the repo root.
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { inflateSync } from 'node:zlib';
import opentype from 'opentype.js';
import { Resvg } from '@resvg/resvg-js';

const brandDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const assetsDir = join(brandDir, 'assets');
const iconsDir = join(assetsDir, 'icons');
mkdirSync(iconsDir, { recursive: true });

const palette = JSON.parse(readFileSync(join(brandDir, 'palette.json'), 'utf8'));
const { ink, warp, weft } = palette.brand;
const neutral = palette.neutral;

// ---------------------------------------------------------------------------
// Logo mark: a woven "N". The warp threads (violet) form the N and stand for
// network structure; the weft thread (cyan) is the tensor flowing through it.
// The weft passes over both uprights and under the diagonal.
// ---------------------------------------------------------------------------

const STROKE = 5.5;
const GAP = 2.5;

function markBody({ warpColor, weftColor, idPrefix }) {
  const cut = STROKE + GAP * 2;
  return `<defs>
    <mask id="${idPrefix}-warp-cut" maskUnits="userSpaceOnUse" x="0" y="0" width="64" height="64">
      <rect width="64" height="64" fill="#fff"/>
      <path d="M10 32H50" stroke="#000" stroke-width="${cut}" stroke-linecap="round"/>
    </mask>
    <mask id="${idPrefix}-weft-cut" maskUnits="userSpaceOnUse" x="0" y="0" width="64" height="64">
      <rect width="64" height="64" fill="#fff"/>
      <path d="M20 18L44 46" stroke="#000" stroke-width="${cut}"/>
    </mask>
  </defs>
  <g stroke="${warpColor}" stroke-width="${STROKE}" stroke-linecap="round">
    <path d="M20 18V46M44 18V46" mask="url(#${idPrefix}-warp-cut)"/>
    <path d="M20 18L44 46"/>
  </g>
  <g fill="${warpColor}">
    <circle cx="20" cy="18" r="4.4"/><circle cx="44" cy="18" r="4.4"/>
    <circle cx="20" cy="46" r="4.4"/><circle cx="44" cy="46" r="4.4"/>
  </g>
  <path d="M10 32H50" stroke="${weftColor}" stroke-width="${STROKE}" stroke-linecap="round" mask="url(#${idPrefix}-weft-cut)"/>
  <circle cx="52.5" cy="32" r="4.4" fill="${weftColor}"/>`;
}

function markSvg({ variant, size = 64 }) {
  const variants = {
    tile: { warpColor: warp['300'], weftColor: weft['400'], tile: true },
    light: { warpColor: warp['600'], weftColor: weft['600'], tile: false },
    dark: { warpColor: warp['300'], weftColor: weft['400'], tile: false },
    mono: { warpColor: 'currentColor', weftColor: 'currentColor', tile: false },
  };
  const v = variants[variant];
  const tile = v.tile ? `<rect width="64" height="64" rx="14" fill="${ink}"/>` : '';
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 64 64" fill="none" role="img" aria-label="NeuroLoom">
  ${tile}
  ${markBody({ ...v, idPrefix: `nl-${variant}` })}
</svg>
`;
}

// ---------------------------------------------------------------------------
// Fonts: wordmarks are outlined to paths so the files render identically
// everywhere, with no font dependency.
// ---------------------------------------------------------------------------

function findPackageDir(name) {
  for (const base of [join(brandDir, 'node_modules'), join(brandDir, '..', 'node_modules')]) {
    const dir = join(base, ...name.split('/'));
    if (existsSync(dir)) return dir;
  }
  throw new Error(`Cannot find ${name}; run npm install first`);
}

// opentype.js's built-in inflate rejects some Fontsource WOFF files, so unpack
// WOFF to a plain sfnt with node:zlib first.
function woffToSfnt(woff) {
  const numTables = woff.readUInt16BE(12);
  const tables = [];
  for (let i = 0; i < numTables; i++) {
    const e = 44 + i * 20;
    const offset = woff.readUInt32BE(e + 4);
    const compLength = woff.readUInt32BE(e + 8);
    const origLength = woff.readUInt32BE(e + 12);
    const raw = woff.subarray(offset, offset + compLength);
    tables.push({
      tag: woff.subarray(e, e + 4),
      checksum: woff.readUInt32BE(e + 16),
      data: compLength < origLength ? inflateSync(raw) : raw,
    });
  }
  const entrySelector = Math.floor(Math.log2(numTables));
  const searchRange = 2 ** entrySelector * 16;
  const head = Buffer.alloc(12 + numTables * 16);
  head.writeUInt32BE(woff.readUInt32BE(4), 0);
  head.writeUInt16BE(numTables, 4);
  head.writeUInt16BE(searchRange, 6);
  head.writeUInt16BE(entrySelector, 8);
  head.writeUInt16BE(numTables * 16 - searchRange, 10);
  const chunks = [head];
  let offset = head.length;
  tables.forEach((t, i) => {
    const r = 12 + i * 16;
    t.tag.copy(head, r);
    head.writeUInt32BE(t.checksum, r + 4);
    head.writeUInt32BE(offset, r + 8);
    head.writeUInt32BE(t.data.length, r + 12);
    const padded = Buffer.alloc((t.data.length + 3) & ~3);
    t.data.copy(padded);
    chunks.push(padded);
    offset += padded.length;
  });
  return Buffer.concat(chunks);
}

function loadFont(path) {
  const sfnt = woffToSfnt(readFileSync(path));
  return opentype.parse(sfnt.buffer.slice(sfnt.byteOffset, sfnt.byteOffset + sfnt.byteLength));
}

const displayFont = loadFont(
  join(findPackageDir('@fontsource/space-grotesk'), 'files', 'space-grotesk-latin-700-normal.woff'),
);

// Noto Sans SC ships as many unicode-range subsets; find the ones holding our glyphs.
function loadCjkFontFor(text) {
  const dir = join(findPackageDir('@fontsource/noto-sans-sc'), 'files');
  const files = readdirSync(dir).filter((f) => f.endsWith('-700-normal.woff'));
  const byChar = new Map();
  for (const file of files) {
    const font = loadFont(join(dir, file));
    for (const ch of text) {
      if (!byChar.has(ch) && font.charToGlyphIndex(ch) > 0) byChar.set(ch, font);
    }
    if (byChar.size === new Set(text).size) break;
  }
  for (const ch of text) if (!byChar.has(ch)) throw new Error(`No Noto Sans SC subset contains ${ch}`);
  return byChar;
}

/** Lays out text glyph by glyph (fonts may differ per character) and returns path data. */
function outline(text, { fontFor, x, baseline, size, tracking = 0 }) {
  let cursor = x;
  let d = '';
  let prev = null;
  for (const ch of text) {
    const font = fontFor(ch);
    const glyph = font.charToGlyph(ch);
    const scale = size / font.unitsPerEm;
    if (prev && prev.font === font) cursor += font.getKerningValue(prev.glyph, glyph) * scale;
    d += glyph.getPath(cursor, baseline, size).toPathData(2);
    cursor += glyph.advanceWidth * scale + tracking * size;
    prev = { font, glyph };
  }
  return { d, width: cursor - x };
}

const capHeight = (font, size) => (font.tables.os2.sCapHeight / font.unitsPerEm) * size;

function lockupSvg({ theme, bilingual }) {
  const textColor = theme === 'light' ? neutral['900'] : neutral['100'];
  const accent = theme === 'light' ? warp['600'] : warp['300'];
  const rule = theme === 'light' ? neutral['300'] : neutral['700'];

  const size = 36;
  const baseline = 32 + capHeight(displayFont, size) / 2;
  const tracking = -0.01;
  const latin = () => displayFont;
  const neuro = outline('Neuro', { fontFor: latin, x: 82, baseline, size, tracking });
  const loom = outline('Loom', { fontFor: latin, x: 82 + neuro.width, baseline, size, tracking });
  let width = 82 + neuro.width + loom.width;
  let extra = '';

  if (bilingual) {
    const cjk = loadCjkFontFor('织络');
    const cjkSize = 31;
    const ruleX = width + 22;
    const zh = outline('织络', { fontFor: (ch) => cjk.get(ch), x: ruleX + 22, baseline: baseline + 1, size: cjkSize });
    extra = `<path d="M${ruleX} 16V48" stroke="${rule}" stroke-width="2" stroke-linecap="round"/>
  <path d="${zh.d}" fill="${textColor}"/>`;
    width = ruleX + 22 + zh.width;
  }

  width = Math.ceil(width + 4);
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="64" viewBox="0 0 ${width} 64" fill="none" role="img" aria-label="NeuroLoom${bilingual ? ' 织络' : ''}">
  <rect width="64" height="64" rx="14" fill="${ink}"/>
  ${markBody({ warpColor: warp['300'], weftColor: weft['400'], idPrefix: `nl-lockup-${theme}` })}
  <path d="${neuro.d}" fill="${textColor}"/>
  <path d="${loom.d}" fill="${accent}"/>
  ${extra}
</svg>
`;
}

// ---------------------------------------------------------------------------
// Raster icons and favicon
// ---------------------------------------------------------------------------

function renderPng(svg, size) {
  return new Resvg(svg, { fitTo: { mode: 'width', value: size } }).render().asPng();
}

/** ICO container holding PNG frames (supported by every current OS and browser). */
function buildIco(frames) {
  const header = Buffer.alloc(6 + frames.length * 16);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(frames.length, 4);
  let offset = header.length;
  frames.forEach(({ size, png }, i) => {
    const e = 6 + i * 16;
    header.writeUInt8(size >= 256 ? 0 : size, e);
    header.writeUInt8(size >= 256 ? 0 : size, e + 1);
    header.writeUInt8(0, e + 2);
    header.writeUInt8(0, e + 3);
    header.writeUInt16LE(1, e + 4);
    header.writeUInt16LE(32, e + 6);
    header.writeUInt32LE(png.length, e + 8);
    header.writeUInt32LE(offset, e + 12);
    offset += png.length;
  });
  return Buffer.concat([header, ...frames.map((f) => f.png)]);
}

// ---------------------------------------------------------------------------
// Design tokens
// ---------------------------------------------------------------------------

const semantic = {
  light: {
    bg: neutral['50'],
    surface: '#FFFFFF',
    'surface-2': neutral['100'],
    border: neutral['200'],
    'border-strong': neutral['300'],
    text: neutral['900'],
    'text-muted': neutral['500'],
    primary: warp['600'],
    'primary-hover': warp['700'],
    'primary-fg': '#FFFFFF',
    'primary-soft': `${warp['500']}1F`,
    flow: weft['600'],
    'flow-soft': `${weft['500']}29`,
    'canvas-bg': neutral['50'],
    'canvas-dot': neutral['300'],
    'code-bg': neutral['100'],
  },
  dark: {
    bg: neutral['950'],
    surface: neutral['900'],
    'surface-2': neutral['800'],
    border: neutral['800'],
    'border-strong': neutral['700'],
    text: neutral['100'],
    'text-muted': neutral['400'],
    primary: warp['400'],
    'primary-hover': warp['300'],
    'primary-fg': ink,
    'primary-soft': `${warp['400']}2E`,
    flow: weft['400'],
    'flow-soft': `${weft['400']}2E`,
    'canvas-bg': '#0B0E1C',
    'canvas-dot': neutral['800'],
    'code-bg': '#0B0E1C',
  },
};

function tokensCss() {
  const lines = (obj, prefix) => Object.entries(obj).map(([k, v]) => `  --nl-${prefix}${k}: ${v};`);
  const block = (vars) => vars.join('\n');
  const base = [
    `  --nl-ink: ${ink};`,
    ...lines(warp, 'warp-'),
    ...lines(weft, 'weft-'),
    ...lines(neutral, 'neutral-'),
    ...lines(palette.status, ''),
    ...lines(palette.category, 'cat-'),
  ];
  return `/* Generated by brand/scripts/build.mjs from brand/palette.json. Do not edit by hand. */
:root {
${block(base)}
${block(lines(semantic.light, ''))}
  color-scheme: light;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme='light']) {
${block(lines(semantic.dark, '').map((l) => `  ${l}`))}
    color-scheme: dark;
  }
}

:root[data-theme='dark'] {
${block(lines(semantic.dark, ''))}
  color-scheme: dark;
}
`;
}

// ---------------------------------------------------------------------------
// Preview page
// ---------------------------------------------------------------------------

function swatches(title, entries) {
  const items = entries
    .map(
      ([name, hex]) =>
        `<div class="sw"><div class="chip" style="background:${hex}"></div><b>${name}</b><code>${hex}</code></div>`,
    )
    .join('');
  return `<section><h2>${title}</h2><div class="grid">${items}</div></section>`;
}

function previewHtml() {
  const cat = Object.entries(palette.category);
  return `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeuroLoom 品牌</title>
<link rel="icon" href="assets/favicon.ico">
<link rel="stylesheet" href="tokens.css">
<style>
  body { margin: 0; font: 14px/1.6 "Segoe UI Variable", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif; background: var(--nl-bg); color: var(--nl-text); }
  main { max-width: 960px; margin: 0 auto; padding: 40px 24px 64px; }
  h1 { font-size: 28px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 40px 0 12px; }
  p { color: var(--nl-text-muted); margin: 0; max-width: 640px; }
  .row { display: flex; flex-wrap: wrap; gap: 16px; }
  .card { border: 1px solid var(--nl-border); border-radius: 12px; padding: 24px; display: flex; align-items: center; justify-content: center; min-height: 96px; }
  .on-light { background: #fff; } .on-dark { background: ${neutral['950']}; border-color: ${neutral['800']}; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }
  .sw { border: 1px solid var(--nl-border); border-radius: 10px; overflow: hidden; background: var(--nl-surface); }
  .sw b, .sw code { display: block; padding: 0 10px; } .sw b { padding-top: 8px; font-size: 13px; }
  .sw code { padding-bottom: 8px; color: var(--nl-text-muted); font-size: 12px; }
  .chip { height: 56px; }
  .sizes img { image-rendering: auto; }
</style>
</head>
<body><main>
<h1>NeuroLoom 织络</h1>
<p>经线（紫）是网络结构，纬线（青）是流过它的张量。标志是一个被“织”出来的 N：纬线从两根竖线上方穿过、从斜线下方穿过，末端的圆点表示数据流向输出。</p>

<h2>标志</h2>
<div class="row">
  <div class="card on-light"><img src="assets/neuroloom-mark.svg" width="96" height="96" alt=""></div>
  <div class="card on-light"><img src="assets/neuroloom-mark-light.svg" width="96" height="96" alt=""></div>
  <div class="card on-dark"><img src="assets/neuroloom-mark-dark.svg" width="96" height="96" alt=""></div>
  <div class="card on-light sizes"><img src="assets/icons/icon-16.png" width="16"> &nbsp; <img src="assets/icons/icon-32.png" width="32"> &nbsp; <img src="assets/icons/icon-64.png" width="64"></div>
</div>

<h2>组合标志</h2>
<div class="row">
  <div class="card on-light"><img src="assets/neuroloom-lockup-light.svg" height="56" alt=""></div>
  <div class="card on-dark"><img src="assets/neuroloom-lockup-dark.svg" height="56" alt=""></div>
  <div class="card on-light"><img src="assets/neuroloom-lockup-zh-light.svg" height="56" alt=""></div>
  <div class="card on-dark"><img src="assets/neuroloom-lockup-zh-dark.svg" height="56" alt=""></div>
</div>

${swatches('品牌色：经线（结构）与纬线（数据）', [['Ink 墨', ink], ...Object.entries(warp).map(([k, v]) => [`Warp ${k}`, v]), ...Object.entries(weft).map(([k, v]) => [`Weft ${k}`, v])])}
${swatches('中性色', Object.entries(neutral))}
${swatches('节点类别色', cat)}
${swatches('状态色', Object.entries(palette.status))}
</main></body>
</html>
`;
}

// ---------------------------------------------------------------------------

const files = {
  'neuroloom-mark.svg': markSvg({ variant: 'tile' }),
  'neuroloom-mark-light.svg': markSvg({ variant: 'light' }),
  'neuroloom-mark-dark.svg': markSvg({ variant: 'dark' }),
  'neuroloom-mark-mono.svg': markSvg({ variant: 'mono' }),
  'neuroloom-lockup-light.svg': lockupSvg({ theme: 'light', bilingual: false }),
  'neuroloom-lockup-dark.svg': lockupSvg({ theme: 'dark', bilingual: false }),
  'neuroloom-lockup-zh-light.svg': lockupSvg({ theme: 'light', bilingual: true }),
  'neuroloom-lockup-zh-dark.svg': lockupSvg({ theme: 'dark', bilingual: true }),
};
for (const [name, svg] of Object.entries(files)) writeFileSync(join(assetsDir, name), svg);

const tile = files['neuroloom-mark.svg'];
const icoFrames = [];
for (const size of [16, 32, 48, 64, 128, 180, 256, 512, 1024]) {
  const png = renderPng(tile, size);
  writeFileSync(join(iconsDir, `icon-${size}.png`), png);
  if (size <= 48 || size === 256) icoFrames.push({ size, png });
}
writeFileSync(join(assetsDir, 'favicon.ico'), buildIco(icoFrames));
writeFileSync(join(brandDir, 'tokens.css'), tokensCss());
writeFileSync(join(brandDir, 'preview.html'), previewHtml());

console.log(`brand: wrote ${Object.keys(files).length} SVGs, 9 PNG icons, favicon.ico, tokens.css, preview.html`);
