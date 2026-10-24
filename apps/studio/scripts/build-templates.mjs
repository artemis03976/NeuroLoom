// Imports every templates/*.py and blocks/*.py through the Python backend and
// writes src/templates/templates.json (built-in templates, PRD 12.2) and
// src/templates/blocks.json (composite module library, PRD 9.6), so both open
// offline (no Python needed).
//
// Usage: npm run templates -w @neuroloom/studio
//
// Each file starts with "# key: value" lines:
//   title, zh, description, description_zh  names and one-line descriptions
//   paper    (templates) link to the original paper
//   class    the entry class, when it is not the last class no other class uses
//   inputs   input shapes of the entry's forward(), e.g. "x=B,3,224,224" or
//            "tokens=B,T:int64"; "Class.name=..." declares the input of another
//            model in the same project (DCGAN's discriminator)
//
// Shapes of code blocks: inference cannot see into code blocks and opaque modules,
// so nodes after them would have no shapes (and no parameter counts). Projects
// that have any are run once through the backend's probe (`python -m neuroloom
// probe`), the editor's "run once to fill in shapes", and the observed shapes are
// written to node.shapes before saving. The instrumented code and the decoding of
// the two runs into symbolic shapes are the editor's own (src/lib/probe.ts and
// @neuroloom/core, both TypeScript): esbuild (installed with Vite) bundles them
// into one ES module in memory, which this script imports from a data: URL, so
// there is no separate build step and no temporary file.
import { spawnSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const studioRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const pythonDir = join(studioRoot, '../../python');
const venv = join(pythonDir, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
const python = process.env.NEUROLOOM_PYTHON ?? (existsSync(venv) ? venv : 'python');

function runPython(args, input) {
  const proc = spawnSync(python, args, {
    cwd: pythonDir,
    input,
    encoding: 'utf8',
    maxBuffer: 256 * 1024 * 1024,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });
  if (proc.error) throw proc.error;
  const last = proc.stdout.trim().split('\n').at(-1) ?? '';
  if (proc.status !== 0 && !last.startsWith('{')) throw new Error(`python ${args.join(' ')} failed:\n${proc.stderr}`);
  return JSON.parse(last);
}

/** The editor's probe helpers and analysis, bundled from TypeScript. */
async function loadEditor() {
  const result = await build({
    stdin: {
      contents: [
        "export { observedShapes, probeRequest } from './src/lib/probe';",
        "export { analyzeProject, parseProject } from '@neuroloom/core';",
      ].join('\n'),
      resolveDir: studioRoot,
      sourcefile: 'build-templates-entry.ts',
      loader: 'ts',
    },
    bundle: true,
    platform: 'node',
    format: 'esm',
    write: false,
    logLevel: 'silent',
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
}

/** "# key: value" lines at the top of a file. */
function header(source) {
  const meta = {};
  for (const line of source.split('\n')) {
    const m = line.match(/^#\s*(\w+):\s*(.*)$/);
    if (!m) break;
    meta[m[1]] = m[2].trim();
  }
  return meta;
}

/** "x=B,3,32,32; Discriminator.x=B,3,64,64" -> entry inputs, and inputs of other models by class. */
function parseInputs(text) {
  const entry = {};
  const others = [];
  for (const item of text.split(';')) {
    const [name, spec] = item.trim().split('=').map((s) => s.trim());
    const dot = name.indexOf('.');
    if (dot < 0) entry[name] = spec;
    else others.push({ className: name.slice(0, dot), name: name.slice(dot + 1), spec });
  }
  return { entry, others };
}

function readDir(kind, required) {
  const dir = join(studioRoot, kind);
  return readdirSync(dir)
    .filter((f) => f.endsWith('.py'))
    .sort()
    .map((file) => {
      const id = file.replace(/\.py$/, '');
      const source = readFileSync(join(dir, file), 'utf8');
      const meta = header(source);
      const missing = required.filter((key) => !meta[key]);
      if (missing.length > 0) throw new Error(`${kind}/${file}: header lacks ${missing.join(', ')}`);
      return { kind, id, meta, source, inputs: parseInputs(meta.inputs) };
    });
}

/** Declares the input shape of a model other than the entry, so it analyzes on its own. */
function declareInput(project, item, { className, name, spec }) {
  const mod = Object.values(project.modules).find((m) => m.className === className && !m.sequential && !m.opaque);
  const node = mod && mod.interface.inputs.map((id) => mod.nodes[id]).find((n) => n.name === name);
  if (!node) throw new Error(`${item.kind}/${item.id}: no input ${name} in class ${className}`);
  const [dims, dtype = 'float32'] = spec.split(':');
  node.params.shape = { literal: dims.split(',').map((d) => d.trim()).join(', ') };
  node.params.dtype = { literal: dtype };
}

const needsProbe = (project) =>
  Object.values(project.modules).some((m) => m.opaque || Object.values(m.nodes).some((n) => n.kind === 'code'));

/** Runs the model once (twice, in fact) and declares the shapes of code-block and opaque-module outputs. */
function probe(editor, item, project) {
  const request = editor.probeRequest(editor.parseProject(structuredClone(project)));
  const result = runPython(['-m', 'neuroloom', '--lang', 'en', 'probe'], JSON.stringify(request));
  if (result.error) throw new Error(`${item.kind}/${item.id}: probe failed: ${result.error.message}`);
  const observed = editor.observedShapes(result);
  for (const o of observed) {
    const node = project.modules[o.moduleId].nodes[o.nodeId];
    node.shapes = { ...node.shapes, [o.port]: o.text };
  }
  const analysis = editor.analyzeProject(editor.parseProject(structuredClone(project)));
  if (!analysis.entry.outputs.every(Boolean)) throw new Error(`${item.kind}/${item.id}: output shapes still unknown after probing`);
  return observed.length;
}

const NAMES = ['title', 'zh', 'description', 'description_zh', 'inputs'];
const items = [...readDir('templates', [...NAMES, 'paper']), ...readDir('blocks', NAMES)];

const results = runPython(
  ['-m', 'neuroloom', '--lang', 'en', 'import-batch'],
  JSON.stringify(items.map((item) => ({ source: item.source, className: item.meta.class, inputShapes: item.inputs.entry }))),
);

const editor = items.some((item, i) => results[i].project && needsProbe(results[i].project)) ? await loadEditor() : undefined;
const out = { templates: [], blocks: [] };
items.forEach((item, i) => {
  const result = results[i];
  if (result.error) throw new Error(`${item.kind}/${item.id}: ${result.error.message}`);
  const { project } = result;
  for (const other of item.inputs.others) declareInput(project, item, other);
  const probed = needsProbe(project) ? probe(editor, item, project) : 0;
  const levels = result.report.map((r) => `${r.className} ${r.level}`).join(', ');
  console.log(`${item.kind}: ${item.id.padEnd(22)} ${levels}${probed ? ` (probe declared ${probed} shape${probed === 1 ? '' : 's'})` : ''}`);
  for (const warning of result.warnings) console.log(`  warning: ${warning}`);
  out[item.kind].push({
    id: item.id,
    title: item.meta.title,
    titleZh: item.meta.zh,
    description: item.meta.description,
    descriptionZh: item.meta.description_zh,
    ...(item.kind === 'templates' ? { paper: item.meta.paper } : {}),
    project,
  });
});

for (const kind of ['templates', 'blocks']) {
  writeFileSync(join(studioRoot, `src/templates/${kind}.json`), `${JSON.stringify(out[kind], null, 2)}\n`);
}
console.log(`templates: wrote ${out.templates.length} templates and ${out.blocks.length} blocks`);
