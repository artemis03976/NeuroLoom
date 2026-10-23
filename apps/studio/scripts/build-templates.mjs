// Imports every templates/*.py through the Python backend and writes
// src/templates/templates.json, so templates open offline (no Python needed).
// Usage: npm run templates -w @neuroloom/studio
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const studioRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const pythonDir = join(studioRoot, '../../python');
const venv = join(pythonDir, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
const python = process.env.NEUROLOOM_PYTHON ?? (existsSync(venv) ? venv : 'python');

/** "# key: value" lines at the top of a template file. */
function header(source) {
  const meta = {};
  for (const line of source.split('\n')) {
    const m = line.match(/^#\s*(\w+):\s*(.*)$/);
    if (!m) break;
    meta[m[1]] = m[2].trim();
  }
  return meta;
}

const dir = join(studioRoot, 'templates');
const files = readdirSync(dir).filter((f) => f.endsWith('.py')).sort();
const templates = files.map((file) => {
  const source = readFileSync(join(dir, file), 'utf8');
  const meta = header(source);
  const inputShapes = Object.fromEntries(meta.inputs.split(';').map((s) => s.trim().split('=')));
  return { id: file.replace(/\.py$/, ''), meta, source, inputShapes };
});

const proc = spawnSync(python, ['-m', 'neuroloom', 'import-batch'], {
  cwd: pythonDir,
  input: JSON.stringify(templates.map((t) => ({ source: t.source, inputShapes: t.inputShapes }))),
  encoding: 'utf8',
  maxBuffer: 64 * 1024 * 1024,
});
if (proc.status !== 0) throw new Error(proc.stderr);
const results = JSON.parse(proc.stdout.trim().split('\n').at(-1));

const out = templates.map((t, i) => {
  const result = results[i];
  if (result.error) throw new Error(`${t.id}: ${result.error.message}`);
  return {
    id: t.id,
    title: t.meta.title,
    titleZh: t.meta.zh,
    description: t.meta.description,
    descriptionZh: t.meta.description_zh,
    project: result.project,
  };
});

mkdirSync(join(studioRoot, 'src/templates'), { recursive: true });
writeFileSync(join(studioRoot, 'src/templates/templates.json'), `${JSON.stringify(out, null, 2)}\n`);
console.log(`templates: wrote ${out.length} templates`);
