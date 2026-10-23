// Runs the project's Python (python/.venv, or $NEUROLOOM_PYTHON) inside python/.
// Usage: node scripts/py.mjs -m pytest -q
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const pythonDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'python');
const venv = join(pythonDir, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
const python = process.env.NEUROLOOM_PYTHON ?? (existsSync(venv) ? venv : 'python');

const { status } = spawnSync(python, process.argv.slice(2), { cwd: pythonDir, stdio: 'inherit' });
process.exit(status ?? 1);
