import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterAll, describe, expect, it } from 'vitest';
import { PythonBackend, discoverPythons, planEnvironment, probePython, venvPython } from '../src/index.ts';

const pythonDir = join(dirname(fileURLToPath(import.meta.url)), '../../../python');
const projectVenv = join(pythonDir, '.venv');
const hasVenv = existsSync(venvPython(projectVenv));

describe('planEnvironment', () => {
  it('uses uv when available', () => {
    const steps = planEnvironment('python', '/tmp/env', 'uv');
    expect(steps.map((s) => s.command)).toEqual(['uv', 'uv']);
    expect(steps[1]!.args).toContain('https://download.pytorch.org/whl/cpu');
  });
  it('falls back to venv + pip', () => {
    const steps = planEnvironment('python', '/tmp/env');
    expect(steps[0]).toMatchObject({ command: 'python', args: ['-m', 'venv', '/tmp/env'] });
    expect(steps.at(-1)!.command).toBe(venvPython('/tmp/env'));
    expect(steps.at(-1)!.args).toEqual(expect.arrayContaining(['install', 'torch', 'numpy']));
  });
});

describe.skipIf(!hasVenv)('with the project venv', () => {
  const backend = new PythonBackend(venvPython(projectVenv), { cwd: pythonDir });
  afterAll(() => backend.dispose());

  it('probes an interpreter', { timeout: 60_000 }, async () => {
    const c = await probePython(venvPython(projectVenv), 'project');
    expect(c.status).toBe('ready');
    expect(c.torch).toMatch(/^2\./);
  });

  it('discovers it and ranks ready interpreters first', { timeout: 120_000 }, async () => {
    const found = await discoverPythons({ projectVenv });
    expect(found.some((c) => c.source === 'project' && c.status === 'ready')).toBe(true);
    const firstNotReady = found.findIndex((c) => c.status !== 'ready');
    if (firstNotReady >= 0) expect(found.slice(firstNotReady).every((c) => c.status !== 'ready')).toBe(true);
  });

  it('speaks JSON-RPC and can be restarted', { timeout: 60_000 }, async () => {
    expect(await backend.call('ping')).toMatchObject({ pong: true });
    backend.setInterpreter(venvPython(projectVenv) + ''); // same path: no restart
    const env = await backend.call<{ torch: string }>('env_info');
    expect(env.torch).toMatch(/^2\./);
    await expect(backend.call('no_such_method')).rejects.toMatchObject({ code: -32601 });
  });
});
