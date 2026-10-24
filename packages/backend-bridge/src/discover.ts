// Finds Python interpreters on this machine and checks which can run the backend (PRD F10.1).
import { existsSync, realpathSync } from 'node:fs';
import { join } from 'node:path';
import { run } from './run.ts';

export const MIN_PYTHON = [3, 10] as const;
export const MIN_TORCH = [2, 4] as const;

export type CandidateSource = 'env' | 'configured' | 'managed' | 'project' | 'launcher' | 'path' | 'conda';

export interface PythonCandidate {
  path: string;
  source: CandidateSource;
  version?: string;
  torch?: string | null;
  cuda?: boolean;
  status: 'ready' | 'no-torch' | 'old-torch' | 'old-python' | 'broken';
}

export function venvPython(dir: string): string {
  return process.platform === 'win32' ? join(dir, 'Scripts', 'python.exe') : join(dir, 'bin', 'python');
}

const PROBE = [
  'import json, platform, sys',
  'info = {"executable": sys.executable, "version": platform.python_version(), "torch": None, "cuda": False}',
  'try:',
  '    import torch',
  '    info["torch"] = torch.__version__',
  '    info["cuda"] = bool(torch.cuda.is_available())',
  'except Exception:',
  '    pass',
  'print(json.dumps(info))',
].join('\n');

const atLeast = (version: string, min: readonly [number, number]): boolean => {
  const [major = 0, minor = 0] = version.split(/[.+]/).map(Number);
  return major > min[0] || (major === min[0] && minor >= min[1]);
};

export async function probePython(path: string, source: CandidateSource): Promise<PythonCandidate> {
  const result = await run(path, ['-c', PROBE], { timeoutMs: 30_000 });
  const line = result.stdout.trim().split('\n').at(-1);
  let info: { executable: string; version: string; torch: string | null; cuda: boolean };
  try {
    info = JSON.parse(line ?? '');
  } catch {
    return { path, source, status: 'broken' };
  }
  const status = !atLeast(info.version, MIN_PYTHON)
    ? 'old-python'
    : !info.torch
      ? 'no-torch'
      : !atLeast(info.torch, MIN_TORCH)
        ? 'old-torch'
        : 'ready';
  return { path, source, version: info.version, torch: info.torch, cuda: info.cuda, status };
}

async function lines(command: string, args: string[]): Promise<string[]> {
  const r = await run(command, args, { timeoutMs: 10_000 });
  return r.code === 0 ? r.stdout.split(/\r?\n/).map((l) => l.trim()).filter(Boolean) : [];
}

async function launcherPythons(): Promise<string[]> {
  if (process.platform !== 'win32') return [];
  // `py -0p` lists installs as " -V:3.12 *        C:\Python312\python.exe".
  return (await lines('py', ['-0p'])).map((l) => l.replace(/^-V:\S+\s+\*?\s*/, '')).filter((p) => /python(w)?\.exe$/i.test(p));
}

async function pathPythons(): Promise<string[]> {
  const found =
    process.platform === 'win32' ? await lines('where', ['python']) : await lines('which', ['-a', 'python3', 'python']);
  // The Microsoft Store alias opens the Store instead of running Python.
  return found.filter((p) => !/WindowsApps/i.test(p));
}

async function condaPythons(): Promise<string[]> {
  const r = await run('conda', ['env', 'list', '--json'], { timeoutMs: 15_000 });
  if (r.code !== 0) return [];
  try {
    const envs: string[] = JSON.parse(r.stdout).envs ?? [];
    return envs.map((dir) => (process.platform === 'win32' ? join(dir, 'python.exe') : join(dir, 'bin', 'python')));
  } catch {
    return [];
  }
}

export interface DiscoverOptions {
  configured?: string;
  managedEnvDir?: string;
  projectVenv?: string;
}

const ORDER: CandidateSource[] = ['configured', 'env', 'managed', 'project', 'conda', 'launcher', 'path'];

/** All usable-looking interpreters, ready ones first, deduplicated by real path. */
export async function discoverPythons(options: DiscoverOptions = {}): Promise<PythonCandidate[]> {
  const listed: [string, CandidateSource][] = [];
  if (options.configured) listed.push([options.configured, 'configured']);
  if (process.env.NEUROLOOM_PYTHON) listed.push([process.env.NEUROLOOM_PYTHON, 'env']);
  if (options.managedEnvDir) listed.push([venvPython(options.managedEnvDir), 'managed']);
  if (options.projectVenv) listed.push([venvPython(options.projectVenv), 'project']);
  const [conda, launcher, path] = await Promise.all([condaPythons(), launcherPythons(), pathPythons()]);
  listed.push(...conda.map((p): [string, CandidateSource] => [p, 'conda']));
  listed.push(...launcher.map((p): [string, CandidateSource] => [p, 'launcher']));
  listed.push(...path.map((p): [string, CandidateSource] => [p, 'path']));

  const seen = new Set<string>();
  const unique: [string, CandidateSource][] = [];
  for (const [p, source] of listed) {
    if (!existsSync(p)) continue;
    let key: string;
    try {
      key = realpathSync(p);
    } catch {
      continue;
    }
    if (process.platform === 'win32') key = key.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push([p, source]);
  }

  const probed = await Promise.all(unique.map(([p, source]) => probePython(p, source)));
  const rank = (c: PythonCandidate) => (c.status === 'ready' ? 0 : 1) * 100 + ORDER.indexOf(c.source);
  return probed.sort((a, b) => rank(a) - rank(b));
}
