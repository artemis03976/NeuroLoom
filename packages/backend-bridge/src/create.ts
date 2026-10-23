// One-click environment creation: a venv with CPU PyTorch (PRD F10.1).
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { venvPython } from './discover';
import { run } from './run';

export const TORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu';
export const PYPI_INDEX = 'https://pypi.org/simple';
const PACKAGES = ['torch', 'numpy'];

export interface Step {
  label: string;
  command: string;
  args: string[];
}

/** The commands that create the environment; uv is used when available because it is much faster. */
export function planEnvironment(base: string, target: string, uv?: string): Step[] {
  const python = venvPython(target);
  const indexes = ['--index-url', TORCH_CPU_INDEX, '--extra-index-url', PYPI_INDEX];
  if (uv) {
    return [
      { label: '创建虚拟环境', command: uv, args: ['venv', target, '--python', base] },
      { label: '安装 PyTorch（CPU 版）', command: uv, args: ['pip', 'install', '--python', python, ...PACKAGES, ...indexes] },
    ];
  }
  return [
    { label: '创建虚拟环境', command: base, args: ['-m', 'venv', target] },
    { label: '升级 pip', command: python, args: ['-m', 'pip', 'install', '--upgrade', 'pip'] },
    { label: '安装 PyTorch（CPU 版）', command: python, args: ['-m', 'pip', 'install', ...PACKAGES, ...indexes] },
  ];
}

async function findUv(): Promise<string | undefined> {
  const r = await run('uv', ['--version'], { timeoutMs: 5_000 });
  return r.code === 0 ? 'uv' : undefined;
}

/** Creates the environment and returns its interpreter path. Throws with the failing step's output. */
export async function createEnvironment(options: { base: string; target: string; onLine?: (line: string) => void }): Promise<string> {
  mkdirSync(dirname(options.target), { recursive: true });
  const steps = planEnvironment(options.base, options.target, await findUv());
  for (const step of steps) {
    options.onLine?.(`▸ ${step.label}`);
    const r = await run(step.command, step.args, { onLine: options.onLine });
    if (r.code !== 0) {
      throw new Error(`${step.label}失败（退出码 ${r.code ?? '无'}）\n${(r.stderr || r.stdout).slice(-2000)}`);
    }
  }
  return venvPython(options.target);
}
