// Helpers for tests that run the real Python backend.
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { entryModule, getSpec, literalOf, outputsOf, withDefaults, type Analysis, type LoomProject } from '../src';

export const pythonDir = join(dirname(fileURLToPath(import.meta.url)), '../../../python');
const venvPython = join(pythonDir, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
export const python = process.env.NEUROLOOM_PYTHON ?? (existsSync(venvPython) ? venvPython : 'python');
export const hasTorch = spawnSync(python, ['-c', 'import torch'], { encoding: 'utf8' }).status === 0;

export function runPython(args: string[], input: string): string {
  const proc = spawnSync(python, args, {
    cwd: pythonDir,
    input,
    encoding: 'utf8',
    maxBuffer: 256 * 1024 * 1024,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });
  if (proc.error) throw proc.error;
  return proc.stdout.trim().split('\n').at(-1) ?? '';
}

export interface ImportResult {
  project: unknown;
  observedShapes: Record<string, Record<string, number[]>>;
  batchProbe: number;
  error?: { message: string };
}

/** "B, 16" + int64 -> "B,16:int64", the CLI's input format. */
export const inputArg = (shape: string, dtype?: string) => `${shape.replace(/\s/g, '')}${dtype && dtype !== 'float32' ? `:${dtype}` : ''}`;

export const inputShapesOf = (p: LoomProject): Record<string, string> =>
  Object.fromEntries(
    Object.values(entryModule(p).nodes)
      .filter((n) => n.kind === 'input')
      .map((n) => [n.name, inputArg(String(literalOf(n.params.shape)), literalOf(n.params.dtype) as string | undefined)]),
  );

export function importCode(code: string, inputShapes: Record<string, string> = {}): ImportResult {
  const args = ['-m', 'neuroloom', 'import', '-'];
  for (const [name, spec] of Object.entries(inputShapes)) args.push('--input', `${name}=${spec}`);
  const out = JSON.parse(runPython(args, code));
  if (out.error) throw new Error(out.error.message);
  return out;
}

export function importMany(requests: { source: string; inputShapes: Record<string, string> }[]): ImportResult[] {
  return JSON.parse(runPython(['-m', 'neuroloom', 'import-batch'], JSON.stringify(requests)));
}

/** Every shape PyTorch observed, next to what the TypeScript inference says for the same port. */
export function shapePairs(p: LoomProject, analysis: Analysis, result: ImportResult): { label: string; ts: number[] | undefined; torch: number[] }[] {
  const mod = entryModule(p);
  const pairs = [];
  for (const [nodeId, ports] of Object.entries(result.observedShapes)) {
    const node = mod.nodes[nodeId]!;
    const spec = getSpec(node.type)!;
    const outs = outputsOf(spec, withDefaults(spec, node.params));
    for (const [port, shape] of Object.entries(ports)) {
      const index = outs.findIndex((o) => o.name === port);
      const ts = analysis.nodes[nodeId]?.outputs?.[index]?.shape;
      pairs.push({
        label: `${node.name}.${port}`,
        ts: ts?.map((d) => (typeof d === 'number' ? d : evalSymbols(d, result.batchProbe))),
        torch: shape,
      });
    }
  }
  return pairs;
}

function evalSymbols(d: string, value: number): number {
  return d.split('+').reduce((sum, term) => sum + term.split('*').reduce((p, f) => p * (/^\d+$/.test(f) ? Number(f) : value), 1), 0);
}
