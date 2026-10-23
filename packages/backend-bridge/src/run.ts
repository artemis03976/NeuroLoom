import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';

export interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

/** Runs a command to completion. Never rejects: a missing binary is code null. */
export function run(
  command: string,
  args: string[],
  options: { timeoutMs?: number; cwd?: string; onLine?: (line: string) => void } = {},
): Promise<RunResult> {
  return new Promise((resolve) => {
    let stdout = '';
    let stderr = '';
    let settled = false;
    const done = (code: number | null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({ code, stdout, stderr });
    };
    let child: ReturnType<typeof spawn>;
    try {
      child = spawn(command, args, {
        cwd: options.cwd,
        windowsHide: true,
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      });
    } catch {
      return done(null);
    }
    const timer = options.timeoutMs
      ? setTimeout(() => {
          child.kill();
          done(null);
        }, options.timeoutMs)
      : undefined;
    child.stdout!.setEncoding('utf8');
    child.stderr!.setEncoding('utf8');
    child.stdout!.on('data', (chunk: string) => (stdout += chunk));
    child.stderr!.on('data', (chunk: string) => (stderr += chunk));
    if (options.onLine) {
      createInterface({ input: child.stdout! }).on('line', options.onLine);
      createInterface({ input: child.stderr! }).on('line', options.onLine);
    }
    child.on('error', () => done(null));
    child.on('close', (code) => done(code));
  });
}
