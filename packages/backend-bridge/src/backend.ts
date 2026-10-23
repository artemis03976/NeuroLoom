// JSON-RPC 2.0 client for `python -m neuroloom serve`, one message per line over stdio.
// Used by the Vite dev server and by the Electron main process.
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';

export interface RpcFailure {
  code: number;
  message: string;
  data?: unknown;
}

type Pending = { resolve: (value: unknown) => void; reject: (error: RpcFailure) => void };

export interface BackendOptions {
  /** Directory containing the `neuroloom` Python package. */
  cwd: string;
  /** Receives the backend's stderr, line by line. */
  onLog?: (line: string) => void;
}

export class PythonBackend {
  private proc?: ChildProcessWithoutNullStreams;
  private pending = new Map<number, Pending>();
  private nextId = 1;

  constructor(
    private python: string,
    private readonly options: BackendOptions,
  ) {}

  get interpreter(): string {
    return this.python;
  }

  /** Switches interpreter; the next call starts a fresh backend process. */
  setInterpreter(python: string): void {
    if (python === this.python) return;
    this.dispose();
    this.python = python;
  }

  private start(): ChildProcessWithoutNullStreams {
    if (this.proc) return this.proc;
    const proc = spawn(this.python, ['-m', 'neuroloom', 'serve'], {
      cwd: this.options.cwd,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      windowsHide: true,
    });
    createInterface({ input: proc.stderr }).on('line', (line) => this.options.onLog?.(line));
    createInterface({ input: proc.stdout }).on('line', (line) => {
      let msg: { id?: number; result?: unknown; error?: RpcFailure };
      try {
        msg = JSON.parse(line);
      } catch {
        return;
      }
      const waiter = msg.id === undefined ? undefined : this.pending.get(msg.id);
      if (!waiter) return;
      this.pending.delete(msg.id!);
      if (msg.error) waiter.reject(msg.error);
      else waiter.resolve(msg.result);
    });
    const fail = (message: string) => {
      if (this.proc !== proc) return;
      for (const waiter of this.pending.values()) waiter.reject({ code: -32000, message });
      this.pending.clear();
      this.proc = undefined;
    };
    proc.on('error', (err) => fail(`无法启动 Python 后端（${this.python}）：${err.message}`));
    proc.on('exit', (code) => fail(`Python 后端已退出（code ${code}）`));
    this.proc = proc;
    return proc;
  }

  call<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    const proc = this.start();
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      proc.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
    });
  }

  dispose(): void {
    const proc = this.proc;
    this.proc = undefined;
    for (const waiter of this.pending.values()) waiter.reject({ code: -32000, message: 'Python 后端已重启' });
    this.pending.clear();
    proc?.kill();
  }
}
