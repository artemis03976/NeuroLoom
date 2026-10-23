// Dev-server bridge to the Python backend: POST /api/rpc {method, params} is
// relayed as JSON-RPC 2.0 over the backend's stdio. In M1 the Electron main
// process takes over this role with the same PythonBackend class.
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';

interface RpcFailure {
  code: number;
  message: string;
  data?: unknown;
}

type Pending = { resolve: (value: unknown) => void; reject: (error: RpcFailure) => void };

export class PythonBackend {
  private proc?: ChildProcessWithoutNullStreams;
  private pending = new Map<number, Pending>();
  private nextId = 1;

  constructor(
    private readonly python: string,
    private readonly cwd: string,
  ) {}

  private start(): ChildProcessWithoutNullStreams {
    if (this.proc) return this.proc;
    const proc = spawn(this.python, ['-m', 'neuroloom', 'serve'], {
      cwd: this.cwd,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    });
    proc.stderr.on('data', (chunk: Buffer) => process.stderr.write(`[python] ${chunk.toString('utf8')}`));
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
      for (const waiter of this.pending.values()) waiter.reject({ code: -32000, message });
      this.pending.clear();
      this.proc = undefined;
    };
    proc.on('error', (err) => fail(`无法启动 Python 后端（${this.python}）：${err.message}`));
    proc.on('exit', (code) => fail(`Python 后端已退出（code ${code}）`));
    this.proc = proc;
    return proc;
  }

  call(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    const proc = this.start();
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      proc.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
    });
  }

  dispose(): void {
    this.proc?.kill();
    this.proc = undefined;
  }
}

export function resolvePython(pythonDir: string): string {
  if (process.env.NEUROLOOM_PYTHON) return process.env.NEUROLOOM_PYTHON;
  const venv = join(pythonDir, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
  return existsSync(venv) ? venv : 'python';
}

export function pythonBridge(): Plugin {
  return {
    name: 'neuroloom-python-bridge',
    apply: 'serve',
    configureServer(server) {
      const pythonDir = join(dirname(fileURLToPath(import.meta.url)), '../../../python');
      const backend = new PythonBackend(resolvePython(pythonDir), pythonDir);
      server.httpServer?.on('close', () => backend.dispose());

      server.middlewares.use('/api/rpc', (req, res) => {
        const send = (status: number, body: unknown) => {
          res.statusCode = status;
          res.setHeader('content-type', 'application/json; charset=utf-8');
          res.end(JSON.stringify(body));
        };
        if (req.method !== 'POST') return send(405, { error: { code: -32600, message: 'POST only' } });
        let raw = '';
        req.setEncoding('utf8');
        req.on('data', (chunk: string) => (raw += chunk));
        req.on('end', () => {
          let request: { method?: string; params?: Record<string, unknown> };
          try {
            request = JSON.parse(raw);
          } catch {
            return send(400, { error: { code: -32700, message: 'invalid JSON' } });
          }
          backend
            .call(String(request.method), request.params ?? {})
            .then((result) => send(200, { result }))
            .catch((error: RpcFailure) => send(200, { error }));
        });
      });
    },
  };
}
