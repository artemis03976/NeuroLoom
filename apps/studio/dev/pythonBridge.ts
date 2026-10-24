// Dev-server bridge to the Python backend: POST /api/rpc {method, params} is
// relayed to the shared PythonBackend. The Electron app uses IPC instead.
// Relative import on purpose: Vite's config loader externalizes workspace packages.
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';
import { PythonBackend, venvPython, type RpcFailure } from '../../../packages/backend-bridge/src/index.ts';

export function pythonBridge(): Plugin {
  return {
    name: 'neuroloom-python-bridge',
    apply: 'serve',
    configureServer(server) {
      const pythonDir = join(dirname(fileURLToPath(import.meta.url)), '../../../python');
      const projectPython = venvPython(join(pythonDir, '.venv'));
      const python = process.env.NEUROLOOM_PYTHON ?? (existsSync(projectPython) ? projectPython : 'python');
      const backend = new PythonBackend(python, { cwd: pythonDir, onLog: (line) => console.error(`[python] ${line}`) });
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
