import type { RpcFailure } from '@neuroloom/host-api';

export class RpcError extends Error {
  constructor(
    message: string,
    readonly data?: unknown,
  ) {
    super(message);
  }
}

/**
 * Calls the Python backend: over IPC in the desktop app, or through the dev
 * server's /api/rpc bridge in the browser.
 */
export async function rpc<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
  if (window.neuroloom) {
    try {
      return (await window.neuroloom.rpc(method, params)) as T;
    } catch (e) {
      const failure = e as Partial<RpcFailure>;
      throw new RpcError(failure.message ?? String(e), failure.data);
    }
  }

  let res: Response;
  try {
    res = await fetch('/api/rpc', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ method, params }),
    });
  } catch {
    throw new RpcError('无法连接到开发服务器');
  }
  if (res.status === 404) throw new RpcError('Python 后端不可用：当前页面不是由开发服务器提供的');
  const body = (await res.json()) as { result?: T; error?: { message: string; data?: unknown } };
  if (body.error) throw new RpcError(body.error.message, body.error.data);
  return body.result as T;
}

export interface EnvInfo {
  backend: string;
  python: string;
  executable: string;
  torch: string | null;
  cuda: boolean;
}

export interface ImportResult {
  project: unknown;
  observedShapes: Record<string, number[][]>;
  batchProbe: number;
  warnings: string[];
}
