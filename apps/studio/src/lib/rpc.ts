import type { RpcFailure } from '@neuroloom/host-api';
import { tr } from '../i18n';

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
    throw new RpcError(tr('无法连接到开发服务器', 'Cannot reach the dev server'));
  }
  if (res.status === 404) throw new RpcError(tr('Python 后端不可用：当前页面不是由开发服务器提供的', 'The Python backend is unavailable: this page is not served by the dev server'));
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

/**
 * How the importer handled each class (PRD 11.2): L1 structured, L1+L3 with code
 * blocks, L2 structured from a run (expressions became numbers), L3 kept as source.
 */
export interface ImportReportEntry {
  className: string;
  level: 'L1' | 'L1+L3' | 'L2' | 'L3';
  codeBlocks: number;
  notes: string[];
}

export interface ImportResult {
  project: unknown;
  observedShapes: Record<string, number[][]>;
  batchProbe: number;
  warnings: string[];
  /** Static and dynamic imports. */
  report?: ImportReportEntry[];
}
