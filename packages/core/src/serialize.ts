// .loom files (PRD 12.1): UTF-8 JSON with sorted keys and 2-space indentation,
// so the same project always serializes to the same bytes and diffs cleanly in git.
import type { LoomProject } from './ir';
import { parseProject } from './validate';

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (typeof value !== 'object' || value === null) return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((k) => [k, sortKeys((value as Record<string, unknown>)[k])]),
  );
}

export function serializeProject(project: LoomProject): string {
  return `${JSON.stringify(sortKeys(project), null, 2)}\n`;
}

export function parseProjectText(text: string): LoomProject {
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (e) {
    throw new Error(`文件不是有效的 JSON：${(e as Error).message}`);
  }
  return parseProject(json);
}

export const LOOM_EXTENSION = '.loom';
