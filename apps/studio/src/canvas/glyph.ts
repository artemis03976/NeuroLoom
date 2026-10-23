import type { NodeSpec } from '@neuroloom/core';

const GLYPHS: Record<string, string> = {
  'neuroloom.Input': 'in',
  'neuroloom.Output': 'out',
  'torch.nn.Linear': 'FC',
  'torch.nn.ReLU': 'R',
  'torch.nn.functional.relu': 'ƒ',
};

/** Two-character badge shown on nodes and in the palette. */
export const glyphOf = (spec: NodeSpec): string => GLYPHS[spec.type] ?? spec.displayName.en.slice(0, 2);

export const categoryColor = (spec: NodeSpec | undefined): string => `var(--nl-cat-${spec?.category ?? 'code'})`;
