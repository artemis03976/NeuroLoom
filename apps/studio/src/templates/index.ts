import { parseProject, type LoomProject } from '@neuroloom/core';
import rawBlocks from './blocks.json';
import rawTemplates from './templates.json';

export interface Template {
  id: string;
  title: string;
  titleZh: string;
  description: string;
  descriptionZh: string;
  /** Link to the original paper. */
  paper: string;
  project: LoomProject;
}

/** A reusable module (PRD 9.6); `project.entry` is the block's class, other modules are the helpers it uses. */
export type Block = Omit<Template, 'paper'>;

/** Simple to complex, the order they are shown in. */
const ORDER = [
  'mlp',
  'lenet5',
  'vae',
  'vgg11',
  'dcgan',
  'resnet18',
  'resnet50',
  'unet',
  'lstm_classifier',
  'transformer_classifier',
  'vit_tiny',
  'gpt2_small',
];

const BLOCK_ORDER = ['residual_block', 'bottleneck', 'se_block', 'multi_head_attention', 'transformer_block', 'unet_down', 'unet_up'];

function load<T extends { id: string; project: LoomProject }>(raw: unknown, order: string[]): T[] {
  return (raw as (Omit<T, 'project'> & { project: unknown })[])
    .map((t) => ({ ...t, project: parseProject(structuredClone(t.project)) }) as T)
    .sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
}

/** Built-in templates (PRD F7.2, 12.2), generated from apps/studio/templates/*.py by `npm run templates`. */
export const TEMPLATES: Template[] = load<Template>(rawTemplates, ORDER);

/** Built-in composite modules (PRD 9.6), generated from apps/studio/blocks/*.py by `npm run templates`. */
export const BLOCKS: Block[] = load<Block>(rawBlocks, BLOCK_ORDER);
