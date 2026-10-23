import { parseProject, type LoomProject } from '@neuroloom/core';
import raw from './templates.json';

export interface Template {
  id: string;
  title: string;
  titleZh: string;
  description: string;
  descriptionZh: string;
  project: LoomProject;
}

/** Simple to complex, the order they are shown in. */
const ORDER = ['mlp', 'lenet5', 'vgg_cnn', 'unet_tiny', 'lstm_classifier', 'transformer_classifier'];

/** Built-in templates (PRD F7.2), generated from apps/studio/templates/*.py by `npm run templates`. */
export const TEMPLATES: Template[] = (raw as unknown as (Omit<Template, 'project'> & { project: unknown })[])
  .map((t) => ({ ...t, project: parseProject(structuredClone(t.project)) }))
  .sort((a, b) => ORDER.indexOf(a.id) - ORDER.indexOf(b.id));
