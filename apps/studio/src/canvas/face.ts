// What every node shows on the canvas (PRD 7.3), as plain data. Computed for the
// whole module after each edit; a face that did not change keeps its previous
// object, so only the nodes that changed render again (PRD 13: 1000 nodes).
import {
  formatShape,
  getSpec,
  literalOf,
  portsOf,
  variadicIndex,
  withDefaults,
  type Analysis,
  type Literal,
  type LoomProject,
  type ModuleDef,
  type NodeId,
  type NodeInst,
} from '@neuroloom/core';
import { tr } from '../i18n';
import { nodeBadge } from './glyph';

export interface HandleInfo {
  id: string;
  label: string;
  optional?: boolean;
}

export interface NodeFace {
  kind: NodeInst['kind'];
  name: string;
  glyph: string;
  /** PyTorch class, module class or "Python". */
  type: string;
  /** Category colour (a CSS variable). */
  color: string;
  /** The type is not known: drawn as an error box. */
  unknown?: boolean;
  inputs: HandleInfo[];
  outputs: HandleInfo[];
  /** Ports are written on rows: several inputs or outputs, or a code block. */
  labelled: boolean;
  /** Key parameters (64→128 · k3 · s2), a shape, or a code block's first line. */
  summary?: string;
  params: number;
  error?: string;
  repeat: boolean;
  /** Repeats: how many copies, when known. */
  copies?: number;
}

/** One node's face; `targetPorts` are the ports its incoming edges use (variadic inputs grow past the last one). */
export function nodeFace(project: LoomProject, node: NodeInst, analysis: Analysis, targetPorts: string[]): NodeFace {
  const info = analysis.nodes[node.id];
  const badge = nodeBadge(project, node);
  const ports = portsOf(project, node);
  const base = { kind: node.kind, name: node.name, glyph: badge.glyph, type: badge.type, color: badge.color, repeat: !!node.repeat, params: info?.paramCount ?? 0 };
  if (!ports) return { ...base, unknown: true, inputs: [], outputs: [], labelled: false, params: 0 };

  const inputs: HandleInfo[] = ports.inputs.flatMap((p) => {
    if (!p.variadic) return [{ id: p.name, label: p.label, optional: p.optional }];
    const used = targetPorts.filter((port) => variadicIndex(port, p.name) !== undefined).length;
    return Array.from({ length: used + 1 }, (_, i) => ({ id: `${p.name}${i}`, label: `${p.label}${i}`, optional: i === used }));
  });
  const outputs: HandleInfo[] = ports.outputs.map((o) => ({ id: o.name, label: o.label }));

  let summary: string | undefined;
  if (node.kind === 'output') {
    summary = info?.inputs[0] ? formatShape(info.inputs[0].shape) : undefined;
  } else if (node.kind === 'input' && info?.outputs?.[0]) {
    // In a submodule the type comes from the call site, not the declared shape.
    const t = info.outputs[0];
    summary = `${formatShape(t.shape)}${t.dtype !== 'float32' ? ` ${t.dtype}` : ''}`;
  } else if (node.kind === 'module') {
    const values = Object.entries(info?.resolved ?? {}).filter(([k]) => !k.startsWith('#'));
    summary = values.map(([k, v]) => `${k}=${Array.isArray(v) ? v.join('×') : String(v)}`).join(' ');
  } else if (node.kind === 'code') {
    summary = node.code?.source.split('\n').find((l) => l.trim())?.trim();
  } else {
    // Resolved values when inference succeeded, otherwise what the user typed.
    const spec = getSpec(node.type)!;
    const params = withDefaults(spec, node.params);
    const typed = Object.fromEntries(Object.entries(params).map(([k, v]) => [k, literalOf(v) ?? ('auto' as Literal)]));
    summary = spec.describe?.({ ...typed, ...info?.resolved });
  }
  const copies = node.repeat ? (info?.children?.length ?? (node.kind === 'layer' ? Number(literalOf(node.repeat.count)) || undefined : undefined)) : undefined;
  const error = analysis.diagnostics.find((d) => d.nodeId === node.id && d.severity === 'error')?.message;
  return {
    ...base,
    inputs,
    outputs,
    labelled: inputs.length > 1 || outputs.length > 1 || node.kind === 'code',
    summary: summary || undefined,
    error,
    copies,
  };
}

/** Faces of every node in `mod`; those equal to `previous` keep the previous object. */
export function nodeFaces(project: LoomProject, mod: ModuleDef, analysis: Analysis, previous: Record<NodeId, NodeFace> = {}): Record<NodeId, NodeFace> {
  const targets = new Map<NodeId, string[]>();
  for (const e of mod.edges) {
    const list = targets.get(e.target.node);
    if (list) list.push(e.target.port);
    else targets.set(e.target.node, [e.target.port]);
  }
  const faces: Record<NodeId, NodeFace> = {};
  for (const node of Object.values(mod.nodes)) {
    const face = nodeFace(project, node, analysis, targets.get(node.id) ?? []);
    const before = previous[node.id];
    faces[node.id] = before && JSON.stringify(before) === JSON.stringify(face) ? before : face;
  }
  return faces;
}

/** Text for nodes of unknown type. */
export const unknownLabel = () => tr('未知节点', 'Unknown node');
