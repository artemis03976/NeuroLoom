// The module on the canvas as plain data for exportImage.ts: what each node shows
// (the same face as canvas/LayerNode.tsx) and where React Flow measured it.
import { formatShape, type Analysis, type LoomProject, type ModuleId, type NodeInst } from '@neuroloom/core';
import { Position } from '@xyflow/react';
import { nodeFace, unknownLabel } from '../canvas/face';
import { HEAD_BOTTOM, PORT_ROW, type Scene, type SceneHandle, type SceneNode } from './exportImage';
import { formatCount } from './format';

/** The parts of a React Flow InternalNode the export reads. */
export interface MeasuredNode {
  measured?: { width?: number; height?: number };
  internals: {
    positionAbsolute: { x: number; y: number };
    handleBounds?: {
      source: { id?: string | null; type: 'source' | 'target'; position: Position; x: number; y: number; width: number; height: number }[] | null;
      target: { id?: string | null; type: 'source' | 'target'; position: Position; x: number; y: number; width: number; height: number }[] | null;
    };
  };
}

/** Sizes React Flow falls back to before a node is measured (FlowCanvas uses the same). */
const DEFAULT_SIZE = { width: 180, height: 72 };
const HANDLE = 10;

interface PortInfo {
  id: string;
  label: string;
  optional?: boolean;
}

type NodeContent = Omit<SceneNode, 'id' | 'x' | 'y' | 'width' | 'height' | 'handles'> & { inputs: PortInfo[]; outputs: PortInfo[] };

/** What a node shows on the canvas: its face (canvas/face.ts), as the export draws it. */
export function nodeContent(project: LoomProject, moduleId: ModuleId, node: NodeInst, analysis: Analysis): NodeContent {
  const mod = project.modules[moduleId]!;
  const face = nodeFace(
    project,
    node,
    analysis,
    mod.edges.filter((e) => e.target.node === node.id).map((e) => e.target.port),
  );
  const variant = node.kind === 'module' ? 'module' : node.kind === 'code' ? 'code' : 'layer';
  if (face.unknown) {
    return { category: 'code', variant, glyph: '?', name: unknownLabel(), type: node.type, error: node.type, inputs: [], outputs: [] };
  }
  const { inputs, outputs } = face;
  const rows = face.labelled
    ? Array.from({ length: Math.max(inputs.length, outputs.length) }, (_, i) => ({
        input: inputs[i] && { label: inputs[i].label, handle: inputs[i].id, optional: inputs[i].optional },
        output: outputs[i] && { label: outputs[i].label, handle: outputs[i].id },
      }))
    : undefined;
  return {
    category: /--nl-cat-([a-z]+)/.exec(face.color)?.[1] ?? 'code',
    variant,
    glyph: face.glyph,
    name: face.name,
    type: face.type,
    repeat: face.repeat ? `×${face.copies ?? '?'}` : undefined,
    rows,
    summary: face.summary,
    params: face.params > 0 ? formatCount(face.params) : undefined,
    error: face.error,
    inputs,
    outputs,
  };
}

/** Handles where LayerNode puts them, for a node React Flow has not measured. */
function fallbackHandles(content: NodeContent, width: number, height: number): SceneHandle[] {
  const labelled = !!content.rows;
  const at = (i: number) => (labelled ? HEAD_BOTTOM + PORT_ROW * i + PORT_ROW / 2 : height / 2) - HANDLE / 2;
  return [
    ...content.inputs.map((p, i) => ({ id: p.id, type: 'target' as const, position: Position.Left, x: -HANDLE / 2, y: at(i), width: HANDLE, height: HANDLE })),
    ...content.outputs.map((p, i) => ({ id: p.id, type: 'source' as const, position: Position.Right, x: width - HANDLE / 2, y: at(i), width: HANDLE, height: HANDLE })),
  ];
}

/**
 * The module on the canvas, or only the nodes in `only` and the edges between
 * them. `getNode` is React Flow's getInternalNode.
 */
export function sceneOf(args: {
  project: LoomProject;
  moduleId: ModuleId;
  analysis: Analysis;
  getNode: (id: string) => MeasuredNode | undefined;
  only?: Iterable<string>;
}): Scene {
  const { project, moduleId, analysis, getNode } = args;
  const mod = project.modules[moduleId]!;
  const only = args.only ? new Set(args.only) : undefined;
  const nodes: SceneNode[] = [];
  for (const node of Object.values(mod.nodes)) {
    if (only && !only.has(node.id)) continue;
    const measured = getNode(node.id);
    const content = nodeContent(project, moduleId, node, analysis);
    const position = measured?.internals.positionAbsolute ?? mod.layout[node.id] ?? { x: 0, y: 0 };
    const width = measured?.measured?.width ?? DEFAULT_SIZE.width;
    const height = measured?.measured?.height ?? DEFAULT_SIZE.height;
    const bounds = measured?.internals.handleBounds;
    const optional = new Set(content.rows ? content.inputs.filter((p) => p.optional).map((p) => p.id) : []);
    const handles: SceneHandle[] = bounds
      ? [...(bounds.target ?? []), ...(bounds.source ?? [])].map((h) => ({
          id: h.id ?? '',
          type: h.type,
          position: h.position,
          x: h.x,
          y: h.y,
          width: h.width,
          height: h.height,
          optional: h.type === 'target' && optional.has(h.id ?? '') ? true : undefined,
        }))
      : fallbackHandles(content, width, height).map((h) => ({ ...h, optional: optional.has(h.id) || undefined }));
    const { inputs: _inputs, outputs: _outputs, ...shown } = content;
    nodes.push({ id: node.id, x: position.x, y: position.y, width, height, handles, ...shown });
  }
  const kept = new Set(nodes.map((n) => n.id));
  const edges = mod.edges
    .filter((e) => kept.has(e.source.node) && kept.has(e.target.node))
    .map((e) => {
      const shape = analysis.edgeTypes[e.id]?.shape;
      return {
        id: e.id,
        source: e.source.node,
        sourceHandle: e.source.port,
        target: e.target.node,
        targetHandle: e.target.port,
        shape: shape !== undefined ? formatShape(shape) : undefined,
      };
    });
  return { title: mod.className, nodes, edges };
}
