// How edges move (PRD 8.2): particles normally; dashes when many edges are in
// view or the canvas is zoomed out; still when the system asks for reduced
// motion. With one node selected, only its data lineage moves.
import type { Edge, ModuleDef, NodeId } from '@neuroloom/core';
import { create } from 'zustand';

export type FlowMode = 'particles' | 'dashes' | 'still';

/** Above this many edges in view, particles give way to dashes (PRD 8.2). */
export const MAX_PARTICLE_EDGES = 300;
/** Above this many, or further out than STILL_ZOOM, edges stand still: every moving dash repaints its edge. */
export const MAX_DASH_EDGES = 600;
/** Below this zoom, nodes draw simplified and edges dash (PRD 7.3). */
export const LOD_ZOOM = 0.4;
export const STILL_ZOOM = 0.25;

export const reducedMotion = typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

export function flowMode(visibleEdges: number, zoom: number): FlowMode {
  if (reducedMotion || visibleEdges > MAX_DASH_EDGES || zoom < STILL_ZOOM) return 'still';
  return visibleEdges > MAX_PARTICLE_EDGES || zoom < LOD_ZOOM ? 'dashes' : 'particles';
}

/** Edges on every path into and out of `nodeId`. */
export function lineage(mod: ModuleDef, nodeId: NodeId): Set<string> {
  const into = new Map<NodeId, Edge[]>();
  const out = new Map<NodeId, Edge[]>();
  for (const e of mod.edges) {
    (into.get(e.target.node) ?? into.set(e.target.node, []).get(e.target.node)!).push(e);
    (out.get(e.source.node) ?? out.set(e.source.node, []).get(e.source.node)!).push(e);
  }
  const edges = new Set<string>();
  const walk = (start: NodeId, next: Map<NodeId, Edge[]>, end: (e: Edge) => NodeId) => {
    const stack = [start];
    const seen = new Set<NodeId>([start]);
    while (stack.length > 0) {
      for (const e of next.get(stack.pop()!) ?? []) {
        edges.add(e.id);
        const n = end(e);
        if (!seen.has(n)) {
          seen.add(n);
          stack.push(n);
        }
      }
    }
  };
  walk(nodeId, into, (e) => e.source.node);
  walk(nodeId, out, (e) => e.target.node);
  return edges;
}

interface FlowState {
  mode: FlowMode;
  /** Edges of the selected node's lineage, or null when every edge moves. */
  lineage: Set<string> | null;
  set(patch: Partial<Pick<FlowState, 'mode' | 'lineage'>>): void;
}

export const useFlow = create<FlowState>()((set) => ({
  mode: reducedMotion ? 'still' : 'particles',
  lineage: null,
  set: (patch) => set(patch),
}));
