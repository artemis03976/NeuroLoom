// Forward replay (PRD F3.5): the module on the canvas runs step by step in
// topological order. Nodes that run together (parallel branches) form one step;
// the current step glows, its input edges flow and the shapes of what it produced
// appear. Module nodes are one step each, or, with "enter modules", play inside.
import type { ModuleDef, ModuleId, NodeId } from '@neuroloom/core';
import { create } from 'zustand';
import { useStudio } from '../store';

export type ReplayStatus = 'idle' | 'pending' | 'active' | 'done';

/** Steps of a module: nodes grouped by their longest distance from an input; module nodes get a step of their own when entered. */
export function replaySteps(mod: ModuleDef, splitModules: boolean): NodeId[][] {
  const preds = new Map<NodeId, NodeId[]>();
  for (const e of mod.edges) (preds.get(e.target.node) ?? preds.set(e.target.node, []).get(e.target.node)!).push(e.source.node);
  const level = new Map<NodeId, number>();
  const visiting = new Set<NodeId>();
  const depth = (id: NodeId): number => {
    if (level.has(id)) return level.get(id)!;
    if (visiting.has(id)) return 0; // a cycle (an invalid graph): cut it here
    visiting.add(id);
    const d = Math.max(-1, ...(preds.get(id) ?? []).map(depth)) + 1;
    visiting.delete(id);
    level.set(id, d);
    return d;
  };
  const groups: NodeId[][] = [];
  for (const id of Object.keys(mod.nodes)) (groups[depth(id)] ??= []).push(id);
  const steps: NodeId[][] = [];
  for (const group of groups.filter(Boolean)) {
    const sorted = group.sort((a, b) => (mod.layout[a]?.y ?? 0) - (mod.layout[b]?.y ?? 0));
    const modules = splitModules ? sorted.filter((id) => mod.nodes[id]!.kind === 'module') : [];
    const rest = sorted.filter((id) => !modules.includes(id));
    if (rest.length > 0) steps.push(rest);
    for (const id of modules) steps.push([id]);
  }
  return steps;
}

interface Frame {
  moduleId: ModuleId;
  depth: number;
  steps: NodeId[][];
  index: number;
}

interface ReplayState {
  open: boolean;
  playing: boolean;
  /** 0.5× to 4×. */
  speed: number;
  /** Play inside module nodes instead of one step each. */
  enter: boolean;
  moduleId: ModuleId | null;
  steps: NodeId[][];
  /** Step being shown; -1 before the first, steps.length once finished. */
  index: number;
  /** step index of every node, for status lookups. */
  stepOf: Record<NodeId, number>;
  /** The modules we came from when playing inside a module node. */
  stack: Frame[];

  start(): void;
  close(): void;
  setPlaying(playing: boolean): void;
  setSpeed(speed: number): void;
  setEnter(enter: boolean): void;
  restart(): void;
  /** One step forward; false when the replay has finished. */
  advance(): boolean;
  /** The module was edited: recomputes its steps, keeping the position. */
  refresh(): void;
}

const stepIndex = (steps: NodeId[][]) => Object.fromEntries(steps.flatMap((ids, i) => ids.map((id) => [id, i])));

/** Steps of the module on the canvas now. */
function current(enter: boolean) {
  const { project, view } = useStudio.getState();
  const steps = replaySteps(project.modules[view.moduleId]!, enter);
  return { moduleId: view.moduleId, steps, stepOf: stepIndex(steps) };
}

export const useReplay = create<ReplayState>()((set, get) => ({
  open: false,
  playing: false,
  speed: 1,
  enter: false,
  moduleId: null,
  steps: [],
  index: -1,
  stepOf: {},
  stack: [],

  start() {
    set({ open: true, playing: true, index: -1, stack: [], ...current(get().enter) });
  },
  close() {
    set({ open: false, playing: false, moduleId: null, steps: [], stepOf: {}, index: -1, stack: [] });
  },
  setPlaying(playing) {
    if (playing && get().index >= get().steps.length) get().restart();
    set({ playing });
  },
  setSpeed(speed) {
    set({ speed });
  },
  setEnter(enter) {
    set({ enter, index: -1, stack: [], ...current(enter) });
  },
  restart() {
    const { stack } = get();
    // Back to where the replay started.
    if (stack.length > 0) useStudio.getState().goToDepth(stack[0]!.depth);
    set({ index: -1, stack: [], ...current(get().enter) });
  },
  advance() {
    const { index, steps, enter, stack, moduleId } = get();
    const studio = useStudio.getState();
    if (index >= steps.length - 1) {
      const outer = stack.at(-1);
      if (!outer) {
        set({ index: steps.length, playing: false });
        return false;
      }
      // Done inside a module: back out to its node, which is now complete.
      studio.goToDepth(outer.depth);
      set({ moduleId: outer.moduleId, steps: outer.steps, stepOf: stepIndex(outer.steps), index: outer.index, stack: stack.slice(0, -1) });
      return true;
    }
    const next = index + 1;
    const nodeId = steps[next]!.length === 1 ? steps[next]![0]! : undefined;
    const node = nodeId && moduleId ? studio.project.modules[moduleId]?.nodes[nodeId] : undefined;
    const def = node?.kind === 'module' ? studio.project.modules[node.type] : undefined;
    if (enter && node && def && !def.opaque && studio.view.moduleId === moduleId) {
      const frame: Frame = { moduleId: moduleId!, depth: studio.view.path.length, steps, index: next };
      studio.enterModule(node.id);
      set({ stack: [...stack, frame], index: -1, ...current(true) });
      return true;
    }
    set({ index: next });
    return true;
  },
  refresh() {
    const { open, moduleId, enter, index } = get();
    if (!open || moduleId !== useStudio.getState().view.moduleId) return;
    const fresh = current(enter);
    set({ ...fresh, index: Math.min(index, fresh.steps.length) });
  },
}));

/** A node's place in the replay; 'idle' when no replay runs in this module. */
export function useReplayStatus(nodeId: NodeId): ReplayStatus {
  const viewModule = useStudio((s) => s.view.moduleId);
  return useReplay((s) => {
    if (!s.open || s.moduleId !== viewModule) return 'idle';
    const i = s.stepOf[nodeId];
    if (i === undefined) return 'idle';
    return i < s.index ? 'done' : i === s.index ? 'active' : 'pending';
  });
}

/**
 * An edge in the replay: 'waiting' before its source ran (faded, no shape),
 * 'flowing' while its target runs, 'ready' once its source ran ('done' when the
 * target has too); 'idle' without a replay.
 */
export type EdgeReplay = 'idle' | 'waiting' | 'flowing' | 'ready' | 'done';

export function useEdgeReplay(source: NodeId, target: NodeId): EdgeReplay {
  const viewModule = useStudio((s) => s.view.moduleId);
  return useReplay((s) => {
    if (!s.open || s.moduleId !== viewModule) return 'idle';
    const from = s.stepOf[source];
    const to = s.stepOf[target];
    if (from === undefined || to === undefined) return 'idle';
    if (to === s.index) return 'flowing';
    if (from > s.index) return 'waiting';
    return to < s.index ? 'done' : 'ready';
  });
}
