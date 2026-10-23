// Editor state. The IR project is the single source of truth (PRD 15.4); React
// Flow nodes are a view of it that additionally carries measured sizes and
// selection, and analysis + code are derived on every semantic change.
//
// Undo/redo keeps whole-project snapshots. Edits are immutable with structural
// sharing, so a snapshot costs little more than the parts that changed.
import {
  analyzeModule,
  addNode,
  connect,
  entryModule,
  generateCode,
  getSpec,
  outputsOf,
  pasteClip,
  removeEdges,
  removeNodes,
  renameNode,
  setLayout,
  setParam,
  type Analysis,
  type Clip,
  type CodegenResult,
  type LoomProject,
  type ModuleDef,
  type NodeId,
  type ParamValue,
} from '@neuroloom/core';
import { applyNodeChanges, type Connection, type EdgeChange, type Node, type NodeChange } from '@xyflow/react';
import { create } from 'zustand';
import { sampleProject } from './lib/sample';

export type FlowNode = Node<Record<string, never>, 'layer'>;

/** An existing port to wire a new node to: from a source port, or into a target port. */
export interface PortLink {
  nodeId: NodeId;
  port: string;
  side: 'source' | 'target';
}

const HISTORY_LIMIT = 200;
/** Consecutive edits to the same field within this window become one undo step. */
const MERGE_WINDOW_MS = 1200;

export interface DocState {
  /** File on disk (desktop) or last saved name (web); null for an unsaved document. */
  path: string | null;
  name: string;
  /** The project as last saved or opened; null when it has never been saved. */
  saved: LoomProject | null;
}

interface History {
  past: LoomProject[];
  future: LoomProject[];
  mergeKey?: string;
  mergeAt: number;
}

interface LayoutRequest {
  id: number;
  /** Whether the resulting move is an undoable edit (false for the layout that follows a load). */
  undoable: boolean;
}

interface StudioState {
  project: LoomProject;
  analysis: Analysis;
  codegen: CodegenResult;
  flowNodes: FlowNode[];
  selectedEdgeIds: string[];
  layoutRequest: LayoutRequest;
  doc: DocState;
  history: History;

  onNodesChange(changes: NodeChange<FlowNode>[]): void;
  onEdgesChange(changes: EdgeChange[]): void;
  onConnect(connection: Connection): void;
  /**
   * Adds a node; with `link`, also connects it to an existing port in the same
   * undo step (quick-add after dropping an edge on empty canvas). Returns its id.
   */
  addNodeAt(type: string, position: { x: number; y: number }, link?: PortLink): NodeId;
  setParam(nodeId: NodeId, name: string, value: ParamValue): void;
  rename(nodeId: NodeId, name: string): void;
  setClassName(name: string): void;
  selectNode(nodeId: NodeId | null): void;
  /** Pastes a clipboard fragment, offset from where it was copied, and selects it. */
  paste(clip: Clip, offset: { x: number; y: number }): void;
  deleteNodes(ids: NodeId[]): void;
  /** Replaces the document. `saved: false` marks it as never saved (e.g. imported code). */
  loadProject(project: LoomProject, doc?: { path?: string | null; name?: string; saved?: boolean; relayout?: boolean }): void;
  applyLayout(positions: Record<NodeId, { x: number; y: number }>, undoable: boolean): void;
  requestLayout(): void;
  /** Records that `project` (the snapshot that was written) is now on disk. */
  markSaved(project: LoomProject, path: string | null, name: string): void;
  undo(): void;
  redo(): void;
}

function derive(project: LoomProject): Pick<StudioState, 'analysis' | 'codegen'> {
  const mod = entryModule(project);
  const analysis = analyzeModule(mod);
  return { analysis, codegen: generateCode(project, { [mod.id]: analysis }) };
}

function withModule(project: LoomProject, update: (mod: ModuleDef) => ModuleDef): LoomProject {
  const mod = entryModule(project);
  return { ...project, modules: { ...project.modules, [mod.id]: update(mod) } };
}

function toFlowNodes(mod: ModuleDef, previous: FlowNode[] = []): FlowNode[] {
  const prev = new Map(previous.map((n) => [n.id, n]));
  return Object.values(mod.nodes).map((n) => ({
    ...prev.get(n.id),
    id: n.id,
    type: 'layer',
    position: mod.layout[n.id] ?? prev.get(n.id)?.position ?? { x: 0, y: 0 },
    data: {},
  }));
}

const initial = sampleProject();
const emptyHistory = (): History => ({ past: [], future: [], mergeAt: 0 });

export const useStudio = create<StudioState>()((set, get) => {
  /** Applies an edit as one undo step (or merges it into the previous one). */
  const commit = (project: LoomProject, options: { flowNodes?: FlowNode[]; merge?: string } = {}) => {
    const { project: before, history } = get();
    if (project === before) return;
    const now = Date.now();
    const merging = !!options.merge && history.mergeKey === options.merge && now - history.mergeAt < MERGE_WINDOW_MS;
    set({
      project,
      flowNodes: options.flowNodes ?? toFlowNodes(entryModule(project), get().flowNodes),
      ...derive(project),
      history: {
        past: merging ? history.past : [...history.past, before].slice(-HISTORY_LIMIT),
        future: [],
        mergeKey: options.merge,
        mergeAt: now,
      },
    });
  };

  /** Moves to a snapshot from the history, keeping selection and measurements. */
  const restore = (project: LoomProject, history: History) =>
    set({ project, history, flowNodes: toFlowNodes(entryModule(project), get().flowNodes), ...derive(project) });

  return {
    project: initial,
    ...derive(initial),
    flowNodes: toFlowNodes(entryModule(initial)),
    selectedEdgeIds: [],
    layoutRequest: { id: 0, undoable: false },
    doc: { path: null, name: initial.name, saved: initial },
    history: emptyHistory(),

    onNodesChange(changes) {
      const flowNodes = applyNodeChanges(changes, get().flowNodes);
      const removed = changes.flatMap((c) => (c.type === 'remove' ? [c.id] : []));
      // Positions are written to the IR when a drag ends, not on every frame.
      const moved: Record<NodeId, { x: number; y: number }> = {};
      for (const c of changes) if (c.type === 'position' && c.position && !c.dragging) moved[c.id] = c.position;

      if (removed.length === 0 && Object.keys(moved).length === 0) return set({ flowNodes });
      let project = get().project;
      if (Object.keys(moved).length > 0) project = withModule(project, (m) => setLayout(m, moved));
      if (removed.length > 0) project = withModule(project, (m) => removeNodes(m, removed));
      commit(project, { flowNodes });
    },

    onEdgesChange(changes) {
      const removed = changes.flatMap((c) => (c.type === 'remove' ? [c.id] : []));
      const selected = new Set(get().selectedEdgeIds);
      for (const c of changes) {
        if (c.type === 'select') {
          if (c.selected) selected.add(c.id);
          else selected.delete(c.id);
        }
      }
      for (const id of removed) selected.delete(id);
      set({ selectedEdgeIds: [...selected] });
      if (removed.length > 0) commit(withModule(get().project, (m) => removeEdges(m, removed)));
    },

    onConnect({ source, sourceHandle, target, targetHandle }) {
      const mod = connect(
        entryModule(get().project),
        { node: source, port: sourceHandle ?? 'out' },
        { node: target, port: targetHandle ?? 'in' },
      );
      if (mod) commit(withModule(get().project, () => mod));
    },

    addNodeAt(type, position, link) {
      let nodeId = '';
      const project = withModule(get().project, (m) => {
        const result = addNode(m, type, position);
        nodeId = result.nodeId;
        const spec = getSpec(type);
        if (!link || !spec) return result.mod;
        const node = result.mod.nodes[nodeId]!;
        const linked =
          link.side === 'source'
            ? spec.inputs[0] &&
              connect(result.mod, { node: link.nodeId, port: link.port }, { node: nodeId, port: spec.inputs[0].variadic ? `${spec.inputs[0].name}0` : spec.inputs[0].name })
            : outputsOf(spec, node.params)[0] &&
              connect(result.mod, { node: nodeId, port: outputsOf(spec, node.params)[0]!.name }, { node: link.nodeId, port: link.port });
        return linked || result.mod;
      });
      const flowNodes = toFlowNodes(entryModule(project), get().flowNodes).map((n) => ({ ...n, selected: n.id === nodeId }));
      commit(project, { flowNodes });
      return nodeId;
    },

    setParam(nodeId, name, value) {
      commit(withModule(get().project, (m) => setParam(m, nodeId, name, value)), { merge: `param:${nodeId}:${name}` });
    },

    rename(nodeId, name) {
      commit(withModule(get().project, (m) => renameNode(m, nodeId, name)), { merge: `rename:${nodeId}` });
    },

    setClassName(className) {
      commit(withModule(get().project, (m) => ({ ...m, className })), { merge: 'className' });
    },

    selectNode(nodeId) {
      set({ flowNodes: get().flowNodes.map((n) => ({ ...n, selected: n.id === nodeId })) });
    },

    paste(clip, offset) {
      let pasted: NodeId[] = [];
      const project = withModule(get().project, (m) => {
        const result = pasteClip(m, clip, offset);
        pasted = result.nodeIds;
        return result.mod;
      });
      const selected = new Set(pasted);
      const flowNodes = toFlowNodes(entryModule(project), get().flowNodes).map((n) => ({ ...n, selected: selected.has(n.id) }));
      commit(project, { flowNodes });
    },

    deleteNodes(ids) {
      if (ids.length > 0) commit(withModule(get().project, (m) => removeNodes(m, ids)));
    },

    loadProject(project, doc = {}) {
      const mod = entryModule(project);
      const relayout = doc.relayout ?? Object.keys(mod.layout).length === 0;
      set({
        project,
        ...derive(project),
        flowNodes: toFlowNodes(mod),
        selectedEdgeIds: [],
        history: emptyHistory(),
        doc: { path: doc.path ?? null, name: doc.name ?? project.name, saved: doc.saved === false ? null : project },
        layoutRequest: relayout ? { id: get().layoutRequest.id + 1, undoable: false } : get().layoutRequest,
      });
    },

    applyLayout(positions, undoable) {
      const project = withModule(get().project, (m) => setLayout(m, positions));
      if (undoable) return commit(project);
      // The layout that follows a load is part of loading: no undo step, and a
      // clean document stays clean.
      const { doc, project: before } = get();
      set({
        project,
        flowNodes: toFlowNodes(entryModule(project), get().flowNodes),
        ...derive(project),
        doc: doc.saved === before ? { ...doc, saved: project } : doc,
      });
    },

    requestLayout() {
      set({ layoutRequest: { id: get().layoutRequest.id + 1, undoable: true } });
    },

    markSaved(project, path, name) {
      set({ doc: { path, name, saved: project } });
    },

    undo() {
      const { history, project } = get();
      const previous = history.past.at(-1);
      if (!previous) return;
      restore(previous, { past: history.past.slice(0, -1), future: [project, ...history.future], mergeAt: 0 });
    },

    redo() {
      const { history, project } = get();
      const next = history.future[0];
      if (!next) return;
      restore(next, { past: [...history.past, project], future: history.future.slice(1), mergeAt: 0 });
    },
  };
});

export const useEntryModule = () => useStudio((s) => entryModule(s.project));
export const useSelectedNodeId = () => useStudio((s) => s.flowNodes.find((n) => n.selected)?.id ?? null);
export const useDirty = () => useStudio((s) => s.doc.saved !== s.project);
