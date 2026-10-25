// Editor state. The IR project is the single source of truth (PRD 15.4); React
// Flow nodes are a view of the module on the canvas that additionally carries
// measured sizes and selection, and analysis + code are derived on every
// semantic change.
//
// Undo/redo keeps whole-project snapshots. Edits are immutable with structural
// sharing, so a snapshot costs little more than the parts that changed.
import {
  CODE_TYPE,
  addCodeNode,
  addModuleNode,
  addNode,
  analysisAt,
  analyzeProject,
  connect,
  createModule as createModuleDef,
  deleteModule as deleteModuleDef,
  findPath,
  generateCode,
  insertModuleFrom,
  makeUnique,
  packAsModule,
  pasteClip,
  portsOf,
  pruneDanglingEdges,
  removeEdges,
  removeNodes,
  renameModule,
  renameNode,
  resolvePath,
  setEntry,
  setHyperparams as setModuleHyperparams,
  setLayout,
  setParam,
  unpackModule,
  updateNode,
  withModule,
  type Analysis,
  type Clip,
  type CodeBlock,
  type CodegenResult,
  type ExportOptions,
  type HyperParam,
  type LoomProject,
  type ModuleDef,
  type ModuleId,
  type NodeId,
  type ParamValue,
  type ProjectAnalysis,
  type Repeat,
} from '@neuroloom/core';
import { applyNodeChanges, type Connection, type EdgeChange, type Node, type NodeChange } from '@xyflow/react';
import { create } from 'zustand';
import { nodeFaces, type NodeFace } from './canvas/face';
import { alignPositions, type AlignMode } from './lib/align';
import { sampleProject } from './lib/sample';
import { BLOCKS } from './templates';

export type FlowNode = Node<Record<string, never>, 'layer'>;

/** An existing port to wire a new node to: from a source port, or into a target port. */
export interface PortLink {
  nodeId: NodeId;
  port: string;
  side: 'source' | 'target';
}

/** Node types that are not NodeSpecs: an instance of a module, or a code block. */
export const moduleType = (moduleId: ModuleId) => `module:${moduleId}`;
/** A built-in composite module (PRD 9.6), added with the definitions it uses. */
export const blockType = (blockId: string) => `block:${blockId}`;

/**
 * The module on the canvas, and the drill-down path that led there (PRD F4.2),
 * which decides the instance whose shapes are shown.
 */
export interface View {
  moduleId: ModuleId;
  /** Module node ids from the entry; empty for the entry, or a module shown on its own. */
  path: NodeId[];
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
  moduleId: ModuleId;
  /** Whether the resulting move is an undoable edit (false for the layout that follows a load). */
  undoable: boolean;
  /** The module is already laid out: only fit it into view. */
  fitOnly?: boolean;
}

interface StudioState {
  project: LoomProject;
  analysis: ProjectAnalysis;
  view: View;
  /** The analysis of the module on the canvas, in the context of its view path. */
  viewAnalysis: Analysis;
  /** What each node of the module on the canvas shows; unchanged faces keep their object across edits. */
  faces: Record<NodeId, NodeFace>;
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
   * Adds a node (a NodeSpec type, `module:<id>`, `block:<id>` or the code block type); with
   * `link`, also connects it to an existing port in the same undo step. Returns
   * its id. Throws EditError when the module cannot go here.
   */
  addNodeAt(type: string, position: { x: number; y: number }, link?: PortLink): NodeId;
  /** `undefined` removes the parameter (a module argument back to its default). */
  setParam(nodeId: NodeId, name: string, value: ParamValue | undefined): void;
  rename(nodeId: NodeId, name: string): void;
  setRepeat(nodeId: NodeId, repeat: Repeat | undefined): void;
  setCode(nodeId: NodeId, code: CodeBlock): void;
  setShapes(nodeId: NodeId, shapes: Record<string, string> | undefined): void;
  /** Declares observed output shapes on nodes in any module, as one undo step. */
  applyShapes(observed: { moduleId: ModuleId; nodeId: NodeId; port: string; text: string }[]): void;
  selectNode(nodeId: NodeId | null): void;
  selectNodes(ids: NodeId[]): void;
  /** Pastes a clipboard fragment, offset from where it was copied, and selects it. */
  paste(clip: Clip, offset: { x: number; y: number }): void;
  deleteNodes(ids: NodeId[]): void;
  alignSelection(mode: AlignMode): void;

  // Modules (PRD section 9).
  enterModule(nodeId: NodeId): void;
  /** Shows a module from the outline, in the context of its first use when it has one. */
  openModule(moduleId: ModuleId, path?: NodeId[]): void;
  /** Breadcrumb: back to the module at this depth of the path. */
  goToDepth(depth: number): void;
  goUp(): void;
  packSelection(className: string): void;
  unpack(nodeId: NodeId): void;
  makeUnique(nodeId: NodeId): void;
  createModule(className: string): ModuleId;
  deleteModule(moduleId: ModuleId): void;
  setClassName(name: string, moduleId?: ModuleId): void;
  setEntry(moduleId: ModuleId): void;
  setHyperparams(hyperparams: HyperParam[], renamed?: Record<string, string>): void;
  /** How the code is written (PRD F5.3); kept in the project, so it is an undoable edit. */
  setExportOptions(options: ExportOptions): void;
  /** Free-text fields: the preamble or epilogue, a module's __init__ tail, other members or opaque source. */
  setText(field: 'preamble' | 'epilogue' | 'initCode' | 'members' | 'source', text: string): void;

  /** Replaces the document. `saved: false` marks it as never saved (e.g. imported code). */
  loadProject(project: LoomProject, doc?: { path?: string | null; name?: string; saved?: boolean; relayout?: boolean }): void;
  applyLayout(positions: Record<NodeId, { x: number; y: number }>, undoable: boolean, moduleId?: ModuleId): void;
  requestLayout(): void;
  /** Records that `project` (the snapshot that was written) is now on disk. */
  markSaved(project: LoomProject, path: string | null, name: string): void;
  undo(): void;
  redo(): void;
  /** Recomputes derived state after the interface language changes. */
  relocalize(): void;
}

function derive(project: LoomProject, view: View, faces?: Record<NodeId, NodeFace>): Pick<StudioState, 'analysis' | 'viewAnalysis' | 'faces' | 'codegen'> {
  const analysis = analyzeProject(project);
  const viewAnalysis = analysisAt(analysis, project, view.path, view.moduleId);
  return { analysis, viewAnalysis, faces: nodeFaces(project, project.modules[view.moduleId]!, viewAnalysis, faces), codegen: generateCode(project, analysis) };
}

/** Keeps the view pointing at something that exists after an edit, undo or load. */
function fixView(project: LoomProject, view: View): View {
  if (!project.modules[view.moduleId]) return { moduleId: project.entry, path: [] };
  const resolved = resolvePath(project, view.path);
  if (resolved.moduleId === view.moduleId) return resolved.path.length === view.path.length ? view : resolved;
  return { moduleId: view.moduleId, path: findPath(project, view.moduleId) ?? [] };
}

/** Shared by every flow node: what a node shows comes from `faces`, not from here. */
const NO_DATA: Record<string, never> = {};

/**
 * React Flow nodes for a module. A node whose position did not change keeps its
 * object, so React Flow (and the memoized node component) leaves it alone.
 */
function toFlowNodes(mod: ModuleDef, previous: FlowNode[] = []): FlowNode[] {
  const prev = new Map(previous.map((n) => [n.id, n]));
  return Object.values(mod.nodes).map((n) => {
    const before = prev.get(n.id);
    const position = mod.layout[n.id] ?? before?.position ?? { x: 0, y: 0 };
    if (before && before.position.x === position.x && before.position.y === position.y) return before;
    return { ...before, id: n.id, type: 'layer', position, data: NO_DATA };
  });
}

/** Selects exactly `ids`, touching only the nodes whose selection changes. */
const withSelection = (nodes: FlowNode[], ids: Set<NodeId>) =>
  nodes.map((n) => (!!n.selected === ids.has(n.id) ? n : { ...n, selected: ids.has(n.id) }));

/** A module some of whose nodes were never placed (imported submodules), which needs an automatic layout. */
const needsLayout = (mod: ModuleDef) => Object.keys(mod.nodes).some((id) => !mod.layout[id]);

const initial = sampleProject();
const initialView: View = { moduleId: initial.entry, path: [] };
const emptyHistory = (): History => ({ past: [], future: [], mergeAt: 0 });

export const useStudio = create<StudioState>()((set, get) => {
  const current = () => get().project.modules[get().view.moduleId]!;
  const inView = (project: LoomProject, update: (mod: ModuleDef) => ModuleDef) => withModule(project, get().view.moduleId, update);
  const selectedIds = () => get().flowNodes.filter((n) => n.selected).map((n) => n.id);

  /** Applies an edit as one undo step (or merges it into the previous one). */
  const commit = (next: LoomProject, options: { flowNodes?: FlowNode[]; merge?: string; view?: View; select?: NodeId[] } = {}) => {
    const { project: before, history } = get();
    const project = pruneDanglingEdges(next);
    if (project === before && !options.view) return;
    const view = fixView(project, options.view ?? get().view);
    const sameModule = view.moduleId === get().view.moduleId;
    let flowNodes = options.flowNodes ?? toFlowNodes(project.modules[view.moduleId]!, sameModule ? get().flowNodes : []);
    if (options.select) {
      const selected = new Set(options.select);
      flowNodes = withSelection(flowNodes, selected);
    }
    const now = Date.now();
    const merging = !!options.merge && history.mergeKey === options.merge && now - history.mergeAt < MERGE_WINDOW_MS;
    set({
      project,
      view,
      flowNodes,
      ...derive(project, view, sameModule ? get().faces : undefined),
      history:
        project === before
          ? history
          : { past: merging ? history.past : [...history.past, before].slice(-HISTORY_LIMIT), future: [], mergeKey: options.merge, mergeAt: now },
    });
    if (!sameModule) afterNavigate();
  };

  /** Moves to a snapshot from the history, keeping selection and measurements. */
  const restore = (project: LoomProject, history: History) => {
    const view = fixView(project, get().view);
    const sameModule = view.moduleId === get().view.moduleId;
    set({ project, history, view, flowNodes: toFlowNodes(project.modules[view.moduleId]!, sameModule ? get().flowNodes : []), ...derive(project, view, sameModule ? get().faces : undefined) });
  };

  /** Changes the module on the canvas without editing anything. */
  const navigate = (view: View) => {
    const { project } = get();
    const next = fixView(project, view);
    const viewAnalysis = analysisAt(get().analysis, project, next.path, next.moduleId);
    set({ view: next, flowNodes: toFlowNodes(project.modules[next.moduleId]!), selectedEdgeIds: [], viewAnalysis, faces: nodeFaces(project, project.modules[next.moduleId]!, viewAnalysis) });
    afterNavigate();
  };

  /** A module seen for the first time may need a layout; either way the canvas fits it into view. */
  const afterNavigate = () => {
    const { view, layoutRequest } = get();
    set({ layoutRequest: { id: layoutRequest.id + 1, moduleId: view.moduleId, undoable: false, fitOnly: !needsLayout(current()) } });
  };

  return {
    project: initial,
    view: initialView,
    ...derive(initial, initialView),
    flowNodes: toFlowNodes(initial.modules[initial.entry]!),
    selectedEdgeIds: [],
    layoutRequest: { id: 0, moduleId: initial.entry, undoable: false, fitOnly: true },
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
      if (Object.keys(moved).length > 0) project = inView(project, (m) => setLayout(m, moved));
      if (removed.length > 0) project = inView(project, (m) => removeNodes(m, removed));
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
      if (removed.length > 0) commit(inView(get().project, (m) => removeEdges(m, removed)));
    },

    onConnect({ source, sourceHandle, target, targetHandle }) {
      const mod = connect(current(), { node: source, port: sourceHandle ?? 'out' }, { node: target, port: targetHandle ?? 'in' });
      if (mod) commit(inView(get().project, () => mod));
    },

    addNodeAt(type, position, link) {
      const moduleId = get().view.moduleId;
      let project: LoomProject;
      let nodeId: NodeId;
      const block = type.startsWith('block:') ? BLOCKS.find((b) => b.id === type.slice('block:'.length)) : undefined;
      if (block) {
        ({ project, nodeId } = insertModuleFrom(get().project, moduleId, block.project, position));
      } else if (type.startsWith('module:')) {
        ({ project, nodeId } = addModuleNode(get().project, moduleId, type.slice('module:'.length), position));
      } else {
        const result = type === CODE_TYPE ? addCodeNode(current(), position) : addNode(current(), type, position);
        nodeId = result.nodeId;
        project = inView(get().project, () => result.mod);
      }
      if (link) {
        const mod = project.modules[moduleId]!;
        const ports = portsOf(project, mod.nodes[nodeId]!);
        const first = link.side === 'source' ? ports?.inputs[0] : ports?.outputs[0];
        const linked =
          first &&
          (link.side === 'source'
            ? connect(mod, { node: link.nodeId, port: link.port }, { node: nodeId, port: first.variadic ? `${first.name}0` : first.name })
            : connect(mod, { node: nodeId, port: first.name }, { node: link.nodeId, port: link.port }));
        if (linked) project = withModule(project, moduleId, () => linked);
      }
      commit(project, { select: [nodeId] });
      return nodeId;
    },

    setParam(nodeId, name, value) {
      commit(inView(get().project, (m) => setParam(m, nodeId, name, value)), { merge: `param:${nodeId}:${name}` });
    },

    rename(nodeId, name) {
      commit(inView(get().project, (m) => renameNode(m, nodeId, name)), { merge: `rename:${nodeId}` });
    },

    setRepeat(nodeId, repeat) {
      commit(inView(get().project, (m) => updateNode(m, nodeId, { repeat })), { merge: `repeat:${nodeId}` });
    },

    setCode(nodeId, code) {
      commit(inView(get().project, (m) => updateNode(m, nodeId, { code })), { merge: `code:${nodeId}` });
    },

    setShapes(nodeId, shapes) {
      commit(inView(get().project, (m) => updateNode(m, nodeId, { shapes })), { merge: `shapes:${nodeId}` });
    },

    applyShapes(observed) {
      let project = get().project;
      for (const o of observed) {
        project = withModule(project, o.moduleId, (m) => {
          const node = m.nodes[o.nodeId];
          return node ? updateNode(m, o.nodeId, { shapes: { ...node.shapes, [o.port]: o.text } }) : m;
        });
      }
      commit(project);
    },

    selectNode(nodeId) {
      set({ flowNodes: withSelection(get().flowNodes, new Set(nodeId ? [nodeId] : [])) });
    },

    selectNodes(ids) {
      const picked = new Set(ids);
      set({ flowNodes: withSelection(get().flowNodes, picked) });
    },

    paste(clip, offset) {
      const { project, nodeIds } = pasteClip(get().project, get().view.moduleId, clip, offset);
      commit(project, { select: nodeIds });
    },

    deleteNodes(ids) {
      if (ids.length > 0) commit(inView(get().project, (m) => removeNodes(m, ids)));
    },

    alignSelection(mode) {
      const picked = get().flowNodes.filter((n) => n.selected);
      if (picked.length < 2) return;
      const positions = alignPositions(
        picked.map((n) => ({ id: n.id, x: n.position.x, y: n.position.y, width: n.measured?.width ?? 168, height: n.measured?.height ?? 64 })),
        mode,
      );
      commit(inView(get().project, (m) => setLayout(m, positions)));
    },

    enterModule(nodeId) {
      const node = current().nodes[nodeId];
      if (node?.kind !== 'module' || !get().project.modules[node.type]) return;
      navigate({ moduleId: node.type, path: [...get().view.path, nodeId] });
    },

    openModule(moduleId, path) {
      navigate({ moduleId, path: path ?? findPath(get().project, moduleId) ?? [] });
    },

    goToDepth(depth) {
      const path = get().view.path.slice(0, depth);
      navigate({ moduleId: resolvePath(get().project, path).moduleId, path });
    },

    goUp() {
      const { view, project } = get();
      if (view.path.length > 0) get().goToDepth(view.path.length - 1);
      else if (view.moduleId !== project.entry) navigate({ moduleId: project.entry, path: [] });
    },

    packSelection(className) {
      const result = packAsModule(get().project, get().view.moduleId, selectedIds(), className, get().viewAnalysis);
      commit(result.project, { select: [result.nodeId] });
    },

    unpack(nodeId) {
      const result = unpackModule(get().project, get().view.moduleId, nodeId);
      commit(result.project, { select: result.nodeIds });
    },

    makeUnique(nodeId) {
      commit(makeUnique(get().project, get().view.moduleId, nodeId).project, { select: [nodeId] });
    },

    createModule(className) {
      const result = createModuleDef(get().project, className);
      commit(result.project, { view: { moduleId: result.moduleId, path: [] } });
      return result.moduleId;
    },

    deleteModule(moduleId) {
      commit(deleteModuleDef(get().project, moduleId));
    },

    setClassName(className, moduleId = get().view.moduleId) {
      commit(renameModule(get().project, moduleId, className), { merge: `className:${moduleId}` });
    },

    setEntry(moduleId) {
      commit(setEntry(get().project, moduleId), { view: { moduleId, path: [] } });
    },

    setHyperparams(hyperparams, renamed = {}) {
      commit(setModuleHyperparams(get().project, get().view.moduleId, hyperparams, renamed), { merge: `hyperparams:${get().view.moduleId}` });
    },

    setExportOptions(options) {
      const { exportOptions: _, ...rest } = get().project;
      commit(Object.keys(options).length > 0 ? { ...rest, exportOptions: options } : rest, { merge: 'exportOptions' });
    },

    setText(field, text) {
      const { project } = get();
      if (field === 'preamble' || field === 'epilogue') return commit({ ...project, [field]: text }, { merge: field });
      commit(
        inView(project, (m) => (field === 'source' ? { ...m, opaque: { ...m.opaque, source: text } } : { ...m, [field]: text })),
        { merge: `${field}:${get().view.moduleId}` },
      );
    },

    loadProject(project, doc = {}) {
      const view: View = { moduleId: project.entry, path: [] };
      const relayout = doc.relayout ?? needsLayout(project.modules[project.entry]!);
      set({
        project,
        view,
        ...derive(project, view),
        flowNodes: toFlowNodes(project.modules[project.entry]!),
        selectedEdgeIds: [],
        history: emptyHistory(),
        doc: { path: doc.path ?? null, name: doc.name ?? project.name, saved: doc.saved === false ? null : project },
        layoutRequest: { id: get().layoutRequest.id + 1, moduleId: project.entry, undoable: false, fitOnly: !relayout },
      });
    },

    applyLayout(positions, undoable, moduleId = get().view.moduleId) {
      const project = withModule(get().project, moduleId, (m) => setLayout(m, positions));
      if (undoable) return commit(project);
      // The layout that follows a load or a first visit is part of loading: no
      // undo step, and a clean document stays clean.
      const { doc, project: before, view } = get();
      set({
        project,
        flowNodes: moduleId === view.moduleId ? toFlowNodes(project.modules[moduleId]!, get().flowNodes) : get().flowNodes,
        ...derive(project, view, get().faces),
        doc: doc.saved === before ? { ...doc, saved: project } : doc,
      });
    },

    requestLayout() {
      set({ layoutRequest: { id: get().layoutRequest.id + 1, moduleId: get().view.moduleId, undoable: true } });
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

    relocalize() {
      set(derive(get().project, get().view, get().faces));
    },
  };
});

export const useCurrentModule = () => useStudio((s) => s.project.modules[s.view.moduleId]!);
export const useSelectedNodeId = () => useStudio((s) => s.flowNodes.find((n) => n.selected)?.id ?? null);
export const useSelectionCount = () => useStudio((s) => s.flowNodes.reduce((n, f) => n + (f.selected ? 1 : 0), 0));
export const useDirty = () => useStudio((s) => s.doc.saved !== s.project);
