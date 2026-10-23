// Editor state. The IR project is the single source of truth (PRD 15.4); React
// Flow nodes are a view of it that additionally carries measured sizes and
// selection, and analysis + code are derived on every semantic change.
import {
  analyzeModule,
  addNode,
  connect,
  entryModule,
  generateCode,
  removeEdges,
  removeNodes,
  renameNode,
  setLayout,
  setParam,
  type Analysis,
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

interface StudioState {
  project: LoomProject;
  analysis: Analysis;
  codegen: CodegenResult;
  flowNodes: FlowNode[];
  selectedEdgeIds: string[];
  /** Bumped when the graph needs an automatic layout (e.g. after an import). */
  layoutRequest: number;

  onNodesChange(changes: NodeChange<FlowNode>[]): void;
  onEdgesChange(changes: EdgeChange[]): void;
  onConnect(connection: Connection): void;
  addNodeAt(type: string, position: { x: number; y: number }): void;
  setParam(nodeId: NodeId, name: string, value: ParamValue): void;
  rename(nodeId: NodeId, name: string): void;
  setClassName(name: string): void;
  selectNode(nodeId: NodeId | null): void;
  loadProject(project: LoomProject, options?: { relayout?: boolean }): void;
  applyPositions(positions: Record<NodeId, { x: number; y: number }>): void;
  requestLayout(): void;
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

export const useStudio = create<StudioState>()((set, get) => {
  const commit = (project: LoomProject, flowNodes = get().flowNodes) => set({ project, flowNodes, ...derive(project) });

  return {
    project: initial,
    ...derive(initial),
    flowNodes: toFlowNodes(entryModule(initial)),
    selectedEdgeIds: [],
    layoutRequest: 0,

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
      commit(project, flowNodes);
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

    addNodeAt(type, position) {
      let nodeId = '';
      const project = withModule(get().project, (m) => {
        const result = addNode(m, type, position);
        nodeId = result.nodeId;
        return result.mod;
      });
      const flowNodes = toFlowNodes(entryModule(project), get().flowNodes).map((n) => ({ ...n, selected: n.id === nodeId }));
      commit(project, flowNodes);
    },

    setParam(nodeId, name, value) {
      commit(withModule(get().project, (m) => setParam(m, nodeId, name, value)));
    },

    rename(nodeId, name) {
      commit(withModule(get().project, (m) => renameNode(m, nodeId, name)));
    },

    setClassName(className) {
      commit(withModule(get().project, (m) => ({ ...m, className })));
    },

    selectNode(nodeId) {
      set({ flowNodes: get().flowNodes.map((n) => ({ ...n, selected: n.id === nodeId })) });
    },

    loadProject(project, options) {
      const mod = entryModule(project);
      const needsLayout = options?.relayout ?? Object.keys(mod.layout).length === 0;
      set({ selectedEdgeIds: [] });
      commit(project, toFlowNodes(mod));
      if (needsLayout) set({ layoutRequest: get().layoutRequest + 1 });
    },

    applyPositions(positions) {
      const project = withModule(get().project, (m) => setLayout(m, positions));
      commit(project, toFlowNodes(entryModule(project), get().flowNodes));
    },

    requestLayout() {
      set({ layoutRequest: get().layoutRequest + 1 });
    },
  };
});

export const useEntryModule = () => useStudio((s) => entryModule(s.project));
export const useSelectedNodeId = () => useStudio((s) => s.flowNodes.find((n) => n.selected)?.id ?? null);
