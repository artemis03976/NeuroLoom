import { entryModule, getSpec, wouldCreateCycle } from '@neuroloom/core';
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  useNodesInitialized,
  useReactFlow,
  type Connection,
  type Edge,
  type IsValidConnection,
  type OnConnectEnd,
} from '@xyflow/react';
import { useCallback, useEffect, useMemo, useRef, type DragEvent } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { computeLayout } from '../lib/layout';
import { useStudio, type FlowNode, type PortLink } from '../store';
import { useUi } from '../ui';
import { categoryColor } from './glyph';
import { LayerNode } from './LayerNode';
import { TensorEdge, type TensorEdgeType } from './TensorEdge';

export const NODE_DRAG_TYPE = 'application/x-neuroloom-node';

const nodeTypes = { layer: LayerNode };
const edgeTypes = { tensor: TensorEdge };

const minimapColor = (n: FlowNode) => categoryColor(getSpec(entryModule(useStudio.getState().project).nodes[n.id]?.type));

export function FlowCanvas({ colorMode }: { colorMode: 'light' | 'dark' | 'system' }) {
  const { flowNodes, project, edgeTypesById, selectedEdgeIds, layoutRequest, onNodesChange, onEdgesChange, onConnect, addNodeAt, applyLayout } =
    useStudio(
      useShallow((s) => ({
        flowNodes: s.flowNodes,
        project: s.project,
        edgeTypesById: s.analysis.edgeTypes,
        selectedEdgeIds: s.selectedEdgeIds,
        layoutRequest: s.layoutRequest,
        onNodesChange: s.onNodesChange,
        onEdgesChange: s.onEdgesChange,
        onConnect: s.onConnect,
        addNodeAt: s.addNodeAt,
        applyLayout: s.applyLayout,
      })),
    );
  const mod = entryModule(project);
  const { screenToFlowPosition, fitView, getInternalNode } = useReactFlow<FlowNode, TensorEdgeType>();
  const nodesInitialized = useNodesInitialized();

  const edges = useMemo<TensorEdgeType[]>(
    () =>
      mod.edges.map((e) => ({
        id: e.id,
        type: 'tensor',
        source: e.source.node,
        sourceHandle: e.source.port,
        target: e.target.node,
        targetHandle: e.target.port,
        selected: selectedEdgeIds.includes(e.id),
        data: { shape: edgeTypesById[e.id]?.shape },
      })),
    [mod.edges, edgeTypesById, selectedEdgeIds],
  );

  // Auto layout once React Flow has measured the nodes, so ELK gets real sizes.
  const handledLayout = useRef(0);
  useEffect(() => {
    if (layoutRequest.id === handledLayout.current || !nodesInitialized) return;
    handledLayout.current = layoutRequest.id;
    const sizes = new Map(
      flowNodes.map((n) => {
        const m = getInternalNode(n.id)?.measured;
        return [n.id, { width: m?.width ?? 180, height: m?.height ?? 72 }];
      }),
    );
    const box = document.querySelector('.react-flow')?.getBoundingClientRect();
    const canvas = box && box.width > 0 && box.height > 0 ? { width: box.width, height: box.height } : undefined;
    void computeLayout(mod, sizes, canvas).then((positions) => {
      applyLayout(positions, layoutRequest.undoable);
      // Wait for React Flow to take the new positions before fitting the view.
      setTimeout(() => void fitView({ padding: 0.15, duration: 300 }), 50);
    });
  }, [layoutRequest, nodesInitialized, flowNodes, mod, getInternalNode, applyLayout, fitView]);

  const isValidConnection: IsValidConnection<Edge> = useCallback(
    (c: Connection | Edge) => !wouldCreateCycle(mod, c.source, c.target),
    [mod],
  );

  const openQuickAdd = (x: number, y: number, link?: PortLink) => useUi.getState().setQuickAdd({ screen: { x, y }, link });

  // An edge dropped on empty canvas opens quick-add, and the new node gets wired to it.
  const onConnectEnd: OnConnectEnd = useCallback(
    (event, state) => {
      if (state.isValid || !state.fromNode || !state.fromHandle?.id) return;
      if (!(event.target instanceof Element) || !event.target.classList.contains('react-flow__pane')) return;
      const { clientX, clientY } = 'changedTouches' in event ? event.changedTouches[0]! : event;
      openQuickAdd(clientX, clientY, { nodeId: state.fromNode.id, port: state.fromHandle.id, side: state.fromHandle.type });
    },
    [],
  );

  const onDragOver = useCallback((e: DragEvent) => {
    if (e.dataTransfer.types.includes(NODE_DRAG_TYPE)) {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    }
  }, []);

  const onDrop = useCallback(
    (e: DragEvent) => {
      const type = e.dataTransfer.getData(NODE_DRAG_TYPE);
      if (!type) return;
      e.preventDefault();
      const p = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      addNodeAt(type, { x: Math.round(p.x - 84), y: Math.round(p.y - 30) });
    },
    [screenToFlowPosition, addNodeAt],
  );

  return (
    <div
      className="h-full"
      onMouseMove={(e) => (useUi.getState().pointer = { x: e.clientX, y: e.clientY })}
      onDoubleClick={(e) => {
        if (e.target instanceof Element && e.target.classList.contains('react-flow__pane')) openQuickAdd(e.clientX, e.clientY);
      }}
    >
    <ReactFlow<FlowNode, TensorEdgeType>
      nodes={flowNodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onConnectEnd={onConnectEnd}
      isValidConnection={isValidConnection}
      zoomOnDoubleClick={false}
      onDragOver={onDragOver}
      onDrop={onDrop}
      colorMode={colorMode}
      deleteKeyCode={['Backspace', 'Delete']}
      fitView
      fitViewOptions={{ padding: 0.25 }}
      minZoom={0.1}
      maxZoom={4}
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1.4} />
      <Controls showInteractive={false} />
      <MiniMap pannable zoomable style={{ width: 168, height: 108 }} nodeColor={minimapColor} nodeBorderRadius={4} />
    </ReactFlow>
    </div>
  );
}
