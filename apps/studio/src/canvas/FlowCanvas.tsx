import { wouldCreateCycle } from '@neuroloom/core';
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
import { useCallback, useEffect, useMemo, useRef, type DragEvent, type MouseEvent as ReactMouseEvent } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { attempt } from '../commands';
import { computeLayout } from '../lib/layout';
import { useStudio, type FlowNode, type PortLink } from '../store';
import { useUi } from '../ui';
import { nodeBadge } from './glyph';
import { LayerNode } from './LayerNode';
import { TensorEdge, type TensorEdgeType } from './TensorEdge';

export const NODE_DRAG_TYPE = 'application/x-neuroloom-node';
const GRID = 20;

const nodeTypes = { layer: LayerNode };
const edgeTypes = { tensor: TensorEdge };

const minimapColor = (n: FlowNode) => {
  const { project, view } = useStudio.getState();
  const node = project.modules[view.moduleId]?.nodes[n.id];
  return node ? nodeBadge(project, node).color : 'var(--nl-cat-code)';
};

export function FlowCanvas({ colorMode }: { colorMode: 'light' | 'dark' | 'system' }) {
  const { flowNodes, mod, edgeTypesById, selectedEdgeIds, layoutRequest, onNodesChange, onEdgesChange, onConnect, addNodeAt, applyLayout, enterModule } =
    useStudio(
      useShallow((s) => ({
        flowNodes: s.flowNodes,
        mod: s.project.modules[s.view.moduleId]!,
        edgeTypesById: s.viewAnalysis.edgeTypes,
        selectedEdgeIds: s.selectedEdgeIds,
        layoutRequest: s.layoutRequest,
        onNodesChange: s.onNodesChange,
        onEdgesChange: s.onEdgesChange,
        onConnect: s.onConnect,
        addNodeAt: s.addNodeAt,
        applyLayout: s.applyLayout,
        enterModule: s.enterModule,
      })),
    );
  const snapToGrid = useUi((s) => s.snapToGrid);
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

  // Auto layout once React Flow has measured the nodes, so ELK gets real sizes;
  // a module that is already laid out is only fitted into view.
  const handledLayout = useRef(-1);
  useEffect(() => {
    if (layoutRequest.id === handledLayout.current || !nodesInitialized || layoutRequest.moduleId !== mod.id) return;
    handledLayout.current = layoutRequest.id;
    const fit = () => setTimeout(() => void fitView({ padding: 0.15, duration: 300 }), 50);
    if (layoutRequest.fitOnly) return void fit();
    const sizes = new Map(
      flowNodes.map((n) => {
        const m = getInternalNode(n.id)?.measured;
        return [n.id, { width: m?.width ?? 180, height: m?.height ?? 72 }];
      }),
    );
    const box = document.querySelector('.react-flow')?.getBoundingClientRect();
    const canvas = box && box.width > 0 && box.height > 0 ? { width: box.width, height: box.height } : undefined;
    void computeLayout(mod, sizes, canvas).then((positions) => {
      applyLayout(positions, layoutRequest.undoable, layoutRequest.moduleId);
      // Wait for React Flow to take the new positions before fitting the view.
      fit();
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
      attempt(() => addNodeAt(type, { x: Math.round(p.x - 84), y: Math.round(p.y - 30) }));
    },
    [screenToFlowPosition, addNodeAt],
  );

  const onContextMenu = (e: ReactMouseEvent | MouseEvent, nodeId?: string) => {
    e.preventDefault();
    useUi.getState().setContextMenu({ screen: { x: e.clientX, y: e.clientY }, nodeId });
  };

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
        onNodeDoubleClick={(_e, node) => enterModule(node.id)}
        onNodeContextMenu={(e, node) => {
          if (!node.selected) useStudio.getState().selectNode(node.id);
          onContextMenu(e, node.id);
        }}
        onSelectionContextMenu={(e) => onContextMenu(e)}
        onPaneContextMenu={(e) => onContextMenu(e)}
        zoomOnDoubleClick={false}
        onDragOver={onDragOver}
        onDrop={onDrop}
        colorMode={colorMode}
        deleteKeyCode={['Backspace', 'Delete']}
        snapToGrid={snapToGrid}
        snapGrid={[GRID, GRID]}
        fitView
        fitViewOptions={{ padding: 0.25 }}
        minZoom={0.1}
        maxZoom={4}
      >
        <Background variant={snapToGrid ? BackgroundVariant.Lines : BackgroundVariant.Dots} gap={GRID} size={snapToGrid ? 0.6 : 1.4} />
        <Controls showInteractive={false} />
        <MiniMap pannable zoomable style={{ width: 168, height: 108 }} nodeColor={minimapColor} nodeBorderRadius={4} />
      </ReactFlow>
    </div>
  );
}
