// Large graphs for measuring the editor (PRD 13: 1000 nodes / 1500 edges at
// 50 fps, shape updates within 100 ms). With `benchEnabled`, the store and the
// React Flow instance are on `window` (main.tsx, FlowCanvas.tsx) for
// apps/desktop/scripts/perf.mjs and for debugging from the console.
import { emptyModule, literal, SCHEMA_VERSION, type LoomProject } from '@neuroloom/core';

/** Development builds, or a page opened with `?bench`. */
export const benchEnabled = import.meta.env.DEV || (typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('bench'));

/**
 * `blocks` residual blocks (Linear → ReLU → add with a skip edge: 3 nodes, 4
 * edges each), laid out in rows so the graph spreads over the canvas.
 */
export function residualChain(blocks: number, perRow = 12): LoomProject {
  const mod = emptyModule('main', 'Deep');
  let edge = 0;
  const place = (id: string, i: number, step: number) => {
    mod.layout[id] = { x: (i % perRow) * 640 + step * 210, y: Math.floor(i / perRow) * 160 };
  };
  const link = (a: string, b: string, port = 'in') => mod.edges.push({ id: `e${edge++}`, source: { node: a, port: 'out' }, target: { node: b, port } });
  mod.nodes.x = { id: 'x', kind: 'input', type: 'neuroloom.Input', name: 'x', params: { shape: literal('B, 64') } };
  mod.layout.x = { x: -260, y: 0 };
  mod.interface.inputs.push('x');
  let prev = 'x';
  for (let i = 0; i < blocks; i++) {
    mod.nodes[`fc${i}`] = { id: `fc${i}`, kind: 'layer', type: 'torch.nn.Linear', name: `fc${i}`, params: { in_features: literal(64), out_features: literal(64) } };
    mod.nodes[`act${i}`] = { id: `act${i}`, kind: 'layer', type: 'torch.nn.ReLU', name: `act${i}`, params: {} };
    mod.nodes[`add${i}`] = { id: `add${i}`, kind: 'op', type: 'torch.add', name: `add${i}`, params: {} };
    place(`fc${i}`, i, 0);
    place(`act${i}`, i, 1);
    place(`add${i}`, i, 2);
    link(prev, `fc${i}`);
    link(`fc${i}`, `act${i}`);
    link(`act${i}`, `add${i}`, 'a');
    link(prev, `add${i}`, 'b');
    prev = `add${i}`;
  }
  mod.nodes.output = { id: 'output', kind: 'output', type: 'neuroloom.Output', name: 'output', params: {} };
  mod.layout.output = { x: (blocks % perRow) * 640, y: Math.floor(blocks / perRow) * 160 };
  mod.interface.outputs.push('output');
  link(prev, 'output');
  return { schemaVersion: SCHEMA_VERSION, name: 'Deep', entry: 'main', modules: { main: mod } };
}
