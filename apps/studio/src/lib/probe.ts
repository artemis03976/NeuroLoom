// "Run once to fill in shapes": the shapes of code-block and opaque-module
// outputs, which inference cannot see, observed by running the generated model.
import { analyzeProject, dimFromRuns, generateCode, getLocale, literalOf, shapeText, type LoomProject, type ModuleId, type NodeId } from '@neuroloom/core';
import { rpc } from './rpc';

export interface ProbeResult {
  /** Two runs: "module/node/port" -> [shape, dtype]. */
  runs: Record<string, [number[], string]>[];
  /** The value each symbolic dimension had in each run. */
  values: Record<string, number>[];
}

export interface ObservedShape {
  moduleId: ModuleId;
  nodeId: NodeId;
  port: string;
  /** Shape text as nodes declare it, e.g. "B*T, 4" or "B, 16:int64". */
  text: string;
}

/** The instrumented code and the entry's input shapes ("B,3,32,32", "B,16:int64") to run it with. */
export function probeRequest(project: LoomProject): { code: string; className: string; inputs: string[] } {
  const { code } = generateCode(project, analyzeProject(project), { probe: true });
  const entry = project.modules[project.entry]!;
  // An optional forward() argument without an example shape is left at its default.
  const given = entry.interface.inputs.filter((id) => !(entry.interface.defaults?.[id] !== undefined && !String(literalOf(entry.nodes[id]!.params.shape) ?? '').trim()));
  const inputs = given.map((id) => {
    const node = entry.nodes[id]!;
    const dtype = String(literalOf(node.params.dtype) ?? 'float32');
    return `${String(literalOf(node.params.shape) ?? '').replace(/\s/g, '')}${dtype === 'float32' ? '' : `:${dtype}`}`;
  });
  return { code, className: entry.className, inputs };
}

/** Observed shapes as declarations; outputs whose dimensions are not products of symbols are left out. */
export function observedShapes(result: ProbeResult): ObservedShape[] {
  const [first, second] = result.runs;
  const [v1, v2] = result.values;
  if (!first || !second || !v1 || !v2) return [];
  const observed: ObservedShape[] = [];
  for (const [key, [shape, dtype]] of Object.entries(first)) {
    const other = second[key];
    const [moduleId, nodeId, port] = key.split('/') as [string, string, string];
    if (!other || other[0].length !== shape.length) continue;
    const dims = shape.map((d, i) => dimFromRuns(d, other[0][i]!, v1, v2));
    if (dims.some((d) => d === undefined)) continue;
    observed.push({ moduleId, nodeId, port, text: `${shapeText(dims as (number | string)[])}${dtype === 'float32' ? '' : `:${dtype}`}` });
  }
  return observed;
}

/** Runs the model (in the backend, in a subprocess) and returns the shapes it observed. */
export async function probeShapes(project: LoomProject): Promise<ObservedShape[]> {
  return observedShapes(await rpc<ProbeResult>('probe_shapes', { ...probeRequest(project), locale: getLocale() }));
}
