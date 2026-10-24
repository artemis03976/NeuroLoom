import type { ImageFormat } from '@neuroloom/host-api';
import { useReactFlow } from '@xyflow/react';
import { useMemo, useState, type ReactNode } from 'react';
import { errorMessage, host } from '../host';
import { tr } from '../i18n';
import { buildSvg, type PaletteName, type Scene } from '../lib/exportImage';
import { sceneOf, type MeasuredNode } from '../lib/exportScene';
import { jpegPdf } from '../lib/pdf';
import { canvasMeasure, rasterScale, rasterize } from '../lib/raster';
import { useSelectionCount, useStudio } from '../store';
import { notify } from '../ui';

type Scope = 'level' | 'selection';

export interface ImageSettings {
  format: ImageFormat;
  /** PNG only: pixels per CSS px. */
  scale: number;
  scope: Scope;
  palette: PaletteName;
}

/** Remembered for the session; the palette follows the theme each time. */
let lastChoice: Pick<ImageSettings, 'format' | 'scale'> = { format: 'png', scale: 2 };

const SCALES = [1, 2, 3, 4];

function currentTheme(): PaletteName {
  const chosen = document.documentElement.dataset.theme;
  if (chosen === 'light' || chosen === 'dark') return chosen;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/** The entry class in lower case, plus the module on the canvas when it is not the entry. */
export function imageFileName(): string {
  const { project, view } = useStudio.getState();
  const clean = (s: string | undefined) => (s ?? '').replace(/[^A-Za-z0-9_]/g, '').toLowerCase();
  const entry = clean(project.modules[project.entry]?.className) || 'model';
  const current = view.moduleId !== project.entry ? clean(project.modules[view.moduleId]?.className) : '';
  return current ? `${entry}_${current}` : entry;
}

function currentScene(scope: Scope, getNode: (id: string) => MeasuredNode | undefined): Scene {
  const { project, view, viewAnalysis, flowNodes } = useStudio.getState();
  const only = scope === 'selection' ? flowNodes.filter((n) => n.selected).map((n) => n.id) : undefined;
  return sceneOf({ project, moduleId: view.moduleId, analysis: viewAnalysis, getNode, only });
}

/**
 * Renders the image and hands it to the host. Returns where it went (null when
 * the save dialog was cancelled) and the PNG scale actually used.
 */
export async function exportImage(
  settings: ImageSettings,
  getNode: (id: string) => MeasuredNode | undefined,
): Promise<{ where: string | null; scale?: number }> {
  const scene = currentScene(settings.scope, getNode);
  if (scene.nodes.length === 0) throw new Error(tr('没有可导出的节点', 'There are no nodes to export'));
  const base = { palette: settings.palette, measure: canvasMeasure() };
  const suggestedName = imageFileName();
  const image = buildSvg(scene, base);

  if (settings.format === 'svg' || (settings.format === 'pdf' && host.kind === 'desktop')) {
    // The desktop app prints the SVG to a vector PDF itself.
    return { where: await host.exportImage({ format: settings.format, data: image.svg, suggestedName }) };
  }
  if (settings.format === 'png') {
    const scale = rasterScale(image.width, image.height, settings.scale);
    const png = await rasterize(buildSvg(scene, { ...base, scale }).svg, 'image/png');
    return { where: await host.exportImage({ format: 'png', data: png.bytes, suggestedName }), scale };
  }
  // A browser has no vector PDF writer: a 2× JPEG on a page the size of the drawing (1 px = 0.75 pt).
  const scale = rasterScale(image.width, image.height, 2);
  const jpeg = await rasterize(buildSvg(scene, { ...base, scale }).svg, 'image/jpeg', 0.92);
  const pdf = jpegPdf(jpeg.bytes, { pixels: jpeg, page: { width: image.width * 0.75, height: image.height * 0.75 }, title: scene.title });
  return { where: await host.exportImage({ format: 'pdf', data: pdf, suggestedName }) };
}

function Choice<T extends string | number>({
  name,
  value,
  current,
  onPick,
  disabled,
  children,
}: {
  name: string;
  value: T;
  current: T;
  onPick: (value: T) => void;
  disabled?: boolean;
  children: ReactNode;
}) {
  const on = value === current;
  return (
    <label
      className={`flex-1 cursor-pointer rounded-md border px-3 py-1.5 text-center text-[12.5px] has-focus-visible:ring-2 has-focus-visible:ring-primary/40 has-disabled:cursor-not-allowed has-disabled:opacity-40 ${
        on ? 'border-primary bg-primary-soft text-primary' : 'border-line hover:bg-surface-2'
      }`}
    >
      <input type="radio" className="sr-only" name={name} value={String(value)} checked={on} disabled={disabled} onChange={() => onPick(value)} />
      {children}
    </label>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <fieldset className="flex flex-col gap-1.5">
      <legend className="mb-1.5 text-[12px] text-muted">{label}</legend>
      <div className="flex gap-2">{children}</div>
    </fieldset>
  );
}

/** File → Export Image (PRD F7.3): SVG, PNG or PDF of the module on the canvas or the selection. */
export function ImageExportDialog({ onClose }: { onClose: () => void }) {
  const { getInternalNode } = useReactFlow();
  const selected = useSelectionCount();
  const [format, setFormat] = useState<ImageFormat>(lastChoice.format);
  const [scale, setScale] = useState(lastChoice.scale);
  const [scope, setScope] = useState<Scope>('level');
  const [palette, setPalette] = useState<PaletteName>(currentTheme);
  const [busy, setBusy] = useState(false);

  // The drawing's size, for the PNG pixel count.
  const size = useMemo(() => {
    const scene = currentScene(scope, getInternalNode);
    return scene.nodes.length > 0 ? buildSvg(scene, { palette: 'light' }) : undefined;
  }, [scope, getInternalNode]);
  const pngScale = size ? rasterScale(size.width, size.height, scale) : scale;

  const submit = async () => {
    lastChoice = { format, scale };
    setBusy(true);
    try {
      const result = await exportImage({ format, scale, scope, palette }, getInternalNode);
      if (!result.where) return;
      const name = result.where.split(/[\\/]/).pop() ?? result.where;
      const reduced =
        result.scale !== undefined && result.scale < scale
          ? [tr(`图片过大，已按 ${result.scale}× 导出`, `The image was too large; exported at ${result.scale}×`)]
          : undefined;
      notify(host.kind === 'desktop' ? tr(`已导出 ${result.where}`, `Exported ${result.where}`) : tr(`已下载 ${name}`, `Downloaded ${name}`), 'success', reduced);
      onClose();
    } catch (e) {
      notify(tr(`导出失败：${errorMessage(e)}`, `Export failed: ${errorMessage(e)}`), 'error');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-6" onMouseDown={() => !busy && onClose()}>
      <form
        role="dialog"
        aria-modal="true"
        aria-labelledby="image-export-title"
        className="w-[min(440px,100%)] rounded-xl border border-line bg-surface shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.key === 'Escape' && !busy && onClose()}
        onSubmit={(e) => {
          e.preventDefault();
          if (!busy) void submit();
        }}
      >
        <header className="border-b border-line px-5 py-3">
          <h2 id="image-export-title" className="text-[15px] font-semibold">
            {tr('导出图片', 'Export image')}
          </h2>
          <p className="mt-1 text-[12px] text-muted">
            {tr('导出画布上的网络图，不含网格、动画和缩放控件。', 'The network as drawn on the canvas, without the grid, animations or zoom controls.')}
          </p>
        </header>
        <div className="flex flex-col gap-4 px-5 py-4">
          <Field label={tr('格式', 'Format')}>
            <Choice name="image-format" value="svg" current={format} onPick={setFormat}>
              SVG
            </Choice>
            <Choice name="image-format" value="png" current={format} onPick={setFormat}>
              PNG
            </Choice>
            <Choice name="image-format" value="pdf" current={format} onPick={setFormat}>
              PDF
            </Choice>
          </Field>
          {format === 'png' && (
            <Field label={tr('分辨率', 'Resolution')}>
              {SCALES.map((s) => (
                <Choice key={s} name="image-scale" value={s} current={scale} onPick={setScale}>
                  {s}×
                </Choice>
              ))}
            </Field>
          )}
          {format === 'png' && size && (
            <p className="-mt-2 font-mono text-[11.5px] text-muted">
              {Math.round(size.width * pngScale)} × {Math.round(size.height * pngScale)} px
            </p>
          )}
          {format === 'pdf' && host.kind === 'web' && (
            <p className="-mt-2 text-[11.5px] text-muted">
              {tr('网页版的 PDF 是 2× 位图；桌面版导出矢量 PDF。', 'In the browser the PDF holds a 2× bitmap; the desktop app exports a vector PDF.')}
            </p>
          )}
          <Field label={tr('范围', 'Scope')}>
            <Choice name="image-scope" value="level" current={scope} onPick={setScope}>
              {tr('当前层级', 'Current level')}
            </Choice>
            <Choice name="image-scope" value="selection" current={scope} onPick={setScope} disabled={selected === 0}>
              {selected > 0 ? tr(`选中的 ${selected} 个节点`, `${selected} selected node${selected === 1 ? '' : 's'}`) : tr('选中的节点', 'Selected nodes')}
            </Choice>
          </Field>
          <Field label={tr('配色', 'Colours')}>
            <Choice name="image-palette" value="light" current={palette} onPick={setPalette}>
              {tr('浅色', 'Light')}
            </Choice>
            <Choice name="image-palette" value="dark" current={palette} onPick={setPalette}>
              {tr('深色', 'Dark')}
            </Choice>
            <Choice name="image-palette" value="print" current={palette} onPick={setPalette}>
              {tr('打印', 'Print')}
            </Choice>
          </Field>
        </div>
        <footer className="flex justify-end gap-2 border-t border-line px-5 py-3">
          <button type="button" className="rounded-md border border-line px-3 py-1.5 text-[13px] hover:bg-surface-2" disabled={busy} onClick={onClose}>
            {tr('取消', 'Cancel')}
          </button>
          <button
            type="submit"
            autoFocus
            disabled={busy}
            className="rounded-md bg-primary px-3 py-1.5 text-[13px] font-medium text-primary-fg hover:bg-primary-hover disabled:opacity-50"
          >
            {busy ? tr('正在导出…', 'Exporting…') : tr('导出…', 'Export…')}
          </button>
        </footer>
      </form>
    </div>
  );
}
