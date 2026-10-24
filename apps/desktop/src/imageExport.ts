// Saving exported diagrams (PRD F7.3). SVG and PNG arrive ready to write; a PDF
// arrives as SVG, which a hidden window prints to a vector PDF the size of the drawing.
import type { ImageExport, ImageFormat } from '@neuroloom/host-api';
import { app, BrowserWindow, dialog, type SaveDialogOptions } from 'electron';
import { rmSync, writeFileSync } from 'node:fs';
import { basename, extname, join } from 'node:path';
import { writeAtomic } from './settings';

const FILTERS: Record<ImageFormat, { name: string; extensions: string[] }> = {
  svg: { name: 'SVG', extensions: ['svg'] },
  png: { name: 'PNG', extensions: ['png'] },
  pdf: { name: 'PDF', extensions: ['pdf'] },
};

/** Many PDF readers refuse pages over 200 in; larger drawings are scaled down to fit (they stay vector). */
const MAX_PAGE_PX = 200 * 96;

/** The width and height attributes of the root <svg>, in CSS px. */
export function svgSize(svg: string): { width: number; height: number } {
  const root = /<svg\b[^>]*>/.exec(svg)?.[0] ?? '';
  const attr = (name: string) => Number(new RegExp(`\\s${name}="([\\d.]+)(?:px)?"`).exec(root)?.[1]);
  const width = attr('width');
  const height = attr('height');
  if (!(width > 0) || !(height > 0)) throw new Error('Not an SVG image with a width and height');
  return { width, height };
}

/** Prints an SVG to a one-page vector PDF whose page is the drawing, with no margins. */
export async function svgToPdf(svg: string): Promise<Buffer> {
  const size = svgSize(svg);
  const fit = Math.min(1, MAX_PAGE_PX / size.width, MAX_PAGE_PX / size.height);
  const width = Math.round(size.width * fit * 100) / 100;
  const height = Math.round(size.height * fit * 100) / 100;
  const html =
    '<!doctype html><html><head><meta charset="utf-8"><style>' +
    `@page{size:${width}px ${height}px;margin:0}` +
    `html,body{margin:0;padding:0;width:${width}px;height:${height}px;overflow:hidden}` +
    `svg{display:block;width:${width}px;height:${height}px}` +
    `</style></head><body>${svg.replace(/^\s*<\?xml[^>]*\?>\s*/, '')}</body></html>`;
  // A file rather than a data: URL, which Chromium caps at 2 MB.
  const file = join(app.getPath('temp'), `neuroloom-export-${process.pid}-${Date.now()}.html`);
  writeFileSync(file, html, 'utf8');
  const win = new BrowserWindow({
    show: false,
    width: 800,
    height: 600,
    webPreferences: { javascript: false, sandbox: true, contextIsolation: true, nodeIntegration: false, spellcheck: false },
  });
  try {
    await win.loadFile(file);
    return await win.webContents.printToPDF({
      pageSize: { width: width / 96, height: height / 96 },
      margins: { top: 0, bottom: 0, left: 0, right: 0 },
      printBackground: true,
      pageRanges: '1',
    });
  } finally {
    win.destroy();
    rmSync(file, { force: true });
  }
}

/** Asks where to save the image and writes it; returns the path, or null when cancelled. */
export async function saveImage(parent: BrowserWindow | undefined, args: ImageExport, title: string): Promise<string | null> {
  const filter = FILTERS[args.format];
  if (!filter) throw new Error(`Unknown image format: ${String(args.format)}`);
  const text = typeof args.data === 'string';
  if (args.format === 'svg' ? !text : args.format === 'png' ? text || !ArrayBuffer.isView(args.data) : !text && !ArrayBuffer.isView(args.data)) {
    throw new Error(`Unexpected data for a ${args.format.toUpperCase()} image`);
  }
  const ext = filter.extensions[0]!;
  const options: SaveDialogOptions = {
    title,
    defaultPath: join(app.getPath('documents'), `${basename(args.suggestedName)}.${ext}`),
    filters: [filter],
  };
  const result = parent ? await dialog.showSaveDialog(parent, options) : await dialog.showSaveDialog(options);
  if (result.canceled || !result.filePath) return null;
  const path = extname(result.filePath) ? result.filePath : `${result.filePath}.${ext}`;
  const data = args.format === 'pdf' && typeof args.data === 'string' ? await svgToPdf(args.data) : args.data;
  writeAtomic(path, data);
  return path;
}
