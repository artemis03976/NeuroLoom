// Browser side of the image export: text measurement with a canvas, and SVG to
// PNG or JPEG through an <img> and a canvas.
import { MONO_FONT, SANS_FONT, type MeasureText } from './exportImage';

/** Measures text in the fonts the exported SVG names, as this machine renders them. */
export function canvasMeasure(): MeasureText | undefined {
  const ctx = document.createElement('canvas').getContext('2d');
  if (!ctx) return undefined;
  return (text, font) => {
    ctx.font = `${font.weight ?? 400} ${font.size}px ${font.mono ? MONO_FONT : SANS_FONT}`;
    return ctx.measureText(text).width;
  };
}

// Browsers refuse canvases much larger than this (and the memory adds up fast).
const MAX_SIDE = 16384;
const MAX_PIXELS = 64e6;

/** The largest scale up to `requested` at which a width × height image can be rasterized. */
export function rasterScale(width: number, height: number, requested: number): number {
  const limit = Math.min(MAX_SIDE / width, MAX_SIDE / height, Math.sqrt(MAX_PIXELS / (width * height)));
  return Math.min(requested, Math.floor(limit * 100) / 100);
}

/** Draws an SVG document (already sized in pixels) onto a canvas and encodes it. */
export async function rasterize(svg: string, type: 'image/png' | 'image/jpeg', quality?: number): Promise<{ bytes: Uint8Array; width: number; height: number }> {
  const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
  try {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('The SVG could not be rendered'));
      img.src = url;
    });
    const width = Math.max(1, Math.round(img.naturalWidth));
    const height = Math.max(1, Math.round(img.naturalHeight));
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('No 2D canvas');
    ctx.drawImage(img, 0, 0, width, height);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, type, quality));
    if (!blob) throw new Error('The image is too large to encode');
    return { bytes: new Uint8Array(await blob.arrayBuffer()), width, height };
  } finally {
    URL.revokeObjectURL(url);
  }
}
