// A one-page PDF around a JPEG, for the web build's image export (the desktop app
// prints a vector PDF in the main process instead). Uncompressed objects, the JPEG
// embedded as is (DCTDecode), and a classic cross-reference table.

export interface JpegPdfOptions {
  /** Size of the JPEG in pixels. */
  pixels: { width: number; height: number };
  /** Page size in points (1/72 in); the image fills the page. */
  page: { width: number; height: number };
  title?: string;
}

const ascii = (s: string) => new TextEncoder().encode(s);

const num = (v: number) => String(Math.round(v * 100) / 100);

/** A PDF text string as UTF-16BE hex with a byte-order mark, so any title survives. */
function hexText(s: string): string {
  let hex = 'FEFF';
  for (let i = 0; i < s.length; i++) hex += s.charCodeAt(i).toString(16).padStart(4, '0').toUpperCase();
  return `<${hex}>`;
}

export function jpegPdf(jpeg: Uint8Array, { pixels, page, title }: JpegPdfOptions): Uint8Array {
  const content = ascii(`q ${num(page.width)} 0 0 ${num(page.height)} 0 0 cm /Im0 Do Q\n`);
  const objects: (string | Uint8Array)[][] = [
    ['<< /Type /Catalog /Pages 2 0 R >>'],
    ['<< /Type /Pages /Kids [3 0 R] /Count 1 >>'],
    [
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${num(page.width)} ${num(page.height)}] ` +
        '/Resources << /XObject << /Im0 4 0 R >> /ProcSet [/PDF /ImageC] >> /Contents 5 0 R >>',
    ],
    [
      `<< /Type /XObject /Subtype /Image /Width ${pixels.width} /Height ${pixels.height} /ColorSpace /DeviceRGB ` +
        `/BitsPerComponent 8 /Filter /DCTDecode /Length ${jpeg.length} >>\nstream\n`,
      jpeg,
      '\nendstream',
    ],
    [`<< /Length ${content.length} >>\nstream\n`, content, 'endstream'],
    [`<< /Producer (NeuroLoom)${title ? ` /Title ${hexText(title)}` : ''} >>`],
  ];

  const chunks: Uint8Array[] = [];
  let length = 0;
  const push = (part: string | Uint8Array) => {
    const bytes = typeof part === 'string' ? ascii(part) : part;
    chunks.push(bytes);
    length += bytes.length;
  };

  // The second line marks the file as binary for transfer tools.
  push('%PDF-1.4\n');
  push(new Uint8Array([0x25, 0xe2, 0xe3, 0xcf, 0xd3, 0x0a]));
  const offsets: number[] = [];
  objects.forEach((parts, i) => {
    offsets.push(length);
    push(`${i + 1} 0 obj\n`);
    parts.forEach(push);
    push('\nendobj\n');
  });
  const xref = length;
  push(`xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`);
  for (const offset of offsets) push(`${String(offset).padStart(10, '0')} 00000 n \n`);
  push(`trailer\n<< /Size ${objects.length + 1} /Root 1 0 R /Info ${objects.length} 0 R >>\nstartxref\n${xref}\n%%EOF\n`);

  const out = new Uint8Array(length);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}
