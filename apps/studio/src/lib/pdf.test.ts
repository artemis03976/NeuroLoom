import { describe, expect, it } from 'vitest';
import { jpegPdf } from './pdf';

// Not a decodable picture, but the writer embeds the bytes as they are.
const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3, 0x0a, 0x0d, 0xff, 0xd9]);
const latin1 = (bytes: Uint8Array) => Array.from(bytes, (b) => String.fromCharCode(b)).join('');

describe('image export: PDF around a JPEG', () => {
  const pdf = jpegPdf(jpeg, { pixels: { width: 1508, height: 456 }, page: { width: 565.5, height: 171 }, title: 'ResNet 残差' });
  const text = latin1(pdf);

  it('has a header, a page the size of the drawing and the image filling it', () => {
    expect(text.startsWith('%PDF-1.4\n%')).toBe(true);
    expect(pdf[10]! > 127).toBe(true); // binary marker
    expect(text).toContain('/MediaBox [0 0 565.5 171]');
    expect(text).toContain('/Width 1508 /Height 456 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length 11 >>');
    expect(text).toContain('q 565.5 0 0 171 0 0 cm /Im0 Do Q');
    expect(text.endsWith('%%EOF\n')).toBe(true);
  });

  it('embeds the JPEG byte for byte', () => {
    const at = text.indexOf('stream\n') + 'stream\n'.length;
    expect(Array.from(pdf.slice(at, at + jpeg.length))).toEqual(Array.from(jpeg));
    expect(text.slice(at + jpeg.length, at + jpeg.length + 10)).toBe('\nendstream');
  });

  it('has a cross-reference table that points at every object', () => {
    const startxref = Number(/startxref\n(\d+)\n%%EOF/.exec(text)![1]);
    expect(text.slice(startxref, startxref + 4)).toBe('xref');
    const entries = text.slice(startxref).split('\n').slice(2, 9);
    expect(entries[0]).toBe('0000000000 65535 f ');
    entries.slice(1).forEach((entry, i) => {
      expect(entry).toMatch(/^\d{10} 00000 n $/);
      const offset = Number(entry.slice(0, 10));
      expect(text.slice(offset, offset + 8)).toBe(`${i + 1} 0 obj\n`);
    });
    expect(text).toContain('trailer\n<< /Size 7 /Root 1 0 R /Info 6 0 R >>');
  });

  it('declares content stream lengths correctly', () => {
    const m = /5 0 obj\n<< \/Length (\d+) >>\nstream\n([\s\S]*?)endstream/.exec(text)!;
    expect(m[2]!.length).toBe(Number(m[1]));
  });

  it('writes the title as UTF-16', () => {
    expect(text).toContain('/Title <FEFF005200650073004E0065007400206B8B5DEE>');
  });
});
