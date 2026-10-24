import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { crc32, zip } from './zip';

const files = [
  { name: 'net/__init__.py', text: 'from .res_net import ResNet\n' },
  { name: 'net/res_net.py', text: '# 织络\nclass ResNet: ...\n' },
];

describe('zip', () => {
  it('computes the standard CRC-32', () => {
    expect(crc32(new TextEncoder().encode('123456789'))).toBe(0xcbf43926);
  });

  it('is deterministic and ends with the end-of-central-directory record', () => {
    const bytes = zip(files);
    expect(zip(files)).toEqual(bytes);
    const view = new DataView(bytes.buffer);
    expect(view.getUint32(0, true)).toBe(0x04034b50);
    expect(view.getUint32(bytes.length - 22, true)).toBe(0x06054b50);
    expect(view.getUint16(bytes.length - 12, true)).toBe(2);
  });

  // Python's zipfile checks every CRC on read.
  const python = spawnSync('python', ['--version']).status === 0 ? 'python' : undefined;
  it.skipIf(!python)('reads back with Python zipfile', () => {
    const path = join(mkdtempSync(join(tmpdir(), 'nl-zip-')), 'package.zip');
    writeFileSync(path, zip(files));
    const script = 'import sys, zipfile, json; z = zipfile.ZipFile(sys.argv[1]); print(json.dumps({n: z.read(n).decode() for n in z.namelist()}))';
    const out = spawnSync(python!, ['-c', script, path], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
    expect(JSON.parse(out.stdout)).toEqual(Object.fromEntries(files.map((f) => [f.name, f.text])));
  });
});
