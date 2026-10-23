// Writes python/neuroloom/specs.json from the TypeScript NodeSpecs.
// Usage: npm run specs   (the core test suite fails when the file is stale)
import { build } from 'esbuild';
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const coreRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const target = join(coreRoot, '../../python/neuroloom/specs.json');

const bundle = await build({
  entryPoints: [join(coreRoot, 'src/manifest.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  write: false,
  logLevel: 'warning',
});
const { manifestText } = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);
writeFileSync(target, manifestText());
console.log(`specs: wrote ${target}`);
