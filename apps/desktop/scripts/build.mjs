// Bundles the main process (ESM) and the sandboxed preload (CommonJS) with esbuild.
// Run directly, it also builds the studio UI that the app serves from app://studio.
import { build } from 'esbuild';
import { spawnSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const desktopRoot = join(dirname(fileURLToPath(import.meta.url)), '..');

export async function buildElectron({ dev = false } = {}) {
  const common = {
    bundle: true,
    platform: 'node',
    target: 'node22',
    external: ['electron'],
    sourcemap: dev ? 'inline' : false,
    logLevel: 'warning',
    absWorkingDir: desktopRoot,
  };
  await build({ ...common, entryPoints: ['src/main.ts'], outfile: 'dist/main.js', format: 'esm' });
  await build({ ...common, entryPoints: ['src/preload.ts'], outfile: 'dist/preload.cjs', format: 'cjs' });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const studio = spawnSync('npm', ['run', 'build', '-w', '@neuroloom/studio'], {
    cwd: join(desktopRoot, '../..'),
    stdio: 'inherit',
    shell: true,
  });
  if (studio.status !== 0) process.exit(studio.status ?? 1);
  await buildElectron();
  console.log('desktop: built dist/main.js and dist/preload.cjs');
}
