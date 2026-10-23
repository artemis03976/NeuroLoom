// Starts the studio Vite dev server, builds the Electron entry points and opens
// the app against the dev server (hot reload for the UI; restart for main-process changes).
import { spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import electron from 'electron';
import { createServer } from 'vite';
import { buildElectron } from './build.mjs';

const desktopRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const studioRoot = join(desktopRoot, '../studio');

// Another port is fine if a browser dev server already holds 5173.
const server = await createServer({
  root: studioRoot,
  configFile: join(studioRoot, 'vite.config.ts'),
  server: { strictPort: false },
});
await server.listen();
const url = server.resolvedUrls?.local[0];
if (!url) throw new Error('Vite did not report a local URL');

await buildElectron({ dev: true });
const child = spawn(electron, ['.'], {
  cwd: desktopRoot,
  stdio: 'inherit',
  env: { ...process.env, NEUROLOOM_DEV_URL: url },
});
child.on('exit', async (code) => {
  await server.close();
  process.exit(code ?? 0);
});
