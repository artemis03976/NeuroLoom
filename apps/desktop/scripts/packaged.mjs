// The packaged app that `npm run dist` (or `npm run dist -- --dir`) leaves in release/.
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const desktopRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const release = join(desktopRoot, 'release');

/** The unpacked app's executable for this platform, or undefined when it has not been built. */
export function packagedExecutable() {
  const candidates = {
    win32: [join(release, 'win-unpacked', 'NeuroLoom.exe')],
    darwin: [process.arch === 'arm64' ? 'mac-arm64' : 'mac', 'mac-arm64', 'mac'].map((d) => join(release, d, 'NeuroLoom.app', 'Contents', 'MacOS', 'NeuroLoom')),
    linux: [join(release, 'linux-unpacked', 'neuroloom')],
  }[process.platform] ?? [];
  return candidates.find((p) => existsSync(p));
}
