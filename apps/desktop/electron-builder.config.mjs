// Installers for Windows (NSIS, x64), macOS (DMG, x64 and arm64, macOS 12+) and Linux
// (AppImage and .deb, x64). Run `npm run dist` from the repo root, which builds the
// studio UI and the Electron entry points first, then packages for this platform.
//
// Packaged layout (process.resourcesPath, see src/main.ts):
//   app.asar          dist/main.js, dist/preload.cjs, package.json (no node_modules:
//                     esbuild bundles every dependency)
//   studio/           apps/studio/dist, served over app://studio
//   python/neuroloom/ the backend package, run as `python -m neuroloom serve` with the
//                     user's own interpreter (Python and torch are not bundled)
//   icon.png          window icon on Linux
//
// Signing happens only when credentials are in the environment, otherwise the build is
// unsigned and still succeeds:
//   Windows  CSC_LINK + CSC_KEY_PASSWORD (or WIN_CSC_LINK + WIN_CSC_KEY_PASSWORD)
//   macOS    CSC_LINK + CSC_KEY_PASSWORD (Developer ID Application certificate), and for
//            notarization APPLE_ID + APPLE_APP_SPECIFIC_PASSWORD + APPLE_TEAM_ID
// Without a certificate the macOS app is signed ad hoc (Apple Silicon refuses to run
// unsigned code) and users have to allow it in System Settings → Privacy & Security.

import { createRequire } from 'node:module';

const macSigning = Boolean(process.env.CSC_LINK || process.env.CSC_NAME);
const icon = '../../brand/assets/icons/icon-1024.png';
// npm hoists electron to the workspace root, where electron-builder does not look for it.
const electronVersion = createRequire(import.meta.url)('electron/package.json').version;

/** @type {import('electron-builder').Configuration} */
export default {
  appId: 'app.neuroloom.studio',
  electronVersion,
  productName: 'NeuroLoom',
  copyright: 'Copyright © NeuroLoom contributors',
  // Written into the packaged package.json: Electron names the app (and its userData
  // directory) after productName, and Linux packages and executables are named after name.
  extraMetadata: { name: 'neuroloom', productName: 'NeuroLoom', description: 'Visual neural network builder for PyTorch' },
  directories: { output: 'release' },
  artifactName: '${productName}-${version}-${os}-${arch}.${ext}',

  files: ['dist/main.js', 'dist/preload.cjs', 'package.json'],
  extraResources: [
    { from: '../studio/dist', to: 'studio' },
    { from: '../../python/neuroloom', to: 'python/neuroloom', filter: ['**/*', '!**/__pycache__${/*}', '!**/*.pyc'] },
    { from: '../../brand/assets/icons/icon-256.png', to: 'icon.png' },
  ],
  // The editor speaks Chinese and English; Chromium's other locale packs only add size.
  electronLanguages: ['en-US', 'zh-CN', 'zh-TW'],
  // Every installer has to stay under 150 MB (PRD M3). Maximum compression takes about a
  // tenth off the NSIS installer and makes the DMGs bzip2 (UDBZ).
  compression: 'maximum',
  npmRebuild: false,

  fileAssociations: [
    {
      ext: 'loom',
      name: 'NeuroLoom Project',
      description: 'NeuroLoom project',
      mimeType: 'application/x-neuroloom',
      role: 'Editor',
      rank: 'Owner',
    },
  ],

  win: {
    target: [{ target: 'nsis', arch: ['x64'] }],
    icon,
  },
  nsis: {
    oneClick: false,
    allowToChangeInstallationDirectory: true,
    // No auto-updater, so no blockmap for differential downloads.
    differentialPackage: false,
    artifactName: '${productName}-${version}-${os}-${arch}-setup.${ext}',
  },

  mac: {
    // One image per architecture: a universal app carries Electron twice (about 1.8 times
    // the size) and would not fit the 150 MB budget.
    target: [{ target: 'dmg', arch: ['x64', 'arm64'] }],
    icon,
    category: 'public.app-category.developer-tools',
    minimumSystemVersion: '12.0',
    // Ad hoc when there is no certificate; hardened runtime and notarization only with one.
    identity: macSigning ? undefined : '-',
    hardenedRuntime: macSigning,
    notarize: macSigning,
  },

  linux: {
    target: [
      { target: 'AppImage', arch: ['x64'] },
      { target: 'deb', arch: ['x64'] },
    ],
    icon,
    category: 'Development',
    synopsis: 'Visual neural network builder for PyTorch',
    maintainer: 'NeuroLoom <artemis03976@users.noreply.github.com>',
  },
};
