import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { pythonBridge } from './dev/pythonBridge';

export default defineConfig({
  plugins: [react(), tailwindcss(), pythonBridge()],
  // Brand assets (favicon, logo SVGs) are served from the brand package as-is.
  publicDir: '../../brand/assets',
  server: { port: 5173, strictPort: true },
});
