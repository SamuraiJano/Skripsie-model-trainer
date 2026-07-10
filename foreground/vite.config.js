import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  build: {
    cssCodeSplit: true,
    sourcemap: false,
    target: 'es2020',
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/react') || id.includes('node_modules/react-dom')) {
            return 'react';
          }
          if (id.includes('node_modules/lucide-react')) {
            return 'icons';
          }
          return undefined;
        },
      },
    },
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: false,
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: false,
  },
});
