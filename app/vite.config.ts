import { defineConfig } from "vite";

export default defineConfig({
  clearScreen: false,
  server: { port: 1420, strictPort: true, watch: { ignored: ["**/src-tauri/**"] } },
  // public/demo holds a recorded session for browser previews only; never ship it.
  build: { target: "es2022", chunkSizeWarningLimit: 2500, copyPublicDir: false },
});
