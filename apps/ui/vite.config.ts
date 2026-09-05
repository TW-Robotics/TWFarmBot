import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  root: ".",
  publicDir: "web/public",
  build: {
    outDir: "src/twfarmbot_ui/static",
    emptyOutDir: true,
  },
  server: {
    host: true,
    port: 8501,
    proxy: {
      "/api": {
        target: process.env.TWFB_API_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
