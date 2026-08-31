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
    port: 8501,
  },
});
