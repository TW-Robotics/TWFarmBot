import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ToastViewport } from "@astryxdesign/core/Toast";
import { App } from "./App";
import { ThemeModeProvider } from "./theme";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeModeProvider>
      <ToastViewport position="bottomEnd">
        <App />
      </ToastViewport>
    </ThemeModeProvider>
  </StrictMode>,
);
