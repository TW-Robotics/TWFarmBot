import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Theme } from "@astryxdesign/core/theme";
import { neutralTheme } from "@astryxdesign/theme-neutral/built";

export type ColorMode = "light" | "dark";

const STORAGE_KEY = "twfb-theme";

function initialMode(): ColorMode {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* storage unavailable — fall through to default */
  }
  return "dark";
}

interface ThemeModeValue {
  mode: ColorMode;
  toggle: () => void;
}

const ThemeModeContext = createContext<ThemeModeValue>({ mode: "dark", toggle: () => {} });

export function useThemeMode(): ThemeModeValue {
  return useContext(ThemeModeContext);
}

export function ThemeModeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<ColorMode>(initialMode);

  const toggle = useCallback(() => {
    setMode((prev) => {
      const next: ColorMode = prev === "dark" ? "light" : "dark";
      try {
        window.localStorage.setItem(STORAGE_KEY, next);
      } catch {
        /* storage unavailable — mode still applies for this session */
      }
      return next;
    });
  }, []);

  const value = useMemo(() => ({ mode, toggle }), [mode, toggle]);

  return (
    <ThemeModeContext.Provider value={value}>
      <Theme theme={neutralTheme} mode={mode}>
        {children}
      </Theme>
    </ThemeModeContext.Provider>
  );
}
