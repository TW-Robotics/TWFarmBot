import { useSyncExternalStore } from "react";
import { useMediaQuery } from "@astryxdesign/core/hooks";

export type DevicePref = "auto" | "tablet" | "desktop";

const KEY = "twfarmbot:device";
const listeners = new Set<() => void>();

function readPref(): DevicePref {
  const value = localStorage.getItem(KEY);
  return value === "tablet" || value === "desktop" ? value : "auto";
}

export function setDevicePref(pref: DevicePref): void {
  if (pref === "auto") localStorage.removeItem(KEY);
  else localStorage.setItem(KEY, pref);
  listeners.forEach((listener) => listener());
}

// `?device=tablet|desktop|auto` is a shortcut for the same preference (handy for
// testing a layout from a desktop browser).
const fromUrl = new URLSearchParams(window.location.search).get("device");
if (fromUrl === "tablet" || fromUrl === "desktop" || fromUrl === "auto") setDevicePref(fromUrl);

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/**
 * Touch-first layout switch. A tablet is a device whose only pointer is a finger
 * (coarse, no hover). A tablet with a trackpad or mouse attached reports a fine
 * pointer and gets the desktop layout; override that in Settings if you prefer.
 */
export function useDevice(): { isTablet: boolean; detectedTablet: boolean; pref: DevicePref } {
  const detectedTablet = useMediaQuery("(pointer: coarse) and (hover: none)");
  const pref = useSyncExternalStore(subscribe, readPref);
  return {
    isTablet: pref === "auto" ? detectedTablet : pref === "tablet",
    detectedTablet,
    pref,
  };
}
