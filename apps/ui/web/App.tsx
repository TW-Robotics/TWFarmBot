import { useCallback, useEffect, useMemo, useState } from "react";
import { AppShell } from "@astryxdesign/core/AppShell";
import { NavIcon } from "@astryxdesign/core/NavIcon";
import {
  SideNav,
  SideNavHeading,
  SideNavItem,
  SideNavSection,
  useSideNavCollapse,
} from "@astryxdesign/core/SideNav";
import { Button } from "@astryxdesign/core/Button";
import { IconButton } from "@astryxdesign/core/IconButton";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/Stack";
import { useToast } from "@astryxdesign/core/Toast";
import {
  BeakerIcon,
  CameraIcon,
  ChatBubbleLeftRightIcon,
  CircleStackIcon,
  Cog6ToothIcon,
  CubeIcon,
  CubeTransparentIcon,
  HomeIcon,
  MoonIcon,
  PlayIcon,
  StopCircleIcon,
  SunIcon,
} from "@heroicons/react/24/outline";
import { api, postAction } from "./api";
import { useThemeMode, type ColorMode } from "./theme";
import { OverviewPage } from "./pages/OverviewPage";
import { MotionPage } from "./pages/MotionPage";
import { CameraPage } from "./pages/CameraPage";
import { AssistantPage } from "./pages/AssistantPage";
import { MemoryPage } from "./pages/MemoryPage";
import { IoPage } from "./pages/IoPage";
import { DiagnosticsPage } from "./pages/DiagnosticsPage";
import { SettingsPage } from "./pages/SettingsPage";

export type Tab =
  | "overview"
  | "motion"
  | "camera"
  | "assistant"
  | "io"
  | "memory"
  | "diagnostics"
  | "settings";

const NAV_ICON = { width: 16, height: 16 };

function SideNavFooter({
  farmbot,
  mode,
  onToggleTheme,
  onEstop,
}: {
  farmbot: string;
  mode: ColorMode;
  onToggleTheme: () => void;
  onEstop: () => void;
}) {
  const { isCollapsed } = useSideNavCollapse();
  if (isCollapsed) return null;
  return (
    <VStack gap={2}>
      <Text type="supporting" color="secondary">
        FarmBot · {farmbot}
      </Text>
      <Button
        label={mode === "dark" ? "Light mode" : "Dark mode"}
        icon={mode === "dark" ? <SunIcon /> : <MoonIcon />}
        variant="secondary"
        width="100%"
        onClick={onToggleTheme}
      />
      <Button label="ESTOP" variant="primary" width="100%" onClick={onEstop} />
    </VStack>
  );
}

function SideNavFooterIcons({
  mode,
  onToggleTheme,
  onEstop,
}: {
  mode: ColorMode;
  onToggleTheme: () => void;
  onEstop: () => void;
}) {
  const { isCollapsed } = useSideNavCollapse();
  if (!isCollapsed) return null;
  return (
    <>
      <IconButton
        label={mode === "dark" ? "Light mode" : "Dark mode"}
        icon={mode === "dark" ? <SunIcon style={NAV_ICON} /> : <MoonIcon style={NAV_ICON} />}
        variant="secondary"
        onClick={onToggleTheme}
      />
      <IconButton
        label="ESTOP"
        icon={<StopCircleIcon style={NAV_ICON} />}
        variant="primary"
        onClick={onEstop}
      />
    </>
  );
}

const TABS: { id: Tab; label: string; icon: typeof HomeIcon }[] = [
  { id: "overview", label: "Overview", icon: HomeIcon },
  { id: "motion", label: "Motion", icon: PlayIcon },
  { id: "camera", label: "Camera", icon: CameraIcon },
  { id: "assistant", label: "Assistant", icon: ChatBubbleLeftRightIcon },
  { id: "io", label: "I/O", icon: BeakerIcon },
  { id: "memory", label: "Memory", icon: CircleStackIcon },
  { id: "diagnostics", label: "Diagnostics", icon: CubeTransparentIcon },
  { id: "settings", label: "Settings", icon: Cog6ToothIcon },
];

function tabFromUrl(): Tab {
  const raw = (new URLSearchParams(window.location.search).get("tab") || "").toLowerCase();
  if (raw === "garden") return "overview";
  return TABS.some((t) => t.id === raw) ? (raw as Tab) : "overview";
}

export function App() {
  const toast = useToast();
  const { mode, toggle } = useThemeMode();
  const [tab, setTab] = useState<Tab>(tabFromUrl);
  const [farmbot, setFarmbot] = useState("unknown");
  const [pose, setPose] = useState<{ x?: number; y?: number; z?: number }>({});

  const selectTab = useCallback((next: Tab) => {
    setTab(next);
    const url = new URL(window.location.href);
    url.searchParams.set("tab", next);
    window.history.replaceState({}, "", url);
  }, []);

  const refreshTelemetry = useCallback(async (refresh = false) => {
    try {
      const positionPath = refresh ? "/position?refresh=true" : "/position";
      const [health, position] = await Promise.all([
        api<{ farmbot?: string }>("/health"),
        api<{ xyz?: { x?: number; y?: number; z?: number } }>(positionPath),
      ]);
      setFarmbot(health.farmbot || "?");
      setPose(position.xyz || {});
    } catch {
      setFarmbot("unreachable");
    }
  }, []);

  useEffect(() => {
    void refreshTelemetry();
    const id = window.setInterval(() => void refreshTelemetry(), 15000);
    return () => window.clearInterval(id);
  }, [refreshTelemetry]);

  const estop = async () => {
    try {
      await postAction("e_stop");
      toast({ body: "ESTOP sent" });
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "ESTOP failed", type: "error" });
    }
  };

  const page = useMemo(() => {
    switch (tab) {
      case "overview":
        return <OverviewPage pose={pose} farmbot={farmbot} />;
      case "motion":
        return <MotionPage pose={pose} onMoved={() => refreshTelemetry(true)} />;
      case "camera":
        return <CameraPage />;
      case "assistant":
        return <AssistantPage />;
      case "io":
        return <IoPage />;
      case "memory":
        return <MemoryPage />;
      case "diagnostics":
        return <DiagnosticsPage />;
      case "settings":
        return <SettingsPage farmbot={farmbot} onSaved={refreshTelemetry} />;
    }
  }, [tab, pose, farmbot, refreshTelemetry]);

  return (
    <AppShell
      height="fill"
      contentPadding={tab === "assistant" ? 0 : 5}
      variant="elevated"
      sideNav={
        <SideNav
          collapsible
          header={
            <SideNavHeading
              icon={<NavIcon icon={<CubeIcon style={{ width: 16, height: 16 }} />} />}
              heading="TWFarmBot"
              headingHref="?tab=overview"
            />
          }
          footer={
            <SideNavFooter
              farmbot={farmbot}
              mode={mode}
              onToggleTheme={toggle}
              onEstop={() => void estop()}
            />
          }
          footerIcons={
            <SideNavFooterIcons mode={mode} onToggleTheme={toggle} onEstop={() => void estop()} />
          }
        >
          <SideNavSection title="Field robotics" isHeaderHidden>
            {TABS.map((item) => (
              <SideNavItem
                key={item.id}
                label={item.label}
                icon={item.icon}
                isSelected={tab === item.id}
                onClick={() => selectTab(item.id)}
              />
            ))}
          </SideNavSection>
        </SideNav>
      }
    >
      {page}
    </AppShell>
  );
}
