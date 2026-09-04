import { useCallback, useEffect, useMemo, useState } from "react";
import { AppShell } from "@astryxdesign/core/AppShell";
import { NavIcon } from "@astryxdesign/core/NavIcon";
import {
  SideNav,
  SideNavHeading,
  SideNavItem,
  SideNavSection,
} from "@astryxdesign/core/SideNav";
import { Button } from "@astryxdesign/core/Button";
import { Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/Stack";
import { useToast } from "@astryxdesign/core/Toast";
import { CubeIcon } from "@heroicons/react/24/outline";
import {
  CameraIcon,
  ChatBubbleLeftRightIcon,
  Cog6ToothIcon,
  CubeTransparentIcon,
  HomeIcon,
  MoonIcon,
  PlayIcon,
  QueueListIcon,
  SunIcon,
  WrenchScrewdriverIcon,
} from "@heroicons/react/24/outline";
import { api, postAction } from "./api";
import { useThemeMode } from "./theme";
import { OverviewPage } from "./pages/OverviewPage";
import { MotionPage } from "./pages/MotionPage";
import { CameraPage } from "./pages/CameraPage";
import { IoPage } from "./pages/IoPage";
import { AssistantPage } from "./pages/AssistantPage";
import { HistoryPage } from "./pages/HistoryPage";
import { DiagnosticsPage } from "./pages/DiagnosticsPage";
import { SettingsPage } from "./pages/SettingsPage";

export type Tab =
  | "overview"
  | "motion"
  | "camera"
  | "io"
  | "assistant"
  | "history"
  | "diagnostics"
  | "settings";

const TABS: { id: Tab; label: string; icon: typeof HomeIcon }[] = [
  { id: "overview", label: "Overview", icon: HomeIcon },
  { id: "motion", label: "Motion", icon: PlayIcon },
  { id: "camera", label: "Camera", icon: CameraIcon },
  { id: "io", label: "I/O", icon: WrenchScrewdriverIcon },
  { id: "assistant", label: "Assistant", icon: ChatBubbleLeftRightIcon },
  { id: "history", label: "History", icon: QueueListIcon },
  { id: "diagnostics", label: "Diagnostics", icon: CubeTransparentIcon },
  { id: "settings", label: "Settings", icon: Cog6ToothIcon },
];

function tabFromUrl(): Tab {
  const raw = (new URLSearchParams(window.location.search).get("tab") || "").toLowerCase();
  if (raw === "sensors" || raw === "operations") return "io";
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
      case "io":
        return <IoPage />;
      case "assistant":
        return <AssistantPage />;
      case "history":
        return <HistoryPage />;
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
            <VStack gap={2}>
              <Text type="supporting" color="secondary">
                FarmBot · {farmbot}
              </Text>
              <Button
                label={mode === "dark" ? "Light mode" : "Dark mode"}
                icon={mode === "dark" ? <SunIcon /> : <MoonIcon />}
                variant="secondary"
                width="100%"
                onClick={toggle}
              />
              <Button label="ESTOP" variant="primary" width="100%" onClick={() => void estop()} />
            </VStack>
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
