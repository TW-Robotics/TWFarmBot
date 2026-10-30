import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  ChartBarIcon,
  MapIcon,
  CameraIcon,
  ChatBubbleLeftRightIcon,
  CircleStackIcon,
  Cog6ToothIcon,
  CubeIcon,
  CubeTransparentIcon,
  HomeIcon,
  MoonIcon,
  QueueListIcon,
  PlayIcon,
  StopCircleIcon,
  SunIcon,
} from "@heroicons/react/24/outline";
import { api, postAction } from "./api";
import { useDevice } from "./device";
import { useThemeMode, type ColorMode } from "./theme";
import { BedSidebar } from "./components/BedSidebar";
import type { GardenCell, Pose, Workspace } from "./components/GantryMap";
import { OverviewPage } from "./pages/OverviewPage";
import { TrendsPage } from "./pages/TrendsPage";
import { GardenPage } from "./pages/GardenPage";
import { MotionPage } from "./pages/MotionPage";
import { ScriptsPage } from "./pages/ScriptsPage";
import { CameraPage } from "./pages/CameraPage";
import { AssistantPage } from "./pages/AssistantPage";
import { MemoryPage } from "./pages/MemoryPage";
import { IoPage } from "./pages/IoPage";
import { DiagnosticsPage } from "./pages/DiagnosticsPage";
import { SettingsPage } from "./pages/SettingsPage";

export type Tab =
  | "overview"
  | "trends"
  | "garden"
  | "motion"
  | "scripts"
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

// `desktopOnly` tabs are hidden in the tablet layout (switch back via ?device=desktop).
const TABS: { id: Tab; label: string; icon: typeof HomeIcon; desktopOnly?: boolean }[] = [
  { id: "overview", label: "Overview", icon: HomeIcon },
  { id: "trends", label: "Trends", icon: ChartBarIcon, desktopOnly: true },
  { id: "garden", label: "Garden", icon: MapIcon, desktopOnly: true },
  { id: "motion", label: "Motion", icon: PlayIcon },
  { id: "scripts", label: "Scripts", icon: QueueListIcon },
  { id: "camera", label: "Camera", icon: CameraIcon },
  { id: "assistant", label: "Assistant", icon: ChatBubbleLeftRightIcon },
  { id: "io", label: "I/O", icon: BeakerIcon },
  { id: "memory", label: "Memory", icon: CircleStackIcon, desktopOnly: true },
  { id: "diagnostics", label: "Diagnostics", icon: CubeTransparentIcon, desktopOnly: true },
  { id: "settings", label: "Settings", icon: Cog6ToothIcon, desktopOnly: true },
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
  const { isTablet } = useDevice();
  // Tablets start with the nav collapsed to its icon rail; follow the device class
  // if it changes (e.g. the layout is switched in Settings).
  const [navCollapsed, setNavCollapsed] = useState(isTablet);
  useEffect(() => setNavCollapsed(isTablet), [isTablet]);
  const [farmbot, setFarmbot] = useState("unknown");
  const [pose, setPose] = useState<Pose>({});
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [cells, setCells] = useState<GardenCell[]>([]);
  const [cellMm, setCellMm] = useState(250);
  const [livePose, setLivePose] = useState(false);
  const poseTicket = useRef(0);

  const applyPolledPose = useCallback((xyz: Pose | undefined, ticket: number) => {
    if (ticket !== poseTicket.current) return;
    setPose(xyz || {});
  }, []);

  const applyGarden = useCallback((garden: { cells?: GardenCell[]; cell_mm?: number } | null) => {
    if (!garden) return;
    setCells(Array.isArray(garden.cells) ? garden.cells : []);
    if (typeof garden.cell_mm === "number" && Number.isFinite(garden.cell_mm)) {
      setCellMm(garden.cell_mm);
    }
  }, []);

  const selectTab = useCallback((next: Tab) => {
    setTab(next);
    const url = new URL(window.location.href);
    url.searchParams.set("tab", next);
    window.history.replaceState({}, "", url);
  }, []);

  // A hidden tab (from the URL or a layout switch) falls back to Overview.
  useEffect(() => {
    if (isTablet && TABS.some((t) => t.id === tab && t.desktopOnly)) selectTab("overview");
  }, [isTablet, tab, selectTab]);

  const refreshPose = useCallback(async () => {
    const ticket = ++poseTicket.current;
    try {
      const [position, garden] = await Promise.all([
        api<{ xyz?: Pose; workspace?: Workspace }>("/position"),
        api<{ cells?: GardenCell[]; cell_mm?: number }>("/garden").catch(() => null),
      ]);
      if (ticket !== poseTicket.current) return;
      if (position.workspace) setWorkspace(position.workspace);
      applyPolledPose(position.xyz, ticket);
      applyGarden(garden);
    } catch {
      /* keep last pose */
    }
  }, [applyGarden, applyPolledPose]);

  const refreshTelemetry = useCallback(async (refresh = false) => {
    const ticket = ++poseTicket.current;
    try {
      const positionPath = refresh ? "/position?refresh=true" : "/position";
      const [health, position, garden] = await Promise.all([
        api<{ farmbot?: string }>("/health"),
        api<{ xyz?: Pose; workspace?: Workspace }>(positionPath),
        api<{ cells?: GardenCell[]; cell_mm?: number }>("/garden").catch(() => null),
      ]);
      if (ticket !== poseTicket.current) {
        setFarmbot(health.farmbot || "?");
        return;
      }
      setFarmbot(health.farmbot || "?");
      if (position.workspace) setWorkspace(position.workspace);
      applyPolledPose(position.xyz, ticket);
      applyGarden(garden);
    } catch {
      setFarmbot("unreachable");
    }
  }, [applyGarden, applyPolledPose]);

  useEffect(() => {
    void refreshTelemetry();
  }, [refreshTelemetry]);

  useEffect(() => {
    let inFlight = false;
    const id = window.setInterval(() => {
      if (inFlight) return;
      inFlight = true;
      void refreshPose().finally(() => {
        inFlight = false;
      });
    }, 400);
    return () => window.clearInterval(id);
  }, [refreshPose]);

  const gotoMap = useCallback(
    (x: number, y: number) => {
      const z = pose.z ?? 0;
      void postAction("move", { x, y, z }, true)
        .then(() => {
          toast({ body: `→ (${x}, ${y}, ${z})` });
          return refreshTelemetry(true);
        })
        .catch((error) => {
          toast({
            body: error instanceof Error ? error.message : "Move failed",
            type: "error",
          });
        });
    },
    [pose.z, refreshTelemetry, toast],
  );

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
      case "trends":
        return <TrendsPage />;
      case "garden":
        return <GardenPage pose={pose} workspace={workspace} cells={cells} cellMm={cellMm} />;
      case "motion":
        return <MotionPage pose={pose} onMoved={() => refreshTelemetry(true)} />;
      case "scripts":
        return <ScriptsPage onMoved={() => refreshTelemetry(true)} />;
      case "camera":
        return <CameraPage />;
      case "assistant":
        return <AssistantPage onBusyChange={setLivePose} />;
      case "io":
        return <IoPage />;
      case "memory":
        return <MemoryPage />;
      case "diagnostics":
        return <DiagnosticsPage />;
      case "settings":
        return <SettingsPage farmbot={farmbot} onSaved={refreshTelemetry} />;
    }
  }, [tab, pose, farmbot, refreshTelemetry, workspace, cells, cellMm]);

  return (
    <AppShell
      height="fill"
      contentPadding={tab === "assistant" || tab === "settings" ? 0 : 5}
      variant="elevated"
      sideNav={
        <SideNav
          collapsible={{ isCollapsed: navCollapsed, onCollapsedChange: setNavCollapsed }}
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
            {TABS.filter((item) => !(isTablet && item.desktopOnly)).map((item) => (
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
      <div className={`app-body${isTablet ? " app-body--tablet" : ""}`}>
        <div className="app-main">{page}</div>
        {/* Tablet motion tab already shows the bed map inline; the sidebar
            would only squeeze the jog UI. */}
        {tab !== "settings" && tab !== "garden" && !(isTablet && tab === "motion") ? (
          <BedSidebar
            pose={pose}
            workspace={workspace}
            live={livePose}
            cells={cells}
            cellMm={cellMm}
            onGoto={gotoMap}
          />
        ) : null}
      </div>
    </AppShell>
  );
}
