import { useState } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { IconButton } from "@astryxdesign/core/IconButton";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import { HStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { ChevronRightIcon, MapIcon } from "@heroicons/react/24/outline";
import { useDevice } from "../device";
import {
  GantryMap,
  LAYER_LEGEND,
  fmt,
  layerCounts,
  type GardenCell,
  type MapLayer,
  type Pose,
  type Workspace,
} from "./GantryMap";

const STORAGE = "twfarmbot:bedSidebar:v1";
const ICON = { width: 16, height: 16 };

export function BedSidebar({
  pose,
  workspace,
  live = false,
  cells = [],
  cellMm = 250,
  onGoto,
}: {
  pose: Pose;
  workspace?: Workspace | null;
  live?: boolean;
  cells?: GardenCell[];
  cellMm?: number;
  onGoto?: (x: number, y: number) => void;
}) {
  const { isTablet } = useDevice();
  // Remembered per device class; with no saved choice, tablets start collapsed.
  const storageKey = isTablet ? `${STORAGE}:tablet` : STORAGE;
  const [saved, setSaved] = useState<{ key: string; open: boolean } | null>(null);
  const stored = localStorage.getItem(storageKey);
  const open = saved?.key === storageKey ? saved.open : stored === null ? !isTablet : stored !== "0";
  const setOpen = (next: boolean) => {
    localStorage.setItem(storageKey, next ? "1" : "0");
    setSaved({ key: storageKey, open: next });
  };
  const [layer, setLayer] = useState<MapLayer>("vigor");
  const counts = layerCounts(cells, layer);

  if (!open) {
    return (
      <aside className="bed-sidebar bed-sidebar--collapsed">
        <IconButton
          label="Show bed"
          icon={<MapIcon style={ICON} />}
          variant="ghost"
          onClick={() => setOpen(true)}
        />
      </aside>
    );
  }

  const present = Object.entries(LAYER_LEGEND[layer]).filter(([key]) => counts[key]);

  return (
    <aside className="bed-sidebar">
      <HStack justify="between" vAlign="center">
        <HStack gap={2} vAlign="center">
          <Text weight="semibold">Bed</Text>
          {live ? <Badge variant="success" label="Live" /> : null}
        </HStack>
        <HStack gap={1} vAlign="center">
          <SegmentedControl
            label="Bed map layer"
            value={layer}
            onChange={(value) => setLayer(value as MapLayer)}
            size="sm"
          >
            <SegmentedControlItem value="vigor" label="Vigor" />
            <SegmentedControlItem value="cover" label="Cover" />
          </SegmentedControl>
          <IconButton
            label="Hide bed"
            icon={<ChevronRightIcon style={ICON} />}
            variant="ghost"
            size="sm"
            onClick={() => setOpen(false)}
          />
        </HStack>
      </HStack>
      <GantryMap
        pose={pose}
        workspace={workspace}
        live={live}
        cells={cells}
        cellMm={cellMm}
        layer={layer}
        onGoto={onGoto}
      />
      <div className="bed-footer">
        <div className="bed-legend">
          {present.length ? (
            present.map(([key, entry]) => (
              <span key={key} className="bed-legend-item">
                <span className="bed-legend-swatch" style={{ background: entry.color }} />
                <Text type="supporting" color="secondary">
                  {entry.label} · {counts[key]}
                </Text>
              </span>
            ))
          ) : (
            <Text type="supporting" color="secondary">
              No scans yet
            </Text>
          )}
        </div>
        <Text type="supporting" color="secondary">
          <span className="bed-pose">
            X {fmt(pose.x)} · Y {fmt(pose.y)} · Z {fmt(pose.z)} mm
          </span>
        </Text>
      </div>
    </aside>
  );
}
