import { useEffect, useRef, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { NumberInput } from "@astryxdesign/core/NumberInput";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, apiUrl, fmt, postAction } from "../api";
import { JogPad } from "../components/JogPad";
import { Metric, PageHeader } from "../components/PageHeader";

function imageSrc(url: string | undefined, version?: string): string {
  if (!url) return "";
  const base =
    url.startsWith("http://") || url.startsWith("https://")
      ? url
      : `${apiUrl()}${url.startsWith("/") ? url : `/${url}`}`;
  if (!version) return base;
  const sep = base.includes("?") ? "&" : "?";
  return `${base}${sep}v=${encodeURIComponent(version)}`;
}

type Pose = { x: number; y: number; z: number };

export function DiagnosticsPage() {
  const toast = useToast();
  const [state, setState] = useState<any>({});
  const [pose, setPose] = useState<Partial<Pose>>({});
  const [offsetX, setOffsetX] = useState(-100);
  const [offsetY, setOffsetY] = useState(0);
  const [offsetZ, setOffsetZ] = useState(0);
  const [alignDx, setAlignDx] = useState(0);
  const [alignDy, setAlignDy] = useState(0);
  const [alignRot, setAlignRot] = useState(0);
  const [nirPose, setNirPose] = useState<Pose | null>(null);
  const [frozenNirId, setFrozenNirId] = useState("");
  const [frozenRededgeId, setFrozenRededgeId] = useState("");
  const [live, setLive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<any>(null);
  const liveRef = useRef(false);
  const inFlightRef = useRef(false);
  const alignRef = useRef({ dx_px: 0, dy_px: 0, rotation_deg: 0 });

  const imageAlign = () => ({
    dx_px: alignDx,
    dy_px: alignDy,
    rotation_deg: alignRot,
  });

  useEffect(() => {
    alignRef.current = imageAlign();
  }, [alignDx, alignDy, alignRot]);

  const refreshPose = async () => {
    try {
      const position = await api<{ xyz?: Partial<Pose> }>("/position?refresh=true", {
        timeoutMs: 15000,
      });
      const xyz = position.xyz || {};
      setPose({
        x: Number(xyz.x) || 0,
        y: Number(xyz.y) || 0,
        z: Number(xyz.z) || 0,
      });
    } catch {
      /* keep last known pose */
    }
  };

  useEffect(() => {
    void refreshPose();
    void api<{
      band_separation_mm?: Pose;
      image_align?: { dx_px?: number; dy_px?: number; rotation_deg?: number };
    }>("/spectral/calibration")
      .then((body) => {
        const offset = body.band_separation_mm || {};
        if (Number.isFinite(Number(offset.x))) setOffsetX(Number(offset.x));
        if (Number.isFinite(Number(offset.y))) setOffsetY(Number(offset.y));
        if (Number.isFinite(Number(offset.z))) setOffsetZ(Number(offset.z));
        const align = body.image_align || {};
        if (Number.isFinite(Number(align.dx_px))) setAlignDx(Number(align.dx_px));
        if (Number.isFinite(Number(align.dy_px))) setAlignDy(Number(align.dy_px));
        if (Number.isFinite(Number(align.rotation_deg))) {
          setAlignRot(Number(align.rotation_deg));
        }
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    liveRef.current = live;
    if (!live || !frozenNirId) return;

    let cancelled = false;
    const tick = async () => {
      while (liveRef.current && !cancelled) {
        if (inFlightRef.current) {
          await new Promise((resolve) => window.setTimeout(resolve, 200));
          continue;
        }
        inFlightRef.current = true;
        try {
          const body = await api("/spectral/live-ndre", {
            method: "POST",
            body: JSON.stringify({
              nir_artifact_id: frozenNirId,
              image_align: alignRef.current,
            }),
            timeoutMs: 60000,
          });
          if (!cancelled && liveRef.current) {
            setResult(body);
            if (body?.rededge?.artifact_id) {
              setFrozenRededgeId(String(body.rededge.artifact_id));
            }
            if (body?.gantry_mm) setPose(body.gantry_mm);
          }
        } catch (error) {
          if (!cancelled) {
            setLive(false);
            toast({
              body: error instanceof Error ? error.message : "Live NDRE failed",
              type: "error",
            });
          }
          break;
        } finally {
          inFlightRef.current = false;
        }
      }
    };
    void tick();
    return () => {
      cancelled = true;
    };
  }, [live, frozenNirId, toast]);

  useEffect(() => {
    if (live || !frozenNirId || !frozenRededgeId) return;
    const timer = window.setTimeout(() => {
      void api("/spectral/analyze-pair", {
        method: "POST",
        body: JSON.stringify({
          nir_artifact_id: frozenNirId,
          rededge_artifact_id: frozenRededgeId,
          image_align: imageAlign(),
        }),
        timeoutMs: 30000,
      })
        .then((body) => setResult(body))
        .catch(() => undefined);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [alignDx, alignDy, alignRot, frozenNirId, frozenRededgeId, live]);

  const load = async () => {
    try {
      const r = await api<{ state?: any }>("/status");
      setState(r.state || {});
      await refreshPose();
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Read failed", type: "error" });
    }
  };

  const freezeNir = async () => {
    setBusy(true);
    setLive(false);
    try {
      const res = await postAction("capture", { band: "nir" }, true);
      const artifactId = String(res?.action?.params?.artifact_id || "");
      if (!artifactId) throw new Error("NIR capture returned no artifact id");
      setFrozenNirId(artifactId);
      setResult({
        nir: {
          artifact_id: artifactId,
          attachment_url: `/captures/${artifactId}/nir`,
        },
        metrics: null,
      });
      toast({ body: "NIR frozen. Start live NDRE, then jog around." });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "NIR freeze failed",
        type: "error",
      });
    } finally {
      setBusy(false);
    }
  };

  const saveOffset = async () => {
    setSaving(true);
    try {
      await api("/spectral/calibration", {
        method: "PUT",
        body: JSON.stringify({
          x: offsetX,
          y: offsetY,
          z: offsetZ,
          image_align: imageAlign(),
        }),
      });
      toast({
        body: `Saved gantry (${offsetX}, ${offsetY}, ${offsetZ}) + image align`,
      });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Could not save calibration",
        type: "error",
      });
    } finally {
      setSaving(false);
    }
  };

  const markNirPose = async () => {
    setBusy(true);
    try {
      const position = await api<{ xyz?: Pose }>("/position?refresh=true", {
        timeoutMs: 15000,
      });
      const xyz = position.xyz || { x: 0, y: 0, z: 0 };
      const pose = {
        x: Number(xyz.x) || 0,
        y: Number(xyz.y) || 0,
        z: Number(xyz.z) || 0,
      };
      setNirPose(pose);
      const res = await postAction("capture", { band: "nir" }, true);
      const artifactId = String(res?.action?.params?.artifact_id || "");
      if (artifactId) {
        setFrozenNirId(artifactId);
        setResult({
          nir: {
            artifact_id: artifactId,
            attachment_url: `/captures/${artifactId}/nir`,
          },
        });
      }
      toast({
        body: `NIR pose marked at X ${pose.x}, Y ${pose.y}. Jog red-edge into place, then finish.`,
      });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Could not mark NIR pose",
        type: "error",
      });
    } finally {
      setBusy(false);
    }
  };

  const finishOffsetFromPose = async () => {
    if (!nirPose) {
      toast({ body: "Mark the NIR pose first", type: "error" });
      return;
    }
    setBusy(true);
    try {
      const position = await api<{ xyz?: Pose }>("/position?refresh=true", {
        timeoutMs: 15000,
      });
      const xyz = position.xyz || { x: 0, y: 0, z: 0 };
      const next = {
        x: Math.round((Number(xyz.x) || 0) - nirPose.x),
        y: Math.round((Number(xyz.y) || 0) - nirPose.y),
        z: Math.round((Number(xyz.z) || 0) - nirPose.z),
      };
      setOffsetX(next.x);
      setOffsetY(next.y);
      setOffsetZ(next.z);
      toast({
        body: `Offset set to X ${next.x}, Y ${next.y}, Z ${next.z}. Save when happy.`,
      });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Could not finish offset",
        type: "error",
      });
    } finally {
      setBusy(false);
    }
  };

  const captureNdrePair = async () => {
    setBusy(true);
    setLive(false);
    try {
      const body = await api("/spectral/capture-pair", {
        method: "POST",
        body: JSON.stringify({
          return_to_start: true,
          offset_mm: { x: offsetX, y: offsetY, z: offsetZ },
          image_align: imageAlign(),
        }),
        timeoutMs: 180000,
      });
      setResult(body);
      if (body?.nir?.artifact_id) setFrozenNirId(String(body.nir.artifact_id));
      if (body?.rededge?.artifact_id) {
        setFrozenRededgeId(String(body.rededge.artifact_id));
      }
      toast({ body: body?.metrics?.summary || "Spectral pair captured" });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Spectral capture failed",
        type: "error",
      });
    } finally {
      setBusy(false);
    }
  };

  const info = state.informational_settings || {};
  const axes = state.location_data?.axis_states || {};
  const pins = state.pins || {};
  const metrics = result?.metrics;

  return (
    <VStack gap={4}>
      <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="Diagnostics" />

      <Card padding={4}>
        <VStack gap={3}>
          <Text weight="semibold">Live NDRE</Text>
          <Text type="supporting" color="secondary">
            Freeze a NIR frame, start live NDRE, then jog below. Tune image-space
            dx / dy / rotation so NIR and red-edge line up, then save.
          </Text>
          <div className="metric-grid">
            <Metric label="X · mm" value={fmt(pose.x)} />
            <Metric label="Y · mm" value={fmt(pose.y)} />
            <Metric label="Z · mm" value={fmt(pose.z)} />
          </div>
          <HStack gap={2} wrap>
            <Button
              label="Freeze NIR"
              variant="primary"
              isDisabled={busy}
              isLoading={busy && !live}
              onClick={() => void freezeNir()}
            />
            <Button
              label={live ? "Stop live NDRE" : "Start live NDRE"}
              isDisabled={busy || !frozenNirId}
              onClick={() => setLive((value) => !value)}
            />
            <Button
              label="Capture NDRE pair"
              isDisabled={busy || live}
              onClick={() => void captureNdrePair()}
            />
          </HStack>
          {frozenNirId ? (
            <Text type="supporting" color="secondary">
              Frozen NIR: {frozenNirId}
              {live ? " · live" : ""}
            </Text>
          ) : null}
          {metrics?.summary ? <Text>{metrics.summary}</Text> : null}
          <HStack gap={3} wrap>
            {result?.nir?.attachment_url ? (
              <VStack gap={1}>
                <img
                  className="camera-preview"
                  src={imageSrc(result.nir.attachment_url, result.nir.artifact_id)}
                  alt="Frozen NIR"
                />
                <Text type="supporting">Frozen NIR</Text>
              </VStack>
            ) : null}
            {result?.rededge?.attachment_url ? (
              <VStack gap={1}>
                <img
                  className="camera-preview"
                  src={imageSrc(
                    result.rededge.attachment_url,
                    result.rededge.artifact_id,
                  )}
                  alt="Live red-edge"
                />
                <Text type="supporting">Live red-edge</Text>
              </VStack>
            ) : null}
            {metrics?.ndre_preview ? (
              <VStack gap={1}>
                <img
                  className="camera-preview"
                  src={
                    String(metrics.ndre_preview).startsWith("data:")
                      ? metrics.ndre_preview
                      : imageSrc(
                          metrics.ndre_preview,
                          `${frozenNirId || ""}-${Date.now()}`,
                        )
                  }
                  alt="Live NDRE"
                />
                <Text type="supporting">
                  NDRE mean {metrics.ndre?.mean ?? "—"}
                </Text>
              </VStack>
            ) : null}
            {metrics?.align_preview ? (
              <VStack gap={1}>
                <img
                  className="camera-preview"
                  src={metrics.align_preview}
                  alt="Align overlay"
                />
                <Text type="supporting">Align overlay (NIR=red, RE=green)</Text>
              </VStack>
            ) : null}
          </HStack>
        </VStack>
      </Card>

      <Card padding={4}>
        <VStack gap={3}>
          <Text weight="semibold">Image-space align</Text>
          <Text type="supporting" color="secondary">
            Warp red-edge onto the frozen NIR in pixels. Yellow in the overlay
            means good overlap. Values are applied live and saved with calib.
          </Text>
          <HStack gap={2} wrap>
            <NumberInput label="dx (px)" value={alignDx} onChange={setAlignDx} />
            <NumberInput label="dy (px)" value={alignDy} onChange={setAlignDy} />
            <NumberInput
              label="rotation (°)"
              value={alignRot}
              onChange={setAlignRot}
            />
          </HStack>
        </VStack>
      </Card>

      <Card padding={4}>
        <VStack gap={3}>
          <Text weight="semibold">Motion</Text>
          <Text type="supporting" color="secondary">
            Jog while live NDRE is running — no need to leave Diagnostics.
          </Text>
          <JogPad pose={pose} onMoved={refreshPose} showGoto />
        </VStack>
      </Card>

      <Card padding={4}>
        <VStack gap={3}>
          <Text weight="semibold">Band offset calibration</Text>
          <Text type="supporting" color="secondary">
            Mark NIR, jog until red-edge matches that patch, finish to fill the
            offset, then save.
          </Text>
          <HStack gap={2} wrap>
            <Button
              label="1. Mark NIR pose"
              isDisabled={busy || live}
              onClick={() => void markNirPose()}
            />
            <Button
              label="2. Finish offset here"
              isDisabled={busy || live || !nirPose}
              onClick={() => void finishOffsetFromPose()}
            />
          </HStack>
          {nirPose ? (
            <Text type="supporting" color="secondary">
              NIR marked at X {nirPose.x}, Y {nirPose.y}, Z {nirPose.z}
            </Text>
          ) : null}
          <HStack gap={2} wrap>
            <NumberInput label="Offset X (mm)" value={offsetX} onChange={setOffsetX} />
            <NumberInput label="Offset Y (mm)" value={offsetY} onChange={setOffsetY} />
            <NumberInput label="Offset Z (mm)" value={offsetZ} onChange={setOffsetZ} />
          </HStack>
          <Button
            label="Save calibration"
            isDisabled={busy || saving || live}
            isLoading={saving}
            onClick={() => void saveOffset()}
          />
        </VStack>
      </Card>

      <Card padding={4}>
        <VStack gap={3}>
          <Text weight="semibold">Controller status</Text>
          <Button label="Load /status" variant="primary" onClick={() => void load()} />
          {!info.controller_version && !Object.keys(pins).length ? (
            <Text color="secondary">Load /status to fetch diagnostic state.</Text>
          ) : (
            <>
              <div className="metric-grid">
                <Metric label="Controller" value={String(info.controller_version ?? "—")} />
                <Metric label="Firmware" value={String(info.firmware_version ?? "—")} />
                <Metric label="Wi-Fi" value={`${fmt(info.wifi_level_percent)}%`} />
                <Metric label="Uptime" value={`${fmt(info.uptime)} s`} />
              </div>
              <HStack gap={3}>
                <Card padding={3}>
                  <Text>
                    CPU {info.cpu_usage ?? "—"}% · Memory {info.memory_usage ?? "—"}% · Disk{" "}
                    {info.disk_usage ?? "—"}%
                  </Text>
                </Card>
                <Card padding={3}>
                  <Text>
                    Axis X {axes.x ?? "—"} · Y {axes.y ?? "—"} · Z {axes.z ?? "—"}
                  </Text>
                </Card>
              </HStack>
            </>
          )}
        </VStack>
      </Card>
    </VStack>
  );
}
