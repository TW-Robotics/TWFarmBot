import { useEffect, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { NumberInput } from "@astryxdesign/core/NumberInput";
import { Selector } from "@astryxdesign/core/Selector";
import { TextArea } from "@astryxdesign/core/TextArea";
import { TextInput } from "@astryxdesign/core/TextInput";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api, apiUrl, localApi, postAction } from "../api";
import { PageHeader } from "../components/PageHeader";

const MODES = [
  "Open Language Similarity",
  "Zero-Shot Segmentation",
  "PCA Feature Visualization",
  "Traversability Estimation",
];

const USB_BANDS = [
  { id: "rgb", label: "RGB (FHD)" },
  { id: "nir", label: "NIR (DMK …551)" },
  { id: "rededge", label: "Red edge (DMK …552)" },
] as const;

function bandLabel(band: string | undefined): string {
  if (!band) return "";
  return USB_BANDS.find((b) => b.id === band)?.label ?? band;
}

function imageSrc(url: string | undefined, version?: string): string {
  if (!url) return "";
  const base = url.startsWith("http://") || url.startsWith("https://")
    ? url
    : `${apiUrl()}${url.startsWith("/") ? url : `/${url}`}`;
  if (!version) return base;
  const sep = base.includes("?") ? "&" : "?";
  return `${base}${sep}v=${encodeURIComponent(version)}`;
}

export function CameraPage() {
  const toast = useToast();
  const [images, setImages] = useState<any[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [mode, setMode] = useState(MODES[0]);
  const [prompt, setPrompt] = useState("");
  const [classes, setClasses] = useState("plant, weed, soil, path");
  const [negative, setNegative] = useState("");
  const [clusters, setClusters] = useState(6);
  const [busy, setBusy] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [result, setResult] = useState<any>(null);

  const load = async (refresh = false, preferId?: string) => {
    const r = await api<{ images?: any[] }>(`/images?limit=20${refresh ? "&refresh=true" : ""}`, {
      timeoutMs: 15000,
    });
    const list = r.images || [];
    setImages(list);
    if (!list.length) {
      setSelectedId("");
      return list;
    }
    const preferred = preferId && list.some((i) => String(i.id) === preferId) ? preferId : "";
    const nextId = preferred || String(list[0].id);
    setSelectedId(nextId);
    return list;
  };

  useEffect(() => {
    void load().catch((error) => toast({ body: String(error.message), type: "error" }));
  }, []);

  const refreshGallery = async (preferId?: string) => {
    setRefreshing(true);
    try {
      await load(true, preferId);
      toast({ body: "Gallery refreshed" });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Gallery refresh failed",
        type: "error",
      });
    } finally {
      setRefreshing(false);
    }
  };

  const selected = images.find((i) => String(i.id) === selectedId) || images[0];

  const captureBand = async (band: string) => {
    setCapturing(true);
    try {
      const res = await postAction("capture", { band }, true);
      const artifactId = res?.action?.params?.artifact_id;
      const preferId = artifactId ? `${artifactId}-${band}` : undefined;
      await load(true, preferId);
      toast({ body: `${band.toUpperCase()} capture saved` });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Capture failed",
        type: "error",
      });
    } finally {
      setCapturing(false);
    }
  };

  const captureAll = async () => {
    setCapturing(true);
    try {
      let lastId: string | undefined;
      for (const band of USB_BANDS) {
        const res = await postAction("capture", { band: band.id }, true);
        const artifactId = res?.action?.params?.artifact_id;
        if (artifactId) lastId = `${artifactId}-${band.id}`;
      }
      await load(true, lastId);
      toast({ body: "All USB cameras captured" });
    } catch (error) {
      toast({
        body: error instanceof Error ? error.message : "Capture failed",
        type: "error",
      });
    } finally {
      setCapturing(false);
    }
  };

  const analyze = async () => {
    if (!selected) return;
    setBusy(true);
    try {
      const body = await localApi("/local/vision", {
        method: "POST",
        body: JSON.stringify({
          image_url: imageSrc(selected.attachment_url, selected.created_at),
          mode,
          prompt,
          classes,
          negative,
          negatives: negative,
          n_clusters: clusters,
        }),
      });
      setResult(body);
    } catch (error) {
      toast({ body: error instanceof Error ? error.message : "Analysis failed", type: "error" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <VStack gap={4}>
      <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="Camera" />

      <Card padding={4}>
        <VStack gap={3}>
          <Text weight="semibold">USB payload cameras</Text>
          <Text type="supporting" color="secondary">
            RGB, NIR, and red-edge cameras via udev symlinks (/dev/camera-*).
            Live NDRE and band-offset tools live on Diagnostics.
          </Text>
          <HStack gap={2} wrap>
            {USB_BANDS.map((band) => (
              <Button
                key={band.id}
                label={band.label}
                isDisabled={capturing}
                onClick={() => void captureBand(band.id)}
              />
            ))}
            <Button
              label="Capture all"
              variant="primary"
              isDisabled={capturing}
              isLoading={capturing}
              onClick={() => void captureAll()}
            />
          </HStack>
        </VStack>
      </Card>

      <HStack gap={2}>
        <Button
          label="Take photo (RGB)"
          variant="primary"
          onClick={() =>
            void postAction("take_photo", {}, true)
              .then(() => load(true))
              .then(() => toast({ body: "RGB photo saved" }))
              .catch((error) => toast({ body: String(error.message), type: "error" }))
          }
        />
        <Button
          label="Refresh gallery"
          isLoading={refreshing}
          onClick={() => void refreshGallery()}
        />
      </HStack>

      {!selected ? (
        <Text color="secondary">Capture from a USB camera or take a photo to populate the gallery.</Text>
      ) : (
        <HStack gap={4}>
          <img
            key={selected.id}
            className="camera-preview"
            src={imageSrc(selected.attachment_url, selected.created_at)}
            alt="Camera capture"
          />
          <Card padding={4} style={{ minWidth: 280 }}>
            <VStack gap={3}>
              <Selector
                label="Gallery"
                options={images.map((i) => String(i.id))}
                value={String(selected.id)}
                onChange={setSelectedId}
                renderValue={() => {
                  const band = selected.band ? ` · ${bandLabel(selected.band)}` : "";
                  return `${selected.created_at || ""}${band} · ${selected.id}`;
                }}
              />
              <Selector label="Mode" options={MODES} value={mode} onChange={setMode} />
              {mode === "Open Language Similarity" && (
                <TextInput label="Target prompt" value={prompt} onChange={setPrompt} />
              )}
              {mode === "Zero-Shot Segmentation" && (
                <>
                  <TextInput label="Classes" value={classes} onChange={setClasses} />
                  <TextInput label="Background prompt" value={negative} onChange={setNegative} />
                </>
              )}
              {mode === "PCA Feature Visualization" && (
                <NumberInput label="K-means clusters" value={clusters} onChange={setClusters} min={2} max={20} />
              )}
              {mode === "Traversability Estimation" && (
                <>
                  <TextInput label="Traversable prompt" value={prompt} onChange={setPrompt} />
                  <TextArea label="Background prompts" value={negative} onChange={setNegative} />
                </>
              )}
              <Button
                label="Analyze selected image"
                variant="primary"
                isLoading={busy}
                onClick={() => void analyze()}
              />
            </VStack>
          </Card>
        </HStack>
      )}
      {result?.images?.length ? (
        <HStack gap={3}>
          {result.images.map((img: { url: string; caption: string }) => (
            <VStack key={img.caption} gap={1}>
              <img className="camera-preview" src={img.url} alt={img.caption} />
              <Text type="supporting">{img.caption}</Text>
            </VStack>
          ))}
        </HStack>
      ) : null}
    </VStack>
  );
}
