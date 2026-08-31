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
import { api, localApi, postAction } from "../api";
import { PageHeader } from "../components/PageHeader";

const MODES = [
  "Open Language Similarity",
  "Zero-Shot Segmentation",
  "PCA Feature Visualization",
  "Traversability Estimation",
];

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
  const [result, setResult] = useState<any>(null);

  const load = async (refresh = false) => {
    const r = await api<{ images?: any[] }>(`/images?limit=10${refresh ? "&refresh=true" : ""}`, {
      timeoutMs: 15000,
    });
    const list = r.images || [];
    setImages(list);
    if (list.length && !selectedId) setSelectedId(String(list[0].id));
  };

  useEffect(() => {
    void load().catch((error) => toast({ body: String(error.message), type: "error" }));
  }, []);

  const selected = images.find((i) => String(i.id) === selectedId) || images[0];

  const analyze = async () => {
    if (!selected) return;
    setBusy(true);
    try {
      const body = await localApi("/local/vision", {
        method: "POST",
        body: JSON.stringify({
          image_url: selected.attachment_url,
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
      <HStack gap={2}>
        <Button
          label="Take photo"
          variant="primary"
          onClick={() =>
            void postAction("take_photo")
              .then(() => load(true))
              .then(() => toast({ body: "Capture queued" }))
              .catch((error) => toast({ body: String(error.message), type: "error" }))
          }
        />
        <Button label="Refresh gallery" onClick={() => void load(true)} />
      </HStack>
      {!selected ? (
        <Text color="secondary">Refresh the gallery to load FarmBot photos.</Text>
      ) : (
        <HStack gap={4}>
          <img className="camera-preview" src={selected.attachment_url} alt="FarmBot capture" />
          <Card padding={4} style={{ minWidth: 280 }}>
            <VStack gap={3}>
              <Selector
                label="Research image"
                options={images.map((i) => String(i.id))}
                value={String(selected.id)}
                onChange={setSelectedId}
                renderValue={() => `${selected.created_at || ""} · image ${selected.id}`}
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
