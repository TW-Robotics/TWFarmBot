import { api, resireg, postAction, errorMessage, getSettings } from "../api.js";
import { h, btn, snack, page, section, card, toolbar, split, stack, metricRow, emptyState } from "../ui.js";
import * as state from "../state.js";

function parseSegmentationLabels(labels) {
  const out = {};
  for (const label of labels) {
    for (const part of String(label).split(",")) {
      const match = part.match(/(.+?)\s*\(\s*([0-9]*\.?[0-9]+)\s*%\s*\)/);
      if (match) out[match[1].trim()] = Math.round(parseFloat(match[2]) * 10) / 1000;
    }
  }
  return out;
}

async function visionChat(imageUrl, text) {
  const r = await resireg("/v1/chat/completions", {
    method: "POST",
    json: {
      model: "SimonSchwaiger/resireg_mini",
      messages: [{ role: "user", content: [
        { type: "text", text }, { type: "image_url", image_url: { url: imageUrl } },
      ] }],
    },
  });
  if (!r.ok) throw new Error(errorMessage(r));
  const content = r.body.choices[0].message.content;
  return typeof content === "string" ? JSON.parse(content) : content;
}

const MODES = [
  "Open Language Similarity",
  "Zero-Shot Segmentation",
  "PCA Feature Visualization",
  "Traversability Estimation",
];

export async function render(container) {
  const takePhotoBtn = btn("filled", {
    icon: "photo_camera",
    onClick: async () => {
      const r = await postAction("take_photo");
      if (r.ok) { snack("Capture queued"); loadGallery(); }
      else snack(errorMessage(r), { error: true });
    },
  }, "Take photo");
  const refreshBtn = btn("outlined", {
    icon: "refresh",
    onClick: () => loadGallery(true),
  }, "Refresh gallery");

  const { root, body } = page("Camera", undefined, { actions: [takePhotoBtn, refreshBtn] });
  const galleryArea = h("div", { class: "stack" });
  const resultArea = h("div", { class: "stack" });
  let aiResult = null;

  body.append(galleryArea, resultArea);
  container.append(root);

  async function loadGallery(refresh = false) {
    const r = await api("/images", { params: refresh ? { refresh: "true" } : undefined, timeoutMs: 15000 });
    if (r.ok && Array.isArray(r.body?.images)) {
      state.store.session.camera_images = r.body.images;
      state.persistSession();
      draw();
    } else snack(errorMessage(r), { error: true });
  }

  function draw() {
    const images = state.store.session?.camera_images || [];
    if (!images.length) {
      galleryArea.replaceChildren(
        emptyState("Refresh the gallery to load FarmBot photos.", { iconName: "photo_library", compact: true }));
      return;
    }
    let selected = images[0];
    const select = h("md-outlined-select", { label: "Research image", class: "field-lg" },
      images.map((img, i) => h("md-select-option", { value: String(i), selected: i === 0 },
        h("div", { slot: "headline" }, `${img.created_at || "Unknown"} · #${img.id ?? "—"}`))));
    const frame = h("img", { class: "frame", src: selected.attachment_url || "", alt: "FarmBot capture" });
    select.addEventListener("change", () => {
      selected = images[Number(select.value)];
      frame.src = selected.attachment_url || "";
    });

    let mode = MODES[0];
    const modeSelect = h("md-outlined-select", { label: "Analysis mode", class: "grow" },
      MODES.map((label, i) => h("md-select-option", { value: label, selected: i === 0 },
        h("div", { slot: "headline" }, label))));
    const promptField = h("md-outlined-text-field", { label: "Target prompt", class: "grow" });
    const classesField = h("md-outlined-text-field", { label: "Classes (comma-separated)", value: "plant, weed, soil, path", class: "grow" });
    const negativeField = h("md-outlined-text-field", { label: "Background prompt", class: "grow" });
    const clusterSlider = h("md-slider", { min: 2, max: 20, value: 6, labeled: true });
    const spinner = h("md-circular-progress", { indeterminate: true, style: "display:none" });

    function syncFields() {
      mode = modeSelect.value || MODES[0];
      const isSim = mode === "Open Language Similarity";
      const isSeg = mode === "Zero-Shot Segmentation";
      const isPca = mode === "PCA Feature Visualization";
      const isTrav = mode === "Traversability Estimation";
      promptField.style.display = isSim || isTrav ? "" : "none";
      promptField.label = isTrav ? "Traversable prompt" : "Target prompt";
      classesField.style.display = isSeg ? "" : "none";
      negativeField.style.display = isSeg || isTrav ? "" : "none";
      clusterSlider.style.display = isPca ? "" : "none";
    }
    modeSelect.addEventListener("change", syncFields);

    async function analyze() {
      spinner.style.display = "";
      try {
        const url = selected.attachment_url;
        if (mode === "Open Language Similarity") {
          const prompt = promptField.value.trim();
          if (!prompt) throw new Error("Enter a target prompt.");
          const result = await visionChat(url, prompt);
          aiResult = { images: [result.result_image_base64], captions: [`Similarity · ${prompt}`] };
        } else if (mode === "Zero-Shot Segmentation") {
          const classes = classesField.value.trim();
          if (!classes) throw new Error("Enter classes.");
          const result = await visionChat(url, `/segment: ${classes}`);
          const labels = [String(result.detected ?? ""), String(result.undetected ?? "")];
          aiResult = {
            images: (result.result_images_base64 || []).slice(0, 2),
            captions: ["Overlay", "Segmentation map"],
            labels, classScores: parseSegmentationLabels(labels),
          };
        } else if (mode === "PCA Feature Visualization") {
          const n = Number(clusterSlider.value);
          const result = await visionChat(url, `/pca: ${n}`);
          aiResult = {
            images: (result.result_images_base64 || []).slice(0, 3),
            captions: ["PCA 1", "PCA 2", "PCA 3"], nClusters: n,
          };
        } else {
          const prompt = promptField.value.trim();
          if (!prompt) throw new Error("Enter a traversable prompt.");
          const negatives = negativeField.value.trim();
          const result = await visionChat(url, `/traverse: ${prompt}${negatives ? ` vs ${negatives}` : ""}`);
          aiResult = { images: [result.result_image_base64], captions: [`Traversability · ${prompt}`] };
        }
        drawResult();
      } catch (err) {
        snack(`AI processing failed: ${err.message}`, { error: true });
      } finally {
        spinner.style.display = "none";
      }
    }

    syncFields();
    galleryArea.replaceChildren(
      select,
      split(
        h("div", {}, frame),
        card("AI analysis", stack(modeSelect, promptField, classesField, negativeField, clusterSlider,
          toolbar(btn("filled", { icon: "neurology", onClick: analyze }, "Analyze"), spinner))),
      ),
      images.length > 1 ? section("Recent captures", h("div", { class: "gallery" },
        images.slice(1, 7).map((img) => {
          const meta = img.meta || {};
          return h("figure", {}, h("img", { src: img.attachment_url || "", alt: "" }),
            h("figcaption", { class: "caption" }, `X ${meta.x ?? "—"} · Y ${meta.y ?? "—"}`));
        }))) : null,
    );
  }

  function drawResult() {
    if (!aiResult) { resultArea.replaceChildren(); return; }
    const parts = [section("Analysis result", h("div", { class: "gallery" },
      aiResult.images.map((src, i) => h("figure", {},
        h("img", { src, alt: aiResult.captions[i] || "" }),
        h("figcaption", { class: "caption" }, aiResult.captions[i] || "")))))];
    const scores = aiResult.classScores || {};
    if (Object.keys(scores).length) {
      parts.push(metricRow(Object.entries(scores).map(([cls, score]) => [cls, `${(score * 100).toFixed(1)}%`])));
    } else if (aiResult.nClusters) {
      parts.push(h("p", { class: "caption" }, `PCA with ${aiResult.nClusters} clusters`));
    }
    for (const label of aiResult.labels || []) parts.push(h("p", { class: "caption" }, label));
    resultArea.replaceChildren(...parts);
  }

  draw();
  const cameraS = getSettings().refreshCameraS;
  const timer = cameraS > 0 ? setInterval(() => loadGallery(true), cameraS * 1000) : null;
  return () => timer && clearInterval(timer);
}
