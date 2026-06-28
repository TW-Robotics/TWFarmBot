#!/usr/bin/env python3
"""
CLI port of the ReSiReg Playground backend for Raspberry Pi 5.

Modes (matching the playground tabs):
  pca                  - PCA visualizations of dense patch features
  language_similarity  - heatmap of prompt similarity, optional negatives
  zero_shot_segmentation - multi-class segmentation from class names
  traversability       - sharp traversable vs background map
  image_similarity     - similarity to a second query image +/- text

Raw features are always accessible via --export-raw.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import psutil
import requests
import torch
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoModel

MODEL_ID = "SimonSchwaiger/resireg_mini"

Mode = Literal[
    "pca",
    "language_similarity",
    "zero_shot_segmentation",
    "traversability",
    "image_similarity",
]

MODE_PCA = "pca"
MODE_LANGUAGE_SIM = "language_similarity"
MODE_ZERO_SHOT_SEG = "zero_shot_segmentation"
MODE_TRAVERSABILITY = "traversability"
MODE_IMG_SIM = "image_similarity"


# ------------------------------------------------------------------
# Pi 5 setup
# ------------------------------------------------------------------
def setup_pi5():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


# ------------------------------------------------------------------
# Pi 5 power & latency benchmark
# ------------------------------------------------------------------
@dataclass
class SystemSample:
    timestamp: float
    total_w: float = 0.0
    core_w: float = 0.0
    soc_w: float = 0.0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_used_mb: float = 0.0
    temp_c: float = 0.0


def read_pmic() -> dict[str, dict[str, float]]:
    """Sample Pi 5 PMIC voltages/currents and return rail power dict."""
    try:
        out = subprocess.check_output(
            ["vcgencmd", "pmic_read_adc"], text=True, timeout=2
        )
    except Exception:
        return {}

    currents: dict[str, float] = {}
    voltages: dict[str, float] = {}
    for line in out.splitlines():
        m = re.search(r"(\S+)_(A|V)\s+(current|volt)\(\d+\)=([0-9.]+)(A|V)", line)
        if not m:
            continue
        name, kind, _, value, _ = m.groups()
        value = float(value)
        if kind == "A":
            currents[name] = value
        else:
            voltages[name] = value

    rails: dict[str, dict[str, float]] = {}
    for name in currents:
        v = voltages.get(name, 0.0)
        a = currents[name]
        rails[name] = {"v": v, "a": a, "w": v * a}
    return rails


def compute_power(rails: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    core_names = {"VDD_CORE"}
    soc_names = {
        "VDD_CORE",
        "DDR_VDD2",
        "DDR_VDDQ",
        "1V1_SYS",
        "0V8_SW",
        "1V8_SYS",
        "3V3_SYS",
    }
    total_w = core_w = soc_w = 0.0
    for name, data in rails.items():
        w = data["w"]
        total_w += w
        if name in core_names:
            core_w += w
        if name in soc_names:
            soc_w += w
    return total_w, core_w, soc_w


def read_cpu_temp() -> float:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return 0.0


def read_system() -> SystemSample:
    rails = read_pmic()
    total_w, core_w, soc_w = compute_power(rails)
    vm = psutil.virtual_memory()
    return SystemSample(
        timestamp=time.perf_counter(),
        total_w=total_w,
        core_w=core_w,
        soc_w=soc_w,
        cpu_percent=psutil.cpu_percent(interval=None),
        ram_percent=vm.percent,
        ram_used_mb=vm.used / (1024 * 1024),
        temp_c=read_cpu_temp(),
    )


class SystemSampler:
    """Background sampler for power, CPU, RAM, and temperature."""

    def __init__(self, interval_s: float = 0.1):
        self.interval = interval_s
        self._samples: list[SystemSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self):
        # Prime cpu_percent so the first sample is meaningful
        _ = psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            self._samples.append(read_system())
            time.sleep(self.interval)

    def start(self):
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[SystemSample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self._samples


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _format_w(w: float) -> str:
    return f"{w:.2f} W"


def _format_j(j: float) -> str:
    return f"{j:.3f} J"


def _format_ms(s: float) -> str:
    return f"{s * 1000:.1f} ms"


def _format_mb(mb: float) -> str:
    return f"{mb:.0f} MB"


def _format_c(c: float) -> str:
    return f"{c:.1f}°C"


def benchmark_report(
    name: str, times: list[float], samples: list[SystemSample], idle: SystemSample
):
    s = _stats(times)
    avg_w = statistics.mean([p.total_w for p in samples]) if samples else 0.0
    avg_core_w = statistics.mean([p.core_w for p in samples]) if samples else 0.0
    avg_soc_w = statistics.mean([p.soc_w for p in samples]) if samples else 0.0
    avg_cpu = statistics.mean([p.cpu_percent for p in samples]) if samples else 0.0
    avg_ram_pct = statistics.mean([p.ram_percent for p in samples]) if samples else 0.0
    avg_ram_mb = statistics.mean([p.ram_used_mb for p in samples]) if samples else 0.0
    avg_temp = statistics.mean([p.temp_c for p in samples]) if samples else 0.0
    delta_w = avg_w - idle.total_w
    energy_j = delta_w * s["mean"]

    print(f"\n=== Benchmark: {name} ===")
    print(
        f"  Latency   mean={_format_ms(s['mean'])}  median={_format_ms(s['median'])}  "
        f"min={_format_ms(s['min'])}  max={_format_ms(s['max'])}  stdev={_format_ms(s['stdev'])}"
    )
    print(
        f"  Power     total={_format_w(avg_w)}  core={_format_w(avg_core_w)}  "
        f"soc={_format_w(avg_soc_w)}  (idle={_format_w(idle.total_w)}  delta={_format_w(delta_w)})"
    )
    print(f"  CPU       {avg_cpu:.1f}%")
    print(f"  RAM       {avg_ram_pct:.1f}% used ({_format_mb(avg_ram_mb)})")
    print(f"  Temp      {_format_c(avg_temp)}")
    print(f"  Energy    {name} = {_format_j(energy_j)} (incremental, excludes idle)")
    if name.lower().startswith("image"):
        print(f"  Throughput ~{1.0 / s['mean']:.2f} frames/s")
    return {
        "name": name,
        "latency_s": s,
        "power_total_w": avg_w,
        "power_core_w": avg_core_w,
        "power_soc_w": avg_soc_w,
        "cpu_percent": avg_cpu,
        "ram_percent": avg_ram_pct,
        "ram_used_mb": avg_ram_mb,
        "temp_c": avg_temp,
        "idle": idle,
        "delta_w": delta_w,
        "energy_j": energy_j,
    }


def sample_idle(duration_s: float = 1.0) -> SystemSample:
    print(f"Sampling idle system state for {duration_s:.1f} s...")
    sampler = SystemSampler(interval_s=0.1)
    sampler.start()
    time.sleep(duration_s)
    samples = sampler.stop()
    if not samples:
        return SystemSample(timestamp=time.perf_counter())
    return SystemSample(
        timestamp=statistics.mean([s.timestamp for s in samples]),
        total_w=statistics.mean([s.total_w for s in samples]),
        core_w=statistics.mean([s.core_w for s in samples]),
        soc_w=statistics.mean([s.soc_w for s in samples]),
        cpu_percent=statistics.mean([s.cpu_percent for s in samples]),
        ram_percent=statistics.mean([s.ram_percent for s in samples]),
        ram_used_mb=statistics.mean([s.ram_used_mb for s in samples]),
        temp_c=statistics.mean([s.temp_c for s in samples]),
    )


def run_benchmark(
    model,
    image: Image.Image,
    prompts: list[str],
    *,
    apply_resireg_lite: bool = True,
    runs: int = 5,
):
    """Benchmark text encode, image encode, and full forward pass."""
    print("\n" + "=" * 60)
    print("BENCHMARK MODE")
    print("=" * 60)

    idle = sample_idle(duration_s=1.0)
    print(
        f"Idle: power={_format_w(idle.total_w)}  CPU={idle.cpu_percent:.1f}%  "
        f"RAM={idle.ram_percent:.1f}%  temp={_format_c(idle.temp_c)}"
    )

    rgb = image.convert("RGB")
    results = []

    text_times: list[float] = []
    img_times: list[float] = []

    def run_text():
        model.encode_text(prompts)

    def run_image():
        model.encode_image(rgb, apply_resireg_lite=apply_resireg_lite)

    # Manually collect times so summary can use raw lists
    for _ in range(runs):
        sampler = SystemSampler(interval_s=0.1)
        sampler.start()
        t0 = time.perf_counter()
        run_text()
        t1 = time.perf_counter()
        text_samples = sampler.stop()
        text_times.append(t1 - t0)
    results.append(benchmark_report("Text encoding", text_times, text_samples, idle))

    for _ in range(runs):
        sampler = SystemSampler(interval_s=0.1)
        sampler.start()
        t0 = time.perf_counter()
        run_image()
        t1 = time.perf_counter()
        img_samples = sampler.stop()
        img_times.append(t1 - t0)
    results.append(benchmark_report("Image encoding", img_times, img_samples, idle))

    full_times: list[float] = []
    full_samples: list[SystemSample] = []
    for i in range(runs):
        sampler = SystemSampler(interval_s=0.1)
        sampler.start()
        t0 = time.perf_counter()
        _, _ = encode(model, rgb, prompts, apply_resireg_lite=apply_resireg_lite)
        t1 = time.perf_counter()
        full_samples.extend(sampler.stop())
        full_times.append(t1 - t0)
    print(
        f"  -> first full-forward latency (cold / time-to-first-output): {_format_ms(full_times[0])}"
    )
    results.append(
        benchmark_report("Full forward pass", full_times, full_samples, idle)
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Time-to-first-output (cold full forward): {_format_ms(full_times[0])}")
    print(f"  Steady full-forward latency: {_format_ms(statistics.median(full_times))}")
    print(f"  Text-only latency: {_format_ms(statistics.median(text_times))}")
    print(f"  Image-only latency: {_format_ms(statistics.median(img_times))}")
    print(f"  Incremental energy per frame: {_format_j(results[2]['energy_j'])}")
    print(
        f"  Equiv. throughput: ~{1.0 / statistics.median(img_times):.2f} image frames/s"
    )
    print("=" * 60)
    return results


def run_stream_benchmark(
    model,
    image: Image.Image,
    prompts: list[str] | None,
    *,
    apply_resireg_lite: bool = True,
    frames: int = 30,
    video_path: str | None = None,
    start_frame: int = 0,
    stride: int = 1,
):
    """Run a video-style stream benchmark. If video_path is given, read real frames."""
    print("\n" + "=" * 60)
    print("STREAM / VIDEO BENCHMARK")
    print("=" * 60)

    idle = sample_idle(duration_s=1.0)
    print(
        f"Idle: power={_format_w(idle.total_w)}  CPU={idle.cpu_percent:.1f}%  "
        f"RAM={idle.ram_percent:.1f}%  temp={_format_c(idle.temp_c)}"
    )

    text = None
    if prompts:
        with torch.no_grad():
            text = model.encode_text(prompts).cpu()
        print(f"Pre-cached text embeddings for {len(prompts)} prompts.")

    sampler = SystemSampler(interval_s=0.1)
    frame_times: list[float] = []
    processed_frames = 0

    if video_path is not None:
        print(
            f"Streaming up to {frames} frames from {video_path} (start={start_frame}, stride={stride})..."
        )
        gen = video_frame_generator(video_path, start_frame=start_frame, stride=stride)
    else:
        print(f"Streaming {frames} repetitions of the same image...")
        rgb = image.convert("RGB")
        gen = ((i, rgb) for i in range(frames))

    sampler.start()
    stream_t0 = time.perf_counter()
    for idx, frame in gen:
        if processed_frames >= frames:
            break
        t0 = time.perf_counter()
        dense = model.encode_image(frame, apply_resireg_lite=apply_resireg_lite)
        if text is not None:
            _ = cosine_similarity_map(dense, text)
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)
        processed_frames += 1
        if processed_frames % 10 == 0:
            print(f"  processed {processed_frames}/{frames} frames...")
    stream_t1 = time.perf_counter()
    samples = sampler.stop()

    total_s = stream_t1 - stream_t0
    fps = processed_frames / total_s if total_s > 0 else 0.0
    avg_frame = statistics.mean(frame_times) if frame_times else 0.0
    median_frame = statistics.median(frame_times) if frame_times else 0.0
    max_frame = max(frame_times) if frame_times else 0.0
    min_frame = min(frame_times) if frame_times else 0.0

    avg_w = statistics.mean([s.total_w for s in samples]) if samples else 0.0
    avg_core_w = statistics.mean([s.core_w for s in samples]) if samples else 0.0
    avg_cpu = statistics.mean([s.cpu_percent for s in samples]) if samples else 0.0
    avg_ram = statistics.mean([s.ram_percent for s in samples]) if samples else 0.0
    avg_temp = statistics.mean([s.temp_c for s in samples]) if samples else 0.0
    delta_w = avg_w - idle.total_w
    energy_per_frame = delta_w * avg_frame

    print(f"\n=== Stream results ({processed_frames} frames) ===")
    print(f"  Total time: {total_s:.2f} s")
    print(f"  Average FPS: {fps:.2f}")
    print(
        f"  Frame latency  mean={_format_ms(avg_frame)}  median={_format_ms(median_frame)}  "
        f"min={_format_ms(min_frame)}  max={_format_ms(max_frame)}"
    )
    print(
        f"  Power   total={_format_w(avg_w)}  core={_format_w(avg_core_w)}  "
        f"delta={_format_w(delta_w)}"
    )
    print(f"  CPU {avg_cpu:.1f}%  RAM {avg_ram:.1f}%  Temp {_format_c(avg_temp)}")
    print(f"  Incremental energy per frame: {_format_j(energy_per_frame)}")
    print("=" * 60)


# ------------------------------------------------------------------
# Model cache / helpers
# ------------------------------------------------------------------
@dataclass
class ImageEmbedCache:
    fingerprint: str
    dense: torch.Tensor  # [1, C, H, W] on CPU
    width: int
    height: int


def _fingerprint(image: Image.Image) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


def load_model():
    print(f"Loading {MODEL_ID}...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).eval()
    model._load_preprocessors()
    return model


def encode(
    model,
    image: Image.Image | None,
    prompts: list[str] | None,
    *,
    apply_resireg_lite: bool = True,
):
    """Unified encode entry point. Returns (dense, text) on CPU."""
    if image is None and not prompts:
        raise ValueError("encode() requires image and/or prompts.")
    dense = text = None
    if image is not None:
        dense = model.encode_image(
            image.convert("RGB"), apply_resireg_lite=apply_resireg_lite
        ).cpu()
    if prompts:
        text = model.encode_text(prompts).cpu()
    return dense, text


def resolve_embeddings(
    model,
    image: Image.Image,
    prompts: list[str] | None,
    cache: ImageEmbedCache | None,
    *,
    apply_resireg_lite: bool = True,
):
    """Use cached image embedding if the upload is unchanged."""
    rgb = image.convert("RGB")
    fingerprint = _fingerprint(rgb)
    image_hit = cache is not None and cache.fingerprint == fingerprint

    need_image = not image_hit
    need_text = bool(prompts)

    if need_image:
        dense, text = encode(
            model,
            rgb,
            prompts if need_text else None,
            apply_resireg_lite=apply_resireg_lite,
        )
        cache = ImageEmbedCache(
            fingerprint=fingerprint, dense=dense, width=rgb.width, height=rgb.height
        )
        return dense, text, cache

    dense = cache.dense if cache is not None else None
    if need_text:
        _, text = encode(model, None, prompts, apply_resireg_lite=apply_resireg_lite)
        return dense, text, cache
    return dense, None, cache


# ------------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------------
def upscale_map(values: torch.Tensor | np.ndarray, image: Image.Image) -> np.ndarray:
    """Upsample a 2D patch grid to the image size (bilinear)."""
    if isinstance(values, np.ndarray):
        t = torch.from_numpy(values).float()
    else:
        t = values.detach().float().cpu()
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(0)
    elif t.ndim != 4:
        raise ValueError(f"Expected a 2D-4D map, got shape {tuple(t.shape)}")
    up = F.interpolate(
        t, size=(image.height, image.width), mode="bilinear", align_corners=False
    )
    return up[0, 0].numpy()


def normalize_similarity(sim: torch.Tensor, *, min_range: float = 0.2) -> torch.Tensor:
    sim_range = torch.clamp(sim.max() - sim.min(), min=min_range)
    return (sim - sim.min()) / (sim_range + 1e-8)


def render_heatmap_overlay(
    image: Image.Image,
    heatmap: torch.Tensor | np.ndarray,
    *,
    title: str | None = None,
    cmap: str = "jet",
    alpha: float = 0.45,
) -> Image.Image:
    image = image.convert("RGB")
    heat = upscale_map(heatmap, image)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(heat, cmap=cmap, alpha=alpha)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def cosine_similarity_map(dense: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """dense [1,C,H,W], text [K,C] -> [K,H,W]."""
    dense_n = F.normalize(dense, dim=1)
    text_n = F.normalize(text, dim=-1)
    return torch.einsum("bchw,kc->bkhw", dense_n, text_n)[0]


# ------------------------------------------------------------------
# PCA modes
# ------------------------------------------------------------------
def _spatial_from_dense(dense: torch.Tensor) -> np.ndarray:
    return dense[0].permute(1, 2, 0).numpy().astype(np.float32)


def _pil_resize_nearest(arr: np.ndarray, width: int, height: int) -> Image.Image:
    resample = getattr(Image, "Resampling", Image).NEAREST
    return Image.fromarray(arr).resize((width, height), resample)


def vis_pca_rgb(spatial: np.ndarray, width: int, height: int) -> Image.Image:
    h, w, d = spatial.shape
    feat = spatial.reshape(-1, d)
    pca = PCA(n_components=3, whiten=True)
    rgb = pca.fit_transform(feat).reshape(h, w, 3)
    rgb = 1.0 / (1.0 + np.exp(-2.0 * rgb))
    arr = (rgb * 255).clip(0, 255).astype(np.uint8)
    return _pil_resize_nearest(arr, width, height)


def vis_pca_component(spatial: np.ndarray, width: int, height: int) -> Image.Image:
    h, w, d = spatial.shape
    feat = spatial.reshape(-1, d)
    comp = PCA(n_components=1).fit_transform(feat).reshape(h, w)
    comp = (comp - comp.min()) / (comp.max() - comp.min() + 1e-8)
    colored = plt.colormaps["inferno"](comp)[:, :, :3]
    arr = (colored * 255).clip(0, 255).astype(np.uint8)
    return _pil_resize_nearest(arr, width, height)


def vis_pca_kmeans(
    spatial: np.ndarray, width: int, height: int, n_clusters: int = 6
) -> Image.Image:
    sp_h, sp_w, d = spatial.shape
    feat = torch.from_numpy(spatial.reshape(-1, d))
    km = KMeans(n_clusters=n_clusters, max_iter=20)
    km.fit(feat)
    scores = -torch.cdist(feat, km.centroids).reshape(sp_h, sp_w, n_clusters)
    scores_t = scores.permute(2, 0, 1).unsqueeze(0).float()
    scores_up = F.interpolate(
        scores_t, size=(height, width), mode="bilinear", align_corners=False
    )
    labels = scores_up[0].permute(1, 2, 0).numpy().argmax(axis=-1)
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, n_clusters))[:, :3]
    seg = palette[labels].astype(np.float32)
    arr = (seg * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ------------------------------------------------------------------
# Segmentation mode
# ------------------------------------------------------------------
def vis_semseg(
    image: Image.Image,
    dense: torch.Tensor,
    text: torch.Tensor,
    class_list: list[str],
    *,
    text_neg: torch.Tensor | None = None,
    temperature: float = 1.0,
):
    from matplotlib.patches import Patch

    n = len(class_list)
    h, w = image.height, image.width

    sims = cosine_similarity_map(dense, text)
    if text_neg is not None:
        neg_sim = cosine_similarity_map(dense, text_neg)[0]
        sims = sims - neg_sim.unsqueeze(0)
    probs = F.softmax(sims / temperature, dim=0)

    probs_up = F.interpolate(
        probs.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=False
    )[0]
    labels = probs_up.argmax(dim=0).numpy()
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(n, 2)))[:n, :3]

    seg_rgb = palette[labels].astype(np.float32)
    mask_pil = Image.fromarray((seg_rgb * 255).clip(0, 255).astype(np.uint8))

    label_rgba = np.zeros((h, w, 4), dtype=np.float32)
    label_rgba[..., :3] = seg_rgb
    label_rgba[..., 3] = 0.6

    unique_ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    unique_ids_sorted = unique_ids[order]
    counts_sorted = counts[order]
    total = counts_sorted.sum()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(label_rgba)
    ax.axis("off")
    legend_handles = [
        Patch(facecolor=palette[cid], label=class_list[cid])
        for cid in unique_ids_sorted
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=9,
        framealpha=0.8,
        ncol=max(1, len(legend_handles) // 8),
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    overlay_pil = Image.open(buf).convert("RGB")

    detected: list[str] = []
    minor: list[str] = []
    for cid, cnt in zip(unique_ids_sorted, counts_sorted):
        pct = cnt / total * 100
        entry = f"{class_list[cid]} ({pct:.1f}%)"
        (detected if pct >= 2.0 else minor).append(entry)
    absent = [f"{class_list[i]} (0.0%)" for i in range(n) if i not in unique_ids_sorted]
    return overlay_pil, mask_pil, ", ".join(detected), ", ".join(minor + absent)


# ------------------------------------------------------------------
# Raw export helper
# ------------------------------------------------------------------
def export_raw(
    base_path: Path,
    dense: torch.Tensor | None,
    text: torch.Tensor | None,
    sims: torch.Tensor | None,
    probs: torch.Tensor | None,
    heatmap: np.ndarray | None,
    prompt_list: list[str],
):
    raw_dir = base_path.parent / (base_path.stem + "_raw")
    raw_dir.mkdir(exist_ok=True)
    if dense is not None:
        np.save(raw_dir / "dense_embeds.npy", dense.cpu().numpy())
    if text is not None:
        np.save(raw_dir / "text_embeds.npy", text.cpu().numpy())
    if sims is not None:
        np.save(raw_dir / "raw_cosine_similarity.npy", sims.cpu().numpy())
    if probs is not None:
        np.save(raw_dir / "softmax_probs.npy", probs.cpu().numpy())
    if heatmap is not None:
        np.save(raw_dir / "heatmap.npy", heatmap)
    with open(raw_dir / "prompts.txt", "w") as f:
        f.write("\n".join(prompt_list))
    print(f"Saved raw tensors to {raw_dir}/")


# ------------------------------------------------------------------
# Mode runners
# ------------------------------------------------------------------
def run_pca(model, image: Image.Image, args, cache: ImageEmbedCache | None):
    dense, _, cache = resolve_embeddings(
        model, image, None, cache, apply_resireg_lite=args.apply_lite
    )
    spatial = _spatial_from_dense(dense)
    w, h = image.width, image.height
    pca_rgb = vis_pca_rgb(spatial, w, h)
    pca_comp = vis_pca_component(spatial, w, h)
    pca_km = vis_pca_kmeans(spatial, w, h, n_clusters=args.n_clusters)

    out_base = Path(args.output)
    pca_rgb.save(out_base.with_name(out_base.stem + "_pca_rgb.png"))
    pca_comp.save(out_base.with_name(out_base.stem + "_pca_component.png"))
    pca_km.save(out_base.with_name(out_base.stem + "_pca_kmeans.png"))
    print(f"Saved PCA visualizations next to {args.output}")
    if args.export_raw:
        export_raw(out_base, dense, None, None, None, None, ["pca"])
    return cache


def run_language_sim(model, image: Image.Image, args, cache: ImageEmbedCache | None):
    if not args.prompt:
        raise ValueError("--prompt is required for language_similarity")
    pos = args.prompt.strip()
    neg_list = [n.strip() for n in args.negatives.split(",") if n.strip()]
    all_prompts = [pos] + neg_list

    dense, text, cache = resolve_embeddings(
        model, image, all_prompts, cache, apply_resireg_lite=args.apply_lite
    )
    sims = cosine_similarity_map(dense, text)
    pos_sim = sims[0]
    if neg_list:
        neg_sim = sims[1:].mean(dim=0)
        sim = pos_sim - neg_sim
    else:
        sim = pos_sim
    sim = normalize_similarity(sim)

    neg_label = f" - ({', '.join(neg_list)})" if neg_list else ""
    overlay = render_heatmap_overlay(image, sim, title=f"'{pos}'{neg_label}")
    out = Path(args.output)
    overlay.save(out)
    print(f"Saved overlay to {out}")
    if args.export_raw:
        export_raw(out, dense, text, sims, None, upscale_map(sim, image), all_prompts)
    return cache


def run_zero_shot_seg(model, image: Image.Image, args, cache: ImageEmbedCache | None):
    if not args.classes:
        raise ValueError("--classes is required for zero_shot_segmentation")
    class_list = [c.strip() for c in args.classes.split(",") if c.strip()]
    neg_prompt = args.seg_negative.strip()
    all_prompts = class_list + ([neg_prompt] if neg_prompt else [])

    dense, text_all, cache = resolve_embeddings(
        model, image, all_prompts, cache, apply_resireg_lite=args.apply_lite
    )
    if neg_prompt:
        text_neg = text_all[-1:]
        text = text_all[:-1]
    else:
        text_neg = None
        text = text_all

    overlay, mask, detected, undetected = vis_semseg(
        image, dense, text, class_list, text_neg=text_neg, temperature=1.0
    )
    out = Path(args.output)
    overlay.save(out.with_name(out.stem + "_overlay.png"))
    mask.save(out.with_name(out.stem + "_mask.png"))
    print(f"Detected: {detected}")
    print(f"Undetected/minor: {undetected}")
    if args.export_raw:
        sims = cosine_similarity_map(dense, text)
        export_raw(out, dense, text, sims, None, None, all_prompts)
    return cache


def run_traversability(model, image: Image.Image, args, cache: ImageEmbedCache | None):
    if not args.prompt:
        raise ValueError("--prompt is required for traversability")
    trav = args.prompt.strip()
    bg_list = [n.strip() for n in args.negatives.split(",") if n.strip()]
    all_prompts = [trav] + bg_list

    dense, text, cache = resolve_embeddings(
        model, image, all_prompts, cache, apply_resireg_lite=args.apply_lite
    )
    sims = cosine_similarity_map(dense, text)
    probs = F.softmax(sims / 0.2, dim=0)
    trav_map = normalize_similarity(probs[0])

    bg_label = f" vs ({', '.join(bg_list)})" if bg_list else ""
    overlay = render_heatmap_overlay(
        image, trav_map, title=f"Traversability: '{trav}'{bg_label}"
    )
    out = Path(args.output)
    overlay.save(out)
    print(f"Saved overlay to {out}")
    if args.export_raw:
        export_raw(
            out, dense, text, sims, probs, upscale_map(trav_map, image), all_prompts
        )
    return cache


def run_image_similarity(
    model, image: Image.Image, args, cache: ImageEmbedCache | None
):
    if args.prompt_image is None:
        raise ValueError("--prompt-image is required for image_similarity")
    prompt_rgb = args.prompt_image.convert("RGB")
    pos_list = [p.strip() for p in args.prompt.split(",") if p.strip()]
    neg_list = [n.strip() for n in args.negatives.split(",") if n.strip()]
    all_text = pos_list + neg_list if (pos_list or neg_list) else None

    dense, _, cache = resolve_embeddings(
        model, image, None, cache, apply_resireg_lite=args.apply_lite
    )
    prompt_dense, text_all = encode(
        model, prompt_rgb, all_text, apply_resireg_lite=args.apply_lite
    )

    img_query = F.normalize(prompt_dense[0].mean(dim=(1, 2)).unsqueeze(0), dim=-1)
    all_queries = (
        torch.cat([img_query, text_all], dim=0) if text_all is not None else img_query
    )
    sims = cosine_similarity_map(dense, all_queries)
    sim = sims[0]
    if pos_list:
        sim = sim + sims[1 : 1 + len(pos_list)].mean(dim=0)
    if neg_list:
        sim = sim - sims[1 + len(pos_list) :].mean(dim=0)
    sim = normalize_similarity(sim)

    overlay = render_heatmap_overlay(image, sim, title="Image-to-Image Similarity")
    out = Path(args.output)
    overlay.save(out)
    print(f"Saved overlay to {out}")
    if args.export_raw:
        export_raw(
            out,
            dense,
            all_queries,
            sims,
            None,
            upscale_map(sim, image),
            ["image_query"] + (all_text or []),
        )
    return cache


# ------------------------------------------------------------------
# Video helpers
# ------------------------------------------------------------------
def load_first_frame(video_path: str) -> Image.Image:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read first frame from: {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def video_frame_generator(video_path: str, start_frame: int = 0, stride: int = 1):
    """Yield (frame_index, PIL RGB Image) tuples from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx - start_frame) % stride == 0:
            yield idx, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ReSiReg-Mini CLI for Raspberry Pi 5")
    parser.add_argument("--image", default=None, help="Input image path or URL.")
    parser.add_argument(
        "--video",
        default=None,
        help="Input video file path. If given, frames are read from the video instead of --image.",
    )
    parser.add_argument(
        "--video-start-frame",
        type=int,
        default=0,
        help="Start frame for video processing / stream benchmark.",
    )
    parser.add_argument(
        "--video-stride", type=int, default=1, help="Process every Nth video frame."
    )
    parser.add_argument(
        "--mode",
        default="language_similarity",
        choices=[
            MODE_PCA,
            MODE_LANGUAGE_SIM,
            MODE_ZERO_SHOT_SEG,
            MODE_TRAVERSABILITY,
            MODE_IMG_SIM,
        ],
        help="Which playground mode to run.",
    )
    parser.add_argument(
        "--prompt", default="", help="Positive prompt / traversable prompt."
    )
    parser.add_argument(
        "--negatives",
        default="",
        help="Comma-separated negative prompts (language_sim, traversability, img_sim).",
    )
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated class names for zero_shot_segmentation.",
    )
    parser.add_argument(
        "--seg-negative",
        default="",
        help="Optional negative prompt subtracted in segmentation.",
    )
    parser.add_argument(
        "--prompt-image", default=None, help="Query image for image_similarity mode."
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=6,
        help="Number of k-means clusters for PCA mode.",
    )
    parser.add_argument(
        "--output",
        default="/home/farmbot/resireg_pi5/outputs/resireg_output.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--apply-lite",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Apply ReSiReg-Lite residual similarity.",
    )
    parser.add_argument(
        "--export-raw",
        action="store_true",
        help="Save raw dense/text features and similarity tensors.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="After the mode run, benchmark text/image/full-forward "
        "latency and PMIC power on the Pi 5.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=5,
        help="Number of benchmark iterations (default 5).",
    )
    parser.add_argument(
        "--benchmark-stream-frames",
        type=int,
        default=0,
        help="If >0, run a video-style stream benchmark on this many "
        "frames and report FPS/power/CPU/RAM/temp.",
    )
    args = parser.parse_args()

    setup_pi5()
    print(
        f"PyTorch {torch.__version__}, oneDNN={torch.backends.mkldnn.is_available()}, "
        f"threads={torch.get_num_threads()}"
    )

    model = load_model()

    # Load input image or first video frame
    if args.video is not None and args.image is not None:
        raise ValueError("Use either --image or --video, not both.")
    if args.video is not None:
        print(f"Loading first frame from video: {args.video}")
        image = load_first_frame(args.video)
    elif args.image is not None:
        if args.image.startswith("http://") or args.image.startswith("https://"):
            image = Image.open(requests.get(args.image, stream=True, timeout=30).raw)
        else:
            image = Image.open(args.image)
    else:
        raise ValueError("Either --image or --video is required.")
    image = image.convert("RGB")

    prompt_image = None
    if args.prompt_image:
        if args.prompt_image.startswith("http://") or args.prompt_image.startswith(
            "https://"
        ):
            prompt_image = Image.open(
                requests.get(args.prompt_image, stream=True, timeout=30).raw
            )
        else:
            prompt_image = Image.open(args.prompt_image)
        prompt_image = prompt_image.convert("RGB")
    args.prompt_image = prompt_image

    cache = None
    start = time.perf_counter()
    if args.mode == MODE_PCA:
        cache = run_pca(model, image, args, cache)
    elif args.mode == MODE_LANGUAGE_SIM:
        cache = run_language_sim(model, image, args, cache)
    elif args.mode == MODE_ZERO_SHOT_SEG:
        cache = run_zero_shot_seg(model, image, args, cache)
    elif args.mode == MODE_TRAVERSABILITY:
        cache = run_traversability(model, image, args, cache)
    elif args.mode == MODE_IMG_SIM:
        cache = run_image_similarity(model, image, args, cache)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    print(f"Total mode runtime: {time.perf_counter() - start:.3f}s")

    if args.benchmark or args.benchmark_stream_frames > 0:
        # Build a representative prompt list from the mode's inputs
        neg_list = [n.strip() for n in args.negatives.split(",") if n.strip()]
        if args.mode == MODE_ZERO_SHOT_SEG:
            prompt_list = [c.strip() for c in args.classes.split(",") if c.strip()]
            if args.seg_negative.strip():
                prompt_list.append(args.seg_negative.strip())
        elif args.mode == MODE_IMG_SIM:
            pos_list = [p.strip() for p in args.prompt.split(",") if p.strip()]
            prompt_list = pos_list + neg_list
        else:
            prompt_list = (
                [args.prompt.strip()] if args.prompt.strip() else []
            ) + neg_list

        if not prompt_list:
            prompt_list = ["wooden bridge"]  # fallback

        if args.benchmark:
            run_benchmark(
                model,
                image,
                prompt_list,
                apply_resireg_lite=args.apply_lite,
                runs=args.benchmark_runs,
            )
        if args.benchmark_stream_frames > 0:
            run_stream_benchmark(
                model,
                image,
                prompt_list,
                apply_resireg_lite=args.apply_lite,
                frames=args.benchmark_stream_frames,
                video_path=args.video,
                start_frame=args.video_start_frame,
                stride=args.video_stride,
            )


if __name__ == "__main__":
    main()
