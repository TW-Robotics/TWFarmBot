#!/usr/bin/env python3
"""
OpenAI-compatible HTTP server for ReSiReg-Mini on Raspberry Pi 5.

Runs under uvicorn. Not a generative text model: chat completions return
a JSON payload with the dense similarity / segmentation result and a
base64-encoded output image.

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions
  POST /v1/embeddings
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModel

from .resireg_cli import (
    MODEL_ID, cosine_similarity_map, normalize_similarity, render_heatmap_overlay,
    vis_semseg, setup_pi5, resolve_embeddings, _spatial_from_dense, vis_pca_rgb,
    vis_pca_component, vis_pca_kmeans,
)


# ------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------
setup_pi5()
print(f"Loading {MODEL_ID}...")
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).eval()
model._load_preprocessors()
print("Model loaded.")

app = FastAPI(title="ReSiReg-Mini Pi 5 Server", version="0.1.0")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_image(source: str | bytes) -> Image.Image:
    """Load an image from a URL, a base64 data URL, or raw bytes."""
    if isinstance(source, str):
        if source.startswith("http://") or source.startswith("https://"):
            img = Image.open(requests.get(source, stream=True, timeout=30).raw)
        elif source.startswith("data:image"):
            # data:image/png;base64,....
            header, b64 = source.split(",", 1)
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
        elif Path(source).exists():
            img = Image.open(source)
        else:
            raise ValueError("Image source is not a URL, data URL, or existing file path.")
    else:
        img = Image.open(io.BytesIO(source))
    return img.convert("RGB")


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_mode(text: str) -> tuple[str, str, list[str], list[str], str, int]:
    """
    Parse a natural-language request into (mode, prompt, negatives, classes, seg_negative, n_clusters).
    Simple conventions:
      - /segment: class1, class2, ...
      - /traverse: prompt vs neg1, neg2
      - /similarity prompt - neg1, neg2
      - /pca: n_clusters
      - default: language_similarity
    """
    text = text.strip()
    if text.startswith("/segment"):
        rest = text[len("/segment"):].strip(": ")
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        return "zero_shot_segmentation", "", [], parts, "", 0
    if text.startswith("/traverse"):
        rest = text[len("/traverse"):].strip(": ")
        if " vs " in rest:
            prompt, negs = rest.split(" vs ", 1)
            return "traversability", prompt.strip(), [n.strip() for n in negs.split(",") if n.strip()], [], "", 0
        return "traversability", rest, [], [], "", 0
    if text.startswith("/similarity"):
        rest = text[len("/similarity"):].strip(": ")
        if " - " in rest:
            prompt, negs = rest.split(" - ", 1)
            return "language_similarity", prompt.strip(), [n.strip() for n in negs.split(",") if n.strip()], [], "", 0
        return "language_similarity", rest, [], [], "", 0
    if text.startswith("/pca"):
        rest = text[len("/pca"):].strip(": ")
        try:
            n_clusters = int(rest)
        except ValueError:
            n_clusters = 6
        return "pca", "", [], [], "", n_clusters
    # Default language_similarity with optional " vs " negatives
    if " vs " in text:
        prompt, negs = text.split(" vs ", 1)
        return "language_similarity", prompt.strip(), [n.strip() for n in negs.split(",") if n.strip()], [], "", 0
    return "language_similarity", text, [], [], "", 0


# ------------------------------------------------------------------
# Request/response models
# ------------------------------------------------------------------
class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "resireg-pi5"


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    temperature: float | None = 0.0
    max_tokens: int | None = None
    stream: bool | None = False


class EmbeddingRequest(BaseModel):
    model: str = MODEL_ID
    input: str | list[str]
    encoding_format: str | None = "float"


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [ModelInfo(id=MODEL_ID).model_dump()],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages required")

    # Extract text and image from the last user message
    last_user = None
    for m in reversed(req.messages):
        if m.role == "user":
            last_user = m
            break
    if last_user is None:
        raise HTTPException(status_code=400, detail="No user message found.")

    text_prompt = ""
    image_source = None

    if isinstance(last_user.content, str):
        text_prompt = last_user.content
    elif isinstance(last_user.content, list):
        for part in last_user.content:
            if part.get("type") == "text":
                text_prompt = part.get("text", "")
            elif part.get("type") == "image_url":
                image_source = part["image_url"].get("url")

    if not image_source:
        raise HTTPException(status_code=400, detail="An image is required.")

    try:
        image = load_image(image_source)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load image: {e}")

    mode, prompt, negatives, classes, seg_negative, n_clusters = parse_mode(text_prompt)
    t0 = time.perf_counter()

    result_payload: dict = {"mode": mode}
    out_images: list[Image.Image] = []

    if mode == "language_similarity":
        neg_list = negatives
        all_prompts = [prompt] + neg_list
        with torch.no_grad():
            dense = model.encode_image(image, apply_resireg_lite=True).cpu()
            text = model.encode_text(all_prompts).cpu()
        sims = cosine_similarity_map(dense, text)
        pos_sim = sims[0]
        if neg_list:
            sim = pos_sim - sims[1:].mean(dim=0)
        else:
            sim = pos_sim
        sim = normalize_similarity(sim)
        out_images.append(render_heatmap_overlay(image, sim, title=f"'{prompt}'"))
        result_payload["prompt"] = prompt
        result_payload["negatives"] = neg_list

    elif mode == "traversability":
        all_prompts = [prompt] + negatives
        with torch.no_grad():
            dense = model.encode_image(image, apply_resireg_lite=True).cpu()
            text = model.encode_text(all_prompts).cpu()
        sims = cosine_similarity_map(dense, text)
        probs = F.softmax(sims / 0.2, dim=0)
        trav_map = normalize_similarity(probs[0])
        out_images.append(render_heatmap_overlay(image, trav_map,
                                            title=f"Traversability: '{prompt}'"))
        result_payload["traversable"] = prompt
        result_payload["background"] = negatives

    elif mode == "zero_shot_segmentation":
        class_list = classes if classes else [prompt]
        all_prompts = class_list + ([seg_negative] if seg_negative else [])
        with torch.no_grad():
            dense = model.encode_image(image, apply_resireg_lite=True).cpu()
            text_all = model.encode_text(all_prompts).cpu()
        text_neg = text_all[-1:] if seg_negative else None
        text = text_all[:-1] if seg_negative else text_all
        overlay, mask, detected, undetected = vis_semseg(
            image, dense, text, class_list, text_neg=text_neg, temperature=1.0
        )
        out_images.extend([overlay, mask])
        result_payload["classes"] = class_list
        result_payload["detected"] = detected
        result_payload["undetected"] = undetected

    elif mode == "pca":
        with torch.no_grad():
            dense, _, _ = resolve_embeddings(model, image, None, None, apply_resireg_lite=True)
        spatial = _spatial_from_dense(dense)
        w, h = image.width, image.height
        out_images.extend([
            vis_pca_rgb(spatial, w, h),
            vis_pca_component(spatial, w, h),
            vis_pca_kmeans(spatial, w, h, n_clusters=n_clusters or 6),
        ])
        result_payload["n_clusters"] = n_clusters or 6

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    latency_ms = (time.perf_counter() - t0) * 1000.0
    result_payload["latency_ms"] = round(latency_ms, 1)
    if out_images:
        if len(out_images) == 1:
            result_payload["result_image_base64"] = f"data:image/png;base64,{pil_to_base64(out_images[0])}"
        else:
            result_payload["result_images_base64"] = [
                f"data:image/png;base64,{pil_to_base64(img)}" for img in out_images
            ]

    content_json = json.dumps(result_payload, indent=2)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content_json,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest):
    """Return image or text embeddings. Input starting with http://, https://,
    data:image, or an existing file path is treated as an image; otherwise text."""
    inputs = req.input if isinstance(req.input, list) else [req.input]
    embeddings_list: list[list[float]] = []

    for src in inputs:
        try:
            img = load_image(src)
            with torch.no_grad():
                emb = model.encode_image(img, apply_resireg_lite=True)
            vec = emb[0].mean(dim=(1, 2)).tolist()  # global average pool
        except Exception:
            # treat as text
            with torch.no_grad():
                emb = model.encode_text([src])
            vec = emb[0].tolist()
        embeddings_list.append(vec)

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": vec,
                "index": i,
            }
            for i, vec in enumerate(embeddings_list)
        ],
        "model": req.model,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.post("/v1/images/file")
def process_image_file(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    mode: str = Form("language_similarity"),
):
    """Simple multipart endpoint: upload an image file + prompt."""
    data = file.file.read()
    try:
        image = load_image(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load image: {e}")

    t0 = time.perf_counter()
    if mode == "language_similarity":
        all_prompts = [prompt] if prompt else ["object"]
        with torch.no_grad():
            dense = model.encode_image(image, apply_resireg_lite=True).cpu()
            text = model.encode_text(all_prompts).cpu()
        sims = cosine_similarity_map(dense, text)
        sim = normalize_similarity(sims[0])
        out_image = render_heatmap_overlay(image, sim, title=f"'{prompt}'")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    latency_ms = (time.perf_counter() - t0) * 1000.0
    b64 = pil_to_base64(out_image)
    return JSONResponse({
        "mode": mode,
        "prompt": prompt,
        "latency_ms": round(latency_ms, 1),
        "result_image_base64": f"data:image/png;base64,{b64}",
    })


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main() -> None:
    host = os.environ.get("RESIREG_HOST", "0.0.0.0")
    port = int(os.environ.get("RESIREG_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
