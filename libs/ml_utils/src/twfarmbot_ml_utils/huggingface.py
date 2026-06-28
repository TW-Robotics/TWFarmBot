"""Local vision model client for the OpenAI-compatible resireg-server."""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import requests


def parse_segmentation_labels(labels: list[str]) -> dict[str, float]:
    """Parse labels like 'weeds (23.4%), soil (45.1%)' into a class mapping."""
    out: dict[str, float] = {}
    for label in labels:
        for part in label.split(","):
            match = re.search(r"(.+?)\s*\(\s*([0-9]*\.?[0-9]+)\s*%\s*\)", part)
            if match:
                name = match.group(1).strip()
                pct = float(match.group(2)) / 100.0
                out[name] = round(pct, 3)
    return out


def _save_b64(b64_data_url: str) -> Path:
    """Decode a base64 data URL and write it to a temp file; return the path."""
    _, b64 = b64_data_url.split(",", 1)
    img_bytes = base64.b64decode(b64)
    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        f.write(img_bytes)
    return Path(path)


class VisionProcessor:
    """Call the local OpenAI-compatible ReSiReg-Mini vision server."""

    def __init__(self, base_url: str, api_name: str = "/run_sim") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_name = api_name

    def _chat(self, image_url: str, text: str) -> dict[str, Any]:
        """POST to /v1/chat/completions and return the parsed content payload."""
        url = f"{self.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": "SimonSchwaiger/resireg_mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return json.loads(content)
        return content

    def process(self, image_url: str, prompt: str, **kwargs: Any) -> Path:
        """Run open-language similarity on an image URL and return the result path."""
        negatives = kwargs.get("negatives", "")
        text = f"{prompt} vs {negatives}" if negatives else prompt
        result = self._chat(image_url, text)
        return _save_b64(result["result_image_base64"])

    def predict(self, image_url: str, api_name: str, **kwargs: Any) -> Any:
        """Call a specific analysis endpoint by its legacy api_name."""
        if api_name == "/run_sim":
            return self.process(image_url, kwargs.get("prompt", ""), **kwargs)

        if api_name == "/run_seg":
            classes = kwargs.get("classes", "")
            text = f"/segment: {classes}"
            result = self._chat(image_url, text)
            images = result.get("result_images_base64", [])
            return (
                _save_b64(images[0]),
                _save_b64(images[1]),
                result.get("detected", ""),
                result.get("undetected", ""),
            )

        if api_name == "/run_trav":
            prompt = kwargs.get("prompt", "")
            negatives = kwargs.get("negatives", "")
            text = f"/traverse: {prompt}" + (f" vs {negatives}" if negatives else "")
            result = self._chat(image_url, text)
            return _save_b64(result["result_image_base64"])

        if api_name == "/run_pca":
            n_clusters = kwargs.get("n_clusters", 6)
            text = f"/pca: {n_clusters}"
            result = self._chat(image_url, text)
            images = result.get("result_images_base64", [])
            return tuple(_save_b64(img) for img in images[:3])

        raise ValueError(f"Unsupported api_name: {api_name}")
