"""Small adapter around the official Gradio client."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file


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


class HuggingFaceImageProcessor:
    """Invoke an image-processing endpoint hosted in a Gradio Space."""

    def __init__(
        self,
        space_id: str,
        api_name: str = "/run_sim",
    ) -> None:
        self._client = Client(space_id, verbose=False)
        self.api_name = api_name

    def process(self, image_url: str, prompt: str, **kwargs: Any) -> Path:
        """Process a public image URL and return the downloaded result path.

        Defaults to the ``/run_sim`` (open-language similarity) endpoint.
        Extra keyword arguments are forwarded to the Gradio client, so
        callers can pass ``negatives`` or other endpoint-specific params.
        """
        result = self._client.predict(
            image=handle_file(image_url),
            prompt=prompt,
            api_name=self.api_name,
            **kwargs,
        )
        return Path(result)

    def predict(self, image_url: str, api_name: str, **kwargs: Any) -> Any:
        """Call any endpoint in the space with the given arguments.

        Returns the raw Gradio result (path, tuple, etc.). Use this when
        the endpoint has a shape that ``process`` does not cover.
        """
        return self._client.predict(
            image=handle_file(image_url),
            api_name=api_name,
            **kwargs,
        )
