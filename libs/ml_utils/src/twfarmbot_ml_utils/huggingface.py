"""Small adapter around the official Gradio client."""

from __future__ import annotations

from pathlib import Path

from gradio_client import Client, handle_file


class HuggingFaceImageProcessor:
    """Invoke an image-processing endpoint hosted in a Gradio Space."""

    def __init__(
        self,
        space_id: str,
        api_name: str = "/run_similarity",
    ) -> None:
        self._client = Client(space_id, verbose=False)
        self.api_name = api_name

    def process(self, image_url: str, prompt: str) -> Path:
        """Process a public image URL and return the downloaded result path."""
        result = self._client.predict(
            image=handle_file(image_url),
            prompt=prompt,
            api_name=self.api_name,
        )
        return Path(result)
