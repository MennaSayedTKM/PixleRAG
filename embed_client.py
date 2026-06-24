"""
embed_client.py
HTTP client for the Colab embedding server.

Usage:
    from embed_client import EmbedClient
    client = EmbedClient("https://xxxx.ngrok-free.app")
    vec = client.embed_image(pil_image_or_bytes)
    vec = client.embed_text("my query")
"""

import base64
import io
import os
import time
from typing import List, Union

import requests
from PIL import Image

# Default: read from environment; overridden by the Streamlit GUI at runtime
DEFAULT_EMBED_API_URL = os.environ.get("EMBED_API_URL", "").rstrip("/")

_TIMEOUT = 30        # seconds per request
_MAX_RETRIES = 2
_RETRY_DELAY = 1.5   # seconds between retries


class EmbedServerError(RuntimeError):
    """Raised when the embedding server is unreachable or returns an error."""


class EmbedClient:
    def __init__(self, base_url: str = DEFAULT_EMBED_API_URL):
        if not base_url:
            raise EmbedServerError(
                "EMBED_API_URL is not set. "
                "Start the Colab notebook, copy the printed ngrok URL, "
                "and paste it into the app's settings."
            )
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """Return the server's /health response or raise EmbedServerError."""
        return self._get("/health")

    def embed_image(self, image: Union[Image.Image, bytes]) -> List[float]:
        """
        Embed an image.

        Args:
            image: PIL.Image.Image or raw bytes (PNG/JPEG).

        Returns:
            List[float] — unit-normalised embedding vector.
        """
        b64 = self._image_to_b64(image)
        result = self._post("/embed_image", {"image_b64": b64})
        return result["embedding"]

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a text query in the same vector space as images.

        Returns:
            List[float] — unit-normalised embedding vector.
        """
        if not text.strip():
            raise ValueError("text must not be empty")
        result = self._post("/embed_text", {"text": text})
        return result["embedding"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _image_to_b64(self, image: Union[Image.Image, bytes]) -> str:
        if isinstance(image, bytes):
            return base64.b64encode(image).decode()
        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        raise TypeError(f"Expected PIL.Image or bytes, got {type(image)}")

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = requests.get(url, timeout=_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.ConnectionError:
                self._maybe_retry(attempt, "Cannot reach the embedding server")
            except requests.exceptions.Timeout:
                self._maybe_retry(attempt, "Embedding server timed out")
            except requests.exceptions.HTTPError as e:
                raise EmbedServerError(
                    f"Embedding server returned error {resp.status_code}: {resp.text}"
                ) from e

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = requests.post(url, json=payload, timeout=_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.ConnectionError:
                self._maybe_retry(
                    attempt,
                    "Cannot reach the embedding server. "
                    "Check that your Colab notebook is running and the URL is current.",
                )
            except requests.exceptions.Timeout:
                self._maybe_retry(
                    attempt,
                    "Embedding server timed out. "
                    "The Colab GPU may be busy — try again in a moment.",
                )
            except requests.exceptions.HTTPError as e:
                raise EmbedServerError(
                    f"Embedding server returned error {resp.status_code}: {resp.text}"
                ) from e

    def _maybe_retry(self, attempt: int, message: str):
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY)
        else:
            raise EmbedServerError(message)
