"""Cliente del modelo multimodal (endpoint vLLM compatible con OpenAI).

Incluye un MockClient para probar el pipeline sin endpoint real.
Configurable por variables de entorno:
  VLLM_BASE_URL  (def. http://localhost:8000/v1)
  VLLM_API_KEY   (def. EMPTY)
  VLLM_MODEL     (def. nvidia/Gemma-4-31B-IT-NVFP4)
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Protocol

DEFAULT_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "nvidia/Gemma-4-31B-IT-NVFP4")


def load_image_b64(path: str) -> tuple[str, str]:
    """Read an image file and return (base64, mime_type)."""
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return b64, mime


class Client(Protocol):
    def ask(self, prompt: str, image_b64: str, mime: str = "image/jpeg") -> str: ...


class VLLMClient:
    """Thin wrapper over the OpenAI SDK pointed at a local vLLM endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        *,
        force_json: bool = False,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI  # lazy import

        self.model = model
        self.force_json = force_json
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def ask(self, prompt: str, image_b64: str, mime: str = "image/jpeg") -> str:
        kwargs = {}
        if self.force_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                        },
                    ],
                }
            ],
            **kwargs,
        )
        return resp.choices[0].message.content or ""


class MockClient:
    """Returns canned JSON so the pipeline can be exercised offline."""

    def __init__(self, value: str = "valor simulado", reasoning: str = "razonamiento simulado") -> None:
        self.value = value
        self.reasoning = reasoning

    def ask(self, prompt: str, image_b64: str, mime: str = "image/jpeg") -> str:
        import json

        return json.dumps({"value": self.value, "reasoning": self.reasoning}, ensure_ascii=False)
