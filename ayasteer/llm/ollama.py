"""Minimal Ollama chat client over HTTP, stdlib only.

Used as the translator (curate) and the judge backend (judge). Thinking is
disabled and temperature is 0 so runs are reproducible for a given model.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_MODEL = "qwen3.5-9b"
DEFAULT_URL = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_URL, timeout: float = 120,
                 num_ctx: int = 2048) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # A small context window keeps the whole model on an 8 GB GPU; the Modelfile default
        # (64k) allocates a KV cache that forces CPU offload and a 5x slowdown.
        self.num_ctx = num_ctx

    def chat(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 256) -> str:
        body: dict = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": self.num_ctx},
        }
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.URLError as exc:  # HTTPError is a subclass
            raise OllamaError(f"ollama request failed: {exc}") from exc
        try:
            data = json.loads(raw)
            return data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError(f"malformed ollama response: {raw[:200]!r}") from exc
