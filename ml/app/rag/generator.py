from __future__ import annotations

from typing import Callable, Iterator, Optional, Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...
    def stream(self, prompt: str) -> Iterator[str]: ...


class OllamaClient:
    """Ollama HTTP client. Inject `post` in tests to avoid httpx/network."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        post: Optional[Callable[[str, dict], dict]] = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._post = post

    def _do_post(self, url: str, payload: dict) -> dict:
        if self._post is not None:
            return self._post(url, payload)
        import httpx  # lazy

        resp = httpx.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def generate(self, prompt: str) -> str:
        data = self._do_post(
            f"{self.base_url}/api/generate",
            {"model": self.model, "prompt": prompt, "stream": False},
        )
        return data["response"]

    def stream(self, prompt: str) -> Iterator[str]:
        # Phase 4 wires SSE end-to-end; a simple non-chunked fallback keeps the
        # Protocol satisfied and callers correct in the meantime.
        yield self.generate(prompt)
