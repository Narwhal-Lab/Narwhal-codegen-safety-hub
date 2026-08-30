from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from qprobe.backends import GenerationRequest, GenerationResponse, TokenUsage
from qprobe.errors import ConfigurationError, ModelCallError


class OpenAIChatBackend:
    def __init__(self, *, model_id: str, api_key: str, base_url: str = "https://api.openai.com/v1", timeout_seconds: float = 120.0) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ConfigurationError("model_id must be non-empty")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigurationError("api_key must be non-empty")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ConfigurationError("base_url must be non-empty")
        if timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds must be positive")
        self._model_id = model_id
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout_seconds = timeout_seconds

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not isinstance(request, GenerationRequest):
            raise ConfigurationError("request must be a GenerationRequest")
        if request.model_id != self._model_id:
            raise ModelCallError("Generation request uses an unexpected model_id")
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_new_tokens,
            "seed": request.seed,
        }
        http_request = Request(self._url, data=json.dumps(payload).encode("utf-8"), headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}, method="POST")
        started = perf_counter()
        try:
            with urlopen(http_request, timeout=self._timeout_seconds) as response:
                raw = response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise ModelCallError("External generation request failed") from error
        latency_seconds = perf_counter() - started
        try:
            decoded = json.loads(raw.decode("utf-8"))
            choice = decoded["choices"][0]
            text = choice["message"]["content"]
            usage_data = decoded.get("usage", {})
            usage = TokenUsage(input_tokens=int(usage_data.get("prompt_tokens", 0)), output_tokens=int(usage_data.get("completion_tokens", 0)))
            finish_reason = str(choice.get("finish_reason") or "unknown")
            request_id = decoded.get("id")
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ModelCallError("External generation response was invalid") from error
        if not isinstance(text, str) or not text.strip():
            raise ModelCallError("External generation returned empty text")
        return GenerationResponse(text=text.strip(), model_id=self._model_id, finish_reason=finish_reason, usage=usage, latency_seconds=latency_seconds, provider_request_id=request_id if isinstance(request_id, str) else None)

    def generate_many(self, requests: Sequence[GenerationRequest]) -> tuple[GenerationResponse, ...]:
        return tuple(self.generate(request) for request in requests)
