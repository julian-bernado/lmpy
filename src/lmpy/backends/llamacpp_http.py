# src/lmpy/backends/llamacpp_http.py
"""
Thin HTTP client for llama.cpp's built-in server.

Endpoints used:
  - GET  {BASE}/v1/models
  - POST {BASE}/v1/chat/completions
  - POST {BASE}/v1/embeddings
  - POST {BASE}/tokenize              (root utility; not under /v1)

Notes:
  - No Authorization header by default.
  - One-model-per-process: we auto-discover the model id from /v1/models, cache it,
    and include it in requests that require a model field.
  - Streaming yields ONLY text deltas (str). Caller can join them if desired.
  - tokenize() uses the tokenizer for whatever model this server loaded on this port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, Optional, Sequence, Union
import json
import pathlib

import requests

from lmpy.config import Config, load as load_config


# ---------- Exceptions ----------

class LmpyHTTPError(RuntimeError):
    """Raised for non-2xx HTTP responses from the llama.cpp server."""
    def __init__(self, status_code: int, message: str, payload: Optional[dict] = None):
        super().__init__(f"[{status_code}] {message}")
        self.status_code = status_code
        self.payload = payload or {}


class LmpyTimeoutError(TimeoutError):
    """Raised when an HTTP request times out."""


# ---------- Helpers ----------

def _maybe_read_grammar(grammar: Optional[Union[str, pathlib.Path]]) -> Optional[str]:
    """If `grammar` looks like a path that exists, read file contents; else pass the string through."""
    if grammar is None:
        return None
    if isinstance(grammar, pathlib.Path):
        p = grammar
    else:
        try:
            p = pathlib.Path(grammar)
        except Exception:
            return str(grammar)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return str(grammar)


def _raise_for_response(r: requests.Response) -> None:
    if 200 <= r.status_code < 300:
        return
    try:
        payload = r.json()
        message = (
            payload.get("error", {}).get("message")
            or payload.get("message")
            or r.text
        )
    except Exception:
        payload, message = None, r.text
    raise LmpyHTTPError(r.status_code, message.strip(), payload=payload)


# ---------- Client ----------

@dataclass
class LlamacppHTTP:
    """
    Minimal HTTP client for llama.cpp.

    Typical usage:
        from lmpy.backends.llamacpp_http import LlamacppHTTP

        chat = LlamacppHTTP()  # defaults to http://127.0.0.1:8080
        chat.models()          # sanity check

        # non-stream
        resp = chat.chat(
            [{"role": "user", "content": "Hello!"}],
            temperature=0.6,
            max_tokens=128,
        )
        text = resp["choices"][0]["message"]["content"]

        # stream
        for chunk in chat.chat_stream([{"role": "user", "content": "Hi"}], max_tokens=64):
            print(chunk, end="", flush=True)

        # embeddings (run a dedicated embedding model on this port)
        e = chat.embeddings(["hello", "world"])

        # tokenize (uses this server's model tokenizer)
        t = chat.tokenize("some text")
    """
    config: Config

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        if config is None:
            overrides = {}
            if base_url is not None:
                overrides["base_url"] = base_url
            if timeout is not None:
                overrides["timeout"] = timeout
            self.config = load_config(**overrides)
        else:
            self.config = config

        self._session = requests.Session()
        self._model_id: Optional[str] = None  # lazy-cached from /v1/models

    # ----- Internal: discover the single model id on this server -----

    def _get_model_id(self) -> str:
        if self._model_id:
            return self._model_id
        url = f"{self.config.base_url}/v1/models"
        try:
            r = self._session.get(url, timeout=self.config.timeout)
        except requests.exceptions.Timeout as e:
            raise LmpyTimeoutError("GET /v1/models timed out") from e
        _raise_for_response(r)
        data = r.json()
        models = data.get("data") or data.get("models") or []
        if not models:
            raise LmpyHTTPError(500, "No models reported by server; start llama-server with a model.")
        mid = models[0].get("id") or models[0].get("name")
        if not mid:
            raise LmpyHTTPError(500, "Server /v1/models returned an entry without an id/name.")
        self._model_id = str(mid)
        return self._model_id

    # ----- Public API -----

    def models(self) -> Dict[str, Any]:
        """Return the raw /v1/models JSON."""
        url = f"{self.config.base_url}/v1/models"
        try:
            r = self._session.get(url, timeout=self.config.timeout)
        except requests.exceptions.Timeout as e:
            raise LmpyTimeoutError("GET /v1/models timed out") from e
        _raise_for_response(r)
        return r.json()

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Iterable[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        grammar: Optional[Union[str, pathlib.Path]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Non-streaming chat completion. Returns the full OpenAI-style JSON.
        """
        payload: Dict[str, Any] = {
            "model": self._get_model_id(),
            "messages": list(messages),
        }
        if temperature is not None:        payload["temperature"] = temperature
        if top_p is not None:              payload["top_p"] = top_p
        if presence_penalty is not None:   payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:  payload["frequency_penalty"] = frequency_penalty
        if max_tokens is not None:         payload["max_tokens"] = max_tokens
        if stop is not None:               payload["stop"] = list(stop)
        if response_format is not None:    payload["response_format"] = response_format
        g = _maybe_read_grammar(grammar)
        if g:                              payload["grammar"] = g
        if extra:                          payload.update(extra)

        url = f"{self.config.base_url}/v1/chat/completions"
        try:
            r = self._session.post(url, json=payload, timeout=self.config.timeout)
        except requests.exceptions.Timeout as e:
            raise LmpyTimeoutError("POST /v1/chat/completions timed out") from e
        _raise_for_response(r)
        return r.json()

    def chat_stream(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Iterable[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        grammar: Optional[Union[str, pathlib.Path]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming chat completion. Yields text deltas (str).
        Caller should join them if they want the final string.
        """
        payload: Dict[str, Any] = {
            "model": self._get_model_id(),
            "messages": list(messages),
            "stream": True,
        }
        if temperature is not None:        payload["temperature"] = temperature
        if top_p is not None:              payload["top_p"] = top_p
        if presence_penalty is not None:   payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:  payload["frequency_penalty"] = frequency_penalty
        if max_tokens is not None:         payload["max_tokens"] = max_tokens
        if stop is not None:               payload["stop"] = list(stop)
        if response_format is not None:    payload["response_format"] = response_format
        g = _maybe_read_grammar(grammar)
        if g:                              payload["grammar"] = g
        if extra:                          payload.update(extra)

        url = f"{self.config.base_url}/v1/chat/completions"
        try:
            r = self._session.post(url, json=payload, timeout=self.config.timeout, stream=True)
        except requests.exceptions.Timeout as e:
            raise LmpyTimeoutError("POST /v1/chat/completions (stream) timed out") from e

        _raise_for_response(r)

        try:
            for raw in r.iter_lines(decode_unicode=False):
                if not raw:
                    continue
                # Expect Server-Sent Events lines like: b"data: {...}"
                if not raw.startswith(b"data:"):
                    continue
                data = raw.split(b"data:", 1)[1].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    # piece is bytes? It shouldn't be, but just in case.
                    yield piece if isinstance(piece, str) else piece.decode("utf-8", "ignore")
        finally:
            r.close()

    def embeddings(
        self,
        input: Union[str, Sequence[str]],
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute embeddings via /v1/embeddings. This assumes you've launched this
        server with a dedicated embedding GGUF (e.g., Qwen3-Embedding-0.6B) and
        appropriate flags (e.g., --embedding --pooling last/mean/cls).
        Returns the full JSON.
        """
        payload: Dict[str, Any] = {"model": self._get_model_id(), "input": input}
        if extra:
            payload.update(extra)

        url = f"{self.config.base_url}/v1/embeddings"
        try:
            r = self._session.post(url, json=payload, timeout=self.config.timeout)
        except requests.exceptions.Timeout as e:
            raise LmpyTimeoutError("POST /v1/embeddings timed out") from e
        _raise_for_response(r)
        return r.json()

    def tokenize(
        self,
        text: str,
        *,
        add_special: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Tokenize text using the server's root /tokenize endpoint for the model
        loaded in THIS server process/port.

        Returns:
          {
            "tokens": [int, ...],
            "n_tokens": int
          }
        """
        payload: Dict[str, Any] = {"content": text}
        if add_special is not None:
            payload["add_special"] = bool(add_special)

        url = f"{self.config.base_url}/tokenize"
        try:
            r = self._session.post(url, json=payload, timeout=self.config.timeout)
        except requests.exceptions.Timeout as e:
            raise LmpyTimeoutError("POST /tokenize timed out") from e
        _raise_for_response(r)
        data = r.json()

        # Normalize variations just in case
        tokens = data.get("tokens") or data.get("token_ids") or []
        if not isinstance(tokens, list):
            tokens = []
        n_tokens = data.get("n_tokens")
        if n_tokens is None:
            n_tokens = len(tokens)

        return {"tokens": tokens, "n_tokens": n_tokens}


if __name__ == "__main__":
    http = LlamacppHTTP()
    for chunk in http.chat_stream([
        {"role":"user", "content":"tell me a joke in 1 sentence"}
    ], max_tokens=1024):
        print(chunk, flush=True)