# src/lmpy/client.py
"""
High-level client API for lmpy (pure llama.cpp HTTP).

Design goals:
- Simple: pass a string prompt, get a string answer.
- Safe defaults: local server on 127.0.0.1:8080, 30s timeout.
- No model juggling: llama-server is one-model-per-process; we auto-detect it.
- Keep streaming easy and explicit.

Typical use:
    from lmpy.client import Client

    llm = Client(system_prompt="You are concise.")
    print(llm.answer("Say hi in five words."))

    for chunk in llm.answer_stream("One quick pun."):
        print(chunk, end="", flush=True)

    print(llm.num_tokens("hello there"))
    print(llm.embed(["hello", "world"]))  # run an embedding model on this port
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union
import pathlib

import tqdm

from lmpy.backends.llamacpp_http import LlamacppHTTP, LmpyHTTPError, LmpyTimeoutError
from lmpy.config import load as load_config


Message = Dict[str, str]
Messages = Sequence[Message]


class Client:
    """Ergonomic wrapper around the HTTP backend."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        system_prompt: str = "You are a concise, helpful assistant.",
    ):
        """
        Args:
            base_url: Root URL of llama.cpp server (e.g., "http://127.0.0.1:8080").
            timeout:  Per-request timeout in seconds.
            system_prompt: Included as the first message for string prompts.
        """
        cfg = load_config(
            **({k: v for k, v in {"base_url": base_url, "timeout": timeout}.items() if v is not None})
        )
        self._http = LlamacppHTTP(cfg)
        self._system_prompt = system_prompt

    # -------- configuration --------

    def set_system_prompt(self, text: str) -> None:
        self._system_prompt = text

    # -------- utilities --------

    def num_tokens(self, text: str, *, add_special: Optional[bool] = None) -> int:
        """Return token count using this server's tokenizer."""
        out = self._http.tokenize(text, add_special=add_special)
        return int(out.get("n_tokens", 0))

    def embed(self, input: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Compute embeddings via /v1/embeddings on THIS port.
        Note: run an embedding model (e.g., Qwen3-Embedding-0.6B) on this server.
        Returns the raw numeric vectors only.
        """
        resp = self._http.embeddings(input)
        data = resp.get("data", [])
        vectors = [row.get("embedding", []) for row in data]
        if isinstance(input, str):
            return vectors[0] if vectors else []
        return vectors

    # -------- chat completions --------

    def answer(
        self,
        prompt_or_messages: Union[str, Messages],
        *,
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.6,
        top_p: float = 0.95,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop: Optional[Iterable[str]] = None,
        grammar: Optional[Union[str, pathlib.Path]] = None,
        response_format: Optional[Dict] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        """
        One-shot completion that returns the final text (no streaming).
        Accepts either a plain string or a full OpenAI-style messages list.
        """
        messages = _coerce_messages(prompt_or_messages, self._system_prompt)
        resp = self._http.chat(
            messages,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stop=stop,
            grammar=grammar,
            response_format=response_format,
            extra=extra,
        )
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception as e:  # pragma: no cover
            raise LmpyHTTPError(500, f"Unexpected response shape: {resp}") from e

    def answer_stream(
        self,
        prompt_or_messages: Union[str, Messages],
        *,
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.6,
        top_p: float = 0.95,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop: Optional[Iterable[str]] = None,
        grammar: Optional[Union[str, pathlib.Path]] = None,
        response_format: Optional[Dict] = None,
        extra: Optional[Dict] = None,
    ) -> Iterator[str]:
        """
        Streaming completion. Yields token deltas (strings).
        Accepts either a plain string or a messages list.
        """
        messages = _coerce_messages(prompt_or_messages, self._system_prompt)
        yield from self._http.chat_stream(
            messages,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stop=stop,
            grammar=grammar,
            response_format=response_format,
            extra=extra,
        )

    def multi_answer(
        self,
        prompts: List[str],
        *,
        progress: bool = False,
        show: bool = False,
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.6,
        top_p: float = 0.95,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        stop: Optional[Iterable[str]] = None,
        grammar: Optional[Union[str, pathlib.Path]] = None,
        response_format: Optional[Dict] = None,
        extra: Optional[Dict] = None,
    ) -> List[str]:
        """
        Answer a list of prompts sequentially.
        Set progress=True to show a tqdm progress bar.
        """
        iterator = prompts
        if progress:
            iterator = tqdm.tqdm(prompts, desc="lmpy.multi_answer", unit="prompt")

        answers: List[str] = []
        for p in iterator:
            ans = self.answer(
                p,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop=stop,
                grammar=grammar,
                response_format=response_format,
                extra=extra,
            )
            if show:
                print("\nPrompt:\n", p)
                print("Answer:\n", ans)
            answers.append(ans)
        return answers


# ---------- helpers ----------

def _coerce_messages(prompt_or_messages: Union[str, Messages], system_prompt: Optional[str]) -> List[Message]:
    """
    Normalize input into an OpenAI-style messages list.
    - If a string, prepend system prompt (if provided) and wrap as a single user message.
    - If already messages, return as list(messages).
    """
    if isinstance(prompt_or_messages, str):
        msgs: List[Message] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt_or_messages})
        return msgs

    msgs = list(prompt_or_messages)
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError("Each message must be a dict with 'role' and 'content' keys.")
    return msgs
