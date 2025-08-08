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
from lmpy.message_builders import create_message_builder, BuilderConfig, MessageBuilder


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
        # New Harmony-specific options
        reasoning_effort: str = "medium",  # high/medium/low
        developer_instructions: Optional[str] = None,  # Override for Harmony models
        enable_builtin_python: bool = False,
        enable_builtin_browser: bool = False,
    ):
        """
        Args:
            base_url: Root URL of llama.cpp server (e.g., "http://127.0.0.1:8080").
            timeout:  Per-request timeout in seconds.
            system_prompt: Included as the first message for string prompts (standard models)
                          or used as developer instructions for Harmony models if developer_instructions not provided.
            reasoning_effort: Reasoning level for Harmony models (high/medium/low).
            developer_instructions: Explicit instructions for Harmony models (takes precedence over system_prompt).
            enable_builtin_python: Enable built-in Python tool for Harmony models.
            enable_builtin_browser: Enable built-in browser tool for Harmony models.
        """
        cfg = load_config(
            **({k: v for k, v in {"base_url": base_url, "timeout": timeout}.items() if v is not None})
        )
        self._http = LlamacppHTTP(cfg)
        self._system_prompt = system_prompt
        
        # Harmony configuration
        self._builder_config = BuilderConfig(
            reasoning_effort=reasoning_effort,
            enable_builtin_python=enable_builtin_python,
            enable_builtin_browser=enable_builtin_browser,
        )
        self._developer_instructions = developer_instructions
        
        # Message builder will be created lazily when we know the model type
        self._message_builder: Optional[MessageBuilder] = None

    # -------- configuration --------

    def set_system_prompt(self, text: str) -> None:
        self._system_prompt = text

    def _get_message_builder(self) -> MessageBuilder:
        """Get or create the appropriate message builder for this model."""
        if self._message_builder is None:
            model_id = self._http._get_model_id()
            self._message_builder = create_message_builder(model_id, self._builder_config)
        return self._message_builder

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
        
        For Harmony models, automatically extracts the 'final' channel content.
        """
        builder = self._get_message_builder()
        messages = builder.build_messages(
            prompt_or_messages, 
            self._system_prompt,
            self._developer_instructions
        )
        
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
            raw_content = resp["choices"][0]["message"]["content"]
            # Use builder to extract final response (handles Harmony channels)
            return builder.extract_final_response(raw_content)
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
        
        Note: For Harmony models, this yields raw tokens which may include
        channel markers and analysis content. Use answer() for clean final responses.
        """
        builder = self._get_message_builder()
        messages = builder.build_messages(
            prompt_or_messages,
            self._system_prompt,
            self._developer_instructions
        )
        
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


