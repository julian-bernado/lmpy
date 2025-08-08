# src/lmpy/server/manager.py
"""
Tiny process manager for llama.cpp's `llama-server`.

Goals:
- Start/stop a single `llama-server` process for one model.
- Wait until the HTTP API is ready before returning.
- Be boring: no magic, just flags you already know.

Usage:
    from lmpy.server.manager import LlamaServer
    from lmpy.client import Client
    from lmpy.paths import find

    # Chat server example
    gguf = find("allenai/olmo-2-1124-13b-q4")
    with LlamaServer(model=gguf, port=8080, ctx_size=8192, batch=512, n_gpu_layers=-1, alias="olmo13-q4") as srv:
        llm = Client(base_url=srv.base_url)
        print(llm.answer("Say 'ok' and nothing else.", max_tokens=5, temperature=0.0))

    # Embedding server example (requires an embedding GGUF)
    emb = find("qwen/Qwen3-Embedding-0.6B-q8")
    with LlamaServer(model=emb, port=8081, embedding=True, pooling="last", alias="qwen-0.6b-embed") as emb_srv:
        emb_client = Client(base_url=emb_srv.base_url)
        print(len(emb_client.embed("hello world")))

Notes:
- One process = one loaded model. Run multiple instances on different ports for multiple models.
- ctx_size/batch/ngl are **startup-time** knobs in llama.cpp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union
import os
import platform
import subprocess
import time
import requests
import json

from lmpy.paths import find as find_model_path

Pathish = Union[str, Path]


@dataclass
class LlamaServer:
    model: Pathish
    llama_server_bin: str = "llama-server"
    host: str = "127.0.0.1"
    port: int = 8080
    ctx_size: Optional[int] = None
    batch: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    embedding: bool = False
    pooling: Optional[str] = None
    alias: Optional[str] = None
    api_key: Optional[str] = None
    extra_args: Optional[Sequence[str]] = None
    env: Optional[dict] = None
    attach: bool = True
    log_to: Optional[Pathish] = None
    verbose: bool = False  # New parameter to control output verbosity

    _proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    # ---------- lifecycle ----------

    def start(self, timeout: float = 60.0) -> None:
        # If something is already serving here and attach=True, just use it.
        if self.attach and self._is_ready_basic():
            self._proc = None
            # still do a readiness probe to avoid 503
            self._wait_until_ready(timeout=timeout)
            return

        model_path = self._resolve_model()
        args = [self._resolve_bin(), "-m", str(model_path), "--host", self.host, "--port", str(self.port)]

        if self.ctx_size is not None:
            args += ["-c", str(self.ctx_size)]
        if self.batch is not None:
            args += ["-b", str(self.batch)]
        if self.n_gpu_layers is not None:
            args += ["-ngl", str(self.n_gpu_layers)]
        if self.embedding:
            args += ["--embedding"]
            if self.pooling:
                args += ["--pooling", self.pooling]
        if self.alias:
            args += ["--alias", self.alias]
        if self.api_key:
            args += ["--api-key", self.api_key]
        if self.extra_args:
            args += list(self.extra_args)

        stdout = stderr = None
        if self.log_to:
            log_path = Path(self.log_to)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            f = open(log_path, "a", buffering=1)
            stdout = stderr = f
        elif not self.verbose:
            # If not verbose and no log file specified, suppress output
            stdout = stderr = subprocess.DEVNULL

        creationflags = 0
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        child_env = os.environ.copy()
        if self.env:
            child_env.update(self.env)

        self._proc = subprocess.Popen(
            args,
            stdout=stdout,
            stderr=stderr,
            env=child_env,
            text=True,
            creationflags=creationflags,
        )

        self._wait_until_ready(timeout=timeout)

    def stop(self, graceful_timeout: float = 5.0) -> None:
        if not self._proc:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=graceful_timeout)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ---------- internals ----------

    def _resolve_model(self) -> Path:
        p = Path(str(self.model))
        if p.exists():
            return p.resolve()
        return find_model_path(str(self.model)).resolve()

    def _resolve_bin(self) -> str:
        return os.getenv("LMPY_LLAMA_SERVER_BIN", self.llama_server_bin)

    def _is_ready_basic(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=0.75)
            return r.ok
        except Exception:
            return False

    def _get_model_id(self, timeout: float = 0.75) -> Optional[str]:
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=timeout)
            if not r.ok:
                return None
            data = r.json()
            models = data.get("data") or data.get("models") or []
            if not models:
                return None
            mid = models[0].get("id") or models[0].get("name")
            return str(mid) if mid else None
        except Exception:
            return None

    def _probe_chat(self, model_id: str, timeout: float = 2.0) -> bool:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        try:
            r = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=timeout)
            if r.status_code == 200:
                return True
            # 503 is the “still loading” signal; anything else we surface after retries
            return False
        except Exception:
            return False

    def _probe_embeddings(self, model_id: str, timeout: float = 2.0) -> bool:
        payload = {"model": model_id, "input": "ok"}
        try:
            r = requests.post(f"{self.base_url}/v1/embeddings", json=payload, timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def _wait_until_ready(self, timeout: float) -> None:
        """
        Wait until the server can actually serve requests.
        /v1/models may be 200 while the model is still loading, so we
        probe a real endpoint (chat or embeddings).
        """
        deadline = time.time() + timeout
        last_err: Optional[str] = None

        model_id: Optional[str] = None
        while time.time() < deadline:
            # Step 1: get a model id
            model_id = model_id or self._get_model_id()
            if not model_id:
                last_err = "no model id from /v1/models"
                time.sleep(0.2)
                continue

            # Step 2: probe the actual endpoint
            ok = (
                self._probe_embeddings(model_id)
                if self.embedding
                else self._probe_chat(model_id)
            )
            if ok:
                return

            last_err = "endpoint not ready (503 or error)"
            time.sleep(0.3)

        raise RuntimeError(f"llama-server did not become ready within {timeout:.1f}s: {last_err}")
