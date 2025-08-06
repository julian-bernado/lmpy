# src/lmpy/paths.py
"""
Path helpers for local GGUF files.

- Default model directory is ./models at your repo root.
- You can override with the env var LMPY_MODEL_DIR or by passing model_dir=...
- Filename convention (recommended, not enforced):
    models/{provider}/{family}-{size}b-q{quant}.gguf
  e.g.:
    models/allenai/olmo-2-1124-13b-q4.gguf
    models/qwen/Qwen3-Embedding-0.6B-q8.gguf

Typical use:
    from lmpy.paths import path_for, list_gguf, find

    # Build a path from parts (doesn't check existence):
    p = path_for("allenai", "olmo-2-1124", 13, "4")

    # List all .gguf files under the models dir:
    files = list_gguf()

    # Find a model by name/alias/substr (case-insensitive):
    p = find("olmo-2-1124-13b-q4")

Notes:
- These helpers do not start llama.cpp or validate GGUF content.
- They strictly deal with files on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import os
import re

# ---------- Config ----------

def get_model_dir(model_dir: Optional[Path | str] = None) -> Path:
    """
    Resolve the base model directory.
    Precedence: explicit arg > $LMPY_MODEL_DIR > ./models
    """
    if model_dir is not None:
        base = Path(model_dir)
    else:
        base = Path(os.getenv("LMPY_MODEL_DIR", "./models"))
    return base.resolve()


# ---------- Filename building ----------

def path_for(
    provider: str,
    family: str,
    size_b: int | str,
    quant: str,
    *,
    model_dir: Optional[Path | str] = None,
) -> Path:
    """
    Construct a GGUF path from parts (does not check existence).

    models/{provider}/{family}-{size}b-q{quant}.gguf
    """
    size_str = str(size_b).lower().rstrip("b")
    fname = f"{family}-{size_str}b-q{quant}.gguf"
    return get_model_dir(model_dir) / provider / fname


# ---------- Listing ----------

def list_gguf(*, model_dir: Optional[Path | str] = None, recursive: bool = True) -> List[Path]:
    """
    Return all .gguf files under the model directory.
    """
    base = get_model_dir(model_dir)
    if not base.exists():
        return []
    if recursive:
        return sorted([p for p in base.rglob("*.gguf") if p.is_file()])
    return sorted([p for p in base.glob("*.gguf") if p.is_file()])


# ---------- Finding ----------

_STEM_RE = re.compile(
    r"""
    ^
    (?P<family>.+?)                # family can contain dashes
    -
    (?P<size>\d+)[bB]              # number of billions, like 7b / 32B
    -
    q(?P<quant>[\w\.\-]+)          # quant tag, e.g., 4, 4_K_M, q8_0, etc.
    $
    """,
    re.VERBOSE,
)

def parse_alias(stem: str) -> Tuple[Optional[str], str, str, str]:
    """
    Parse an alias or filename stem into (provider, family, size_b, quant).
    Accepts optional 'provider/' prefix. Provider returns None if not present.

    Examples of accepted stems:
      - "allenai/olmo-2-1124-13b-q4"
      - "olmo-2-1124-13b-q4"
      - "Qwen3-Embedding-0.6B-q8"
    """
    provider = None
    rest = stem
    if "/" in stem:
        provider, rest = stem.split("/", 1)
        provider = provider or None

    m = _STEM_RE.match(rest)
    if not m:
        raise ValueError(
            f"Could not parse alias/stem '{stem}'. "
            "Expected '<family>-<size>b-q<quant>' (optionally 'provider/' prefix)."
        )
    family = m.group("family")
    size_b = m.group("size")
    quant = m.group("quant")
    return provider, family, size_b, quant


def find(name_or_alias: str, *, model_dir: Optional[Path | str] = None) -> Path:
    """
    Resolve a model path from a friendly name/alias or an explicit path.

    Resolution order:
      1) If 'name_or_alias' is an absolute path -> return it (must exist).
      2) If it ends with .gguf -> interpret relative to model_dir, check existence.
      3) Try to parse as '<provider/>family-sizeb-qquant' and build the expected path.
      4) Fallback: fuzzy search all .gguf stems for case-insensitive match or substring.

    Raises FileNotFoundError if nothing matches.
    Raises ValueError if ambiguous fuzzy matches are found.
    """
    base = get_model_dir(model_dir)
    s = name_or_alias.strip()

    # 1) Absolute path
    p = Path(s)
    if p.is_absolute():
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"GGUF not found at absolute path: {p}")

    # 2) Relative explicit filename
    if s.lower().endswith(".gguf"):
        candidate = (base / s).resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"GGUF not found: {candidate}")

    # 3) Structured alias (provider optional)
    try:
        provider, family, size_b, quant = parse_alias(s)
        structured = path_for(provider or "", family, size_b, quant, model_dir=base)
        if structured.exists():
            return structured.resolve()
        # If provider omitted, try any provider folder with that filename
        if provider is None:
            candidates = [p for p in list_gguf(model_dir=base) if p.stem.lower() == f"{family}-{size_b}b-q{quant}".lower()]
            if len(candidates) == 1:
                return candidates[0].resolve()
            if len(candidates) > 1:
                raise ValueError(
                    "Ambiguous alias: found multiple providers for the same filename stem:\n"
                    + "\n".join(str(c) for c in candidates)
                )
        # fall through to fuzzy if not found
    except ValueError:
        pass  # not a structured alias; try fuzzy

    # 4) Fuzzy search (case-insensitive)
    wanted = s.lower().removesuffix(".gguf")
    all_files = list_gguf(model_dir=base)
    if not all_files:
        raise FileNotFoundError(f"No .gguf files found under: {base}")

    exact = [p for p in all_files if p.stem.lower() == wanted]
    if len(exact) == 1:
        return exact[0].resolve()
    if len(exact) > 1:
        raise ValueError(
            f"Ambiguous name '{s}': multiple exact stems match.\n" + "\n".join(str(p) for p in exact)
        )

    # substring match as a last resort
    partial = [p for p in all_files if wanted in p.stem.lower()]
    if len(partial) == 1:
        return partial[0].resolve()
    if len(partial) > 1:
        raise ValueError(
            f"Ambiguous name '{s}': multiple stems contain that substring.\n" + "\n".join(str(p) for p in partial)
        )

    raise FileNotFoundError(
        f"Could not resolve '{s}' to a GGUF. Looked under: {base}\n"
        "Tip: use a full alias like 'provider/family-7b-q4' or pass an absolute path."
    )
