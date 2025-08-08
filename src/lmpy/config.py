# src/lmpy/config.py
"""
Environment variables:
  - LMPY_BASE_URL  (default: "http://127.0.0.1:8080")
  - LMPY_TIMEOUT   (default: "60" seconds; float accepted)
"""

from dataclasses import dataclass
import os


def _normalize_base_url(url: str) -> str:
    """
    Clean up the URL
    """
    url = url.strip().rstrip("/")
    if not url:
        raise ValueError("BASE_URL cannot be empty")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"BASE_URL must start with http:// or https:// (got: {url})")
    return url


def _parse_timeout(val: str | None, default: float) -> float:
    if val is None or val == "":
        return default
    try:
        t = float(val)
    except ValueError as e:
        raise ValueError(f"LMPY_TIMEOUT must be a number (got: {val!r})") from e
    if t <= 0:
        raise ValueError("LMPY_TIMEOUT must be > 0")
    return t


@dataclass(frozen=True)
class Config:
    base_url: str
    timeout: float  # seconds


def load(**overrides) -> Config:
    """
    Load config from env with tiny validation.
    Precedence: explicit overrides > env vars > defaults.
    """
    base_url = overrides.get(
        "base_url",
        os.getenv("LMPY_BASE_URL", "http://127.0.0.1:8080"),
    )
    timeout = overrides.get(
        "timeout",
        _parse_timeout(os.getenv("LMPY_TIMEOUT"), 60.0),
    )
    return Config(base_url=_normalize_base_url(str(base_url)), timeout=float(timeout))
