# src/lmpy/__init__.py
"""
lmpy: Python client for llama.cpp HTTP server with OpenAI Harmony support.
"""

from .client import Client
from .model_detection import ModelType, detect_model_type, is_harmony_model

__version__ = "0.1.0"

__all__ = [
    "Client",
    "ModelType",
    "detect_model_type",
    "is_harmony_model",
]