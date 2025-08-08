# src/lmpy/model_detection.py
"""
Model type detection for selecting appropriate prompt formats.

Harmony format is required for gpt-oss models and other models that support
the OpenAI Harmony response format with channels and special tokens.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
import os
import re


class ModelType(Enum):
    """Model types that require different prompt formatting."""
    STANDARD_CHAT = "standard"
    OPENAI_HARMONY = "harmony"


def detect_model_type(model_id: str) -> ModelType:
    """
    Detect if model requires Harmony format based on model ID/name.
    
    Args:
        model_id: The model identifier from the llama.cpp server
        
    Returns:
        ModelType indicating which prompt format to use
    """
    # Environment variable override for manual format selection
    override = os.getenv("LMPY_FORCE_FORMAT")
    if override:
        override_lower = override.lower()
        if override_lower in ("harmony", "openai_harmony"):
            return ModelType.OPENAI_HARMONY
        elif override_lower in ("standard", "chat"):
            return ModelType.STANDARD_CHAT
    
    # Normalize model ID for pattern matching
    model_lower = model_id.lower()
    
    # Patterns that indicate Harmony format models
    harmony_patterns = [
        r"gpt-oss",           # OpenAI gpt-oss models
        r"gpt.*oss",          # Variations like gpt-4-oss
        r"harmony",           # Any model with "harmony" in name
        r"reasoning",         # Reasoning models
    ]
    
    for pattern in harmony_patterns:
        if re.search(pattern, model_lower):
            return ModelType.OPENAI_HARMONY
    
    # Default to standard chat format
    return ModelType.STANDARD_CHAT


def is_harmony_model(model_id: str) -> bool:
    """Convenience function to check if model uses Harmony format."""
    return detect_model_type(model_id) == ModelType.OPENAI_HARMONY


def get_model_format_hint(model_id: str) -> str:
    """
    Get a human-readable hint about the model's format requirements.
    Useful for debugging and logging.
    """
    model_type = detect_model_type(model_id)
    if model_type == ModelType.OPENAI_HARMONY:
        return f"Model '{model_id}' uses OpenAI Harmony format (system/developer/channels)"
    else:
        return f"Model '{model_id}' uses standard chat format (system/user/assistant)"
