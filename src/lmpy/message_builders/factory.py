# src/lmpy/message_builders/factory.py
"""
Factory for creating appropriate message builders based on model type.
"""

from __future__ import annotations

from typing import Optional

from ..model_detection import ModelType, detect_model_type
from .base import MessageBuilder, BuilderConfig
from .standard import StandardMessageBuilder
from .harmony import HarmonyMessageBuilder


def create_message_builder(
    model_id: str,
    config: Optional[BuilderConfig] = None
) -> MessageBuilder:
    """
    Create an appropriate message builder for the given model.
    
    Args:
        model_id: The model identifier to detect format for
        config: Configuration for the message builder
        
    Returns:
        MessageBuilder instance appropriate for the model type
    """
    model_type = detect_model_type(model_id)
    builder_config = config or BuilderConfig()
    
    if model_type == ModelType.OPENAI_HARMONY:
        return HarmonyMessageBuilder(builder_config)
    else:
        return StandardMessageBuilder(builder_config)
