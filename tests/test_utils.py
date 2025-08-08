#!/usr/bin/env python3
"""
Shared utilities for lmpy tests.
"""

from lmpy.paths import find
import sys


def get_available_models(prefer_harmony=False, verbose=True):
    """
    Find available models and return them in priority order.
    
    Args:
        prefer_harmony: If True, prioritize Harmony models for advanced features
        verbose: If True, print model discovery messages
    """
    if prefer_harmony:
        models_to_try = [
            # Try Harmony models first for advanced features
            ("openai/gpt-oss-20b-q4", "gpt-oss-20b", "harmony"),
            # Then try qwen3 models (standard models)
            ("qwen/qwen3-14b-q4", "qwen3-14b", "standard"),
            ("qwen/qwen3-8b-q4", "qwen3-8b", "standard"), 
            ("qwen/qwen3-32b-q4", "qwen3-32b", "standard"),
            # Fallback options
            ("google/gemma-3-12b-q4", "gemma-3-12b", "standard"),
            ("google/gemma-3-4b-q4", "gemma-3-4b", "standard"),
            ("allenai/olmo-2-1124-13b-q4", "olmo-13b", "standard"),
        ]
    else:
        models_to_try = [
            # Try qwen3 models first (standard models)
            ("qwen/qwen3-14b-q4", "qwen3-14b", "standard"),
            ("qwen/qwen3-8b-q4", "qwen3-8b", "standard"), 
            ("qwen/qwen3-32b-q4", "qwen3-32b", "standard"),
            # Then try gpt-oss (Harmony model)
            ("openai/gpt-oss-20b-q4", "gpt-oss-20b", "harmony"),
            # Fallback options
            ("google/gemma-3-12b-q4", "gemma-3-12b", "standard"),
            ("google/gemma-3-4b-q4", "gemma-3-4b", "standard"),
            ("allenai/olmo-2-1124-13b-q4", "olmo-13b", "standard"),
        ]
    
    available = []
    for model_path, model_name, model_type in models_to_try:
        try:
            path = find(model_path)
            available.append((path, model_name, model_type))
            if verbose:
                print(f"✓ Found {model_name} ({model_type}) at {path}")
        except FileNotFoundError:
            if verbose:
                print(f"✗ {model_name} not found")
    
    if not available:
        print("❌ No models found! Please download some models to the ./models directory.")
        sys.exit(1)
    
    return available


def get_primary_model(models, prefer_harmony=False):
    """Get the primary model to use for testing."""
    if prefer_harmony:
        harmony_model = next((m for m in models if m[2] == "harmony"), None)
        standard_model = next((m for m in models if m[2] == "standard"), None)
        return harmony_model or standard_model or models[0]
    else:
        return models[0]
