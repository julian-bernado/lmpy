# src/lmpy/response.py
"""
Enhanced response objects for Harmony models.

Provides access to different channels of model output while maintaining
safety warnings about chain-of-thought content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings


@dataclass
class HarmonyResponse:
    """
    Response object for Harmony models with multi-channel content.
    
    Provides access to:
    - final: User-facing response content
    - analysis: Chain-of-thought (with safety warnings)
    - commentary: Tool calls and preambles
    - raw: Complete raw response
    """
    final: str
    analysis: Optional[str] = None
    commentary: Optional[str] = None
    raw: Optional[str] = None
    
    def __post_init__(self):
        """Issue safety warning if analysis content is accessed."""
        if self.analysis:
            warnings.warn(
                "Analysis channel content (chain-of-thought) has not been trained to the same "
                "safety standards as final responses. This content should not be shown to end users "
                "as it may contain harmful content. See the OpenAI gpt-oss model card for details.",
                UserWarning,
                stacklevel=2
            )


def parse_harmony_response(response_content: str) -> HarmonyResponse:
    """
    Parse a Harmony model response into its constituent channels.
    
    Args:
        response_content: Raw response content from the model
        
    Returns:
        HarmonyResponse with parsed channel content
    """
    final_content = ""
    analysis_content = None
    commentary_content = None
    
    # Split response by message boundaries
    messages = response_content.split("<|start|>assistant")
    
    for message in messages:
        if not message.strip():
            continue
            
        # Look for channel markers
        if "<|channel|>final<|message|>" in message:
            # Extract final channel content
            parts = message.split("<|channel|>final<|message|>")
            if len(parts) > 1:
                content = parts[1]
                # Remove end markers
                for marker in ["<|end|>", "<|return|>"]:
                    if marker in content:
                        content = content.split(marker)[0]
                final_content = content.strip()
        
        elif "<|channel|>analysis<|message|>" in message:
            # Extract analysis channel content
            parts = message.split("<|channel|>analysis<|message|>")
            if len(parts) > 1:
                content = parts[1]
                # Remove end markers
                for marker in ["<|end|>", "<|return|>"]:
                    if marker in content:
                        content = content.split(marker)[0]
                analysis_content = content.strip()
        
        elif "<|channel|>commentary<|message|>" in message:
            # Extract commentary channel content
            parts = message.split("<|channel|>commentary<|message|>")
            if len(parts) > 1:
                content = parts[1]
                # Remove end markers
                for marker in ["<|end|>", "<|return|>", "<|call|>"]:
                    if marker in content:
                        content = content.split(marker)[0]
                if commentary_content:
                    commentary_content += "\n" + content.strip()
                else:
                    commentary_content = content.strip()
    
    # If no channel markers found, treat entire response as final
    if not final_content and not analysis_content and not commentary_content:
        final_content = response_content.strip()
    
    return HarmonyResponse(
        final=final_content,
        analysis=analysis_content,
        commentary=commentary_content,
        raw=response_content
    )
