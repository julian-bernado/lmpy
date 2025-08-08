# src/lmpy/message_builders/base.py
"""
Base interface for message builders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union
from dataclasses import dataclass


Message = Dict[str, str]
Messages = Sequence[Message]


@dataclass
class BuilderConfig:
    """Configuration for message builders."""
    reasoning_effort: str = "medium"  # high/medium/low
    knowledge_cutoff: str = "2024-06"
    enable_builtin_python: bool = False
    enable_builtin_browser: bool = False
    model_identity: str = "You are ChatGPT, a large language model trained by OpenAI."
    required_channels: List[str] = None
    
    def __post_init__(self):
        if self.required_channels is None:
            self.required_channels = ["analysis", "commentary", "final"]


class MessageBuilder(ABC):
    """
    Abstract base class for message builders.
    
    Message builders are responsible for converting user inputs into
    properly formatted message sequences for different model types.
    """
    
    def __init__(self, config: Optional[BuilderConfig] = None):
        self.config = config or BuilderConfig()
    
    @abstractmethod
    def build_messages(
        self,
        prompt_or_messages: Union[str, Messages],
        system_prompt: Optional[str] = None,
        developer_instructions: Optional[str] = None,
    ) -> List[Message]:
        """
        Build a properly formatted message sequence.
        
        Args:
            prompt_or_messages: Either a string prompt or existing messages
            system_prompt: System-level prompt (meaning varies by format)
            developer_instructions: Developer instructions (Harmony-specific)
            
        Returns:
            List of formatted messages ready for the model
        """
        pass
    
    @abstractmethod
    def extract_final_response(self, response_content: str) -> str:
        """
        Extract the final user-facing response from model output.
        
        Args:
            response_content: Raw response content from the model
            
        Returns:
            Cleaned final response for the user
        """
        pass
