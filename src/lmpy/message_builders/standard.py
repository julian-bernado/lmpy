# src/lmpy/message_builders/standard.py
"""
Standard OpenAI chat format message builder.

This builder maintains the traditional system/user/assistant message format
that works with most language models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

from .base import MessageBuilder, BuilderConfig, Message, Messages


class StandardMessageBuilder(MessageBuilder):
    """
    Message builder for standard OpenAI chat format.
    
    Creates messages in the traditional format:
    - System messages contain instructions
    - User messages contain queries
    - Assistant messages contain responses
    """
    
    def build_messages(
        self,
        prompt_or_messages: Union[str, Messages],
        system_prompt: Optional[str] = None,
        developer_instructions: Optional[str] = None,
    ) -> List[Message]:
        """
        Build messages in standard OpenAI format.
        
        For standard models:
        - system_prompt is used as the system message
        - developer_instructions is ignored (Harmony-specific)
        """
        if isinstance(prompt_or_messages, str):
            # Simple string prompt
            msgs: List[Message] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt_or_messages})
            return msgs
        
        # Already a message sequence
        msgs = list(prompt_or_messages)
        for m in msgs:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError("Each message must be a dict with 'role' and 'content' keys.")
        return msgs
    
    def extract_final_response(self, response_content: str) -> str:
        """
        For standard models, the response content is the final response.
        """
        return response_content
