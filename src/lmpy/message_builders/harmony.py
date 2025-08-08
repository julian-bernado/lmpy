# src/lmpy/message_builders/harmony.py
"""
OpenAI Harmony format message builder.

This builder creates messages in the Harmony format required by gpt-oss
and other reasoning models, with proper system/developer role separation
and channel management.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union
import re
from datetime import datetime

from .base import MessageBuilder, BuilderConfig, Message, Messages


class HarmonyMessageBuilder(MessageBuilder):
    """
    Message builder for OpenAI Harmony format.
    
    Creates messages with proper Harmony structure:
    - System messages contain metadata, reasoning settings, channels
    - Developer messages contain instructions (traditional "system prompt")
    - Proper channel handling for multi-channel responses
    """
    
    def build_messages(
        self,
        prompt_or_messages: Union[str, Messages],
        system_prompt: Optional[str] = None,
        developer_instructions: Optional[str] = None,
    ) -> List[Message]:
        """
        Build messages in OpenAI Harmony format.
        
        For Harmony models:
        - system_prompt becomes developer instructions if developer_instructions not provided
        - System message is auto-generated with metadata
        - developer_instructions takes precedence over system_prompt
        """
        if isinstance(prompt_or_messages, str):
            # Build complete Harmony conversation
            msgs: List[Message] = []
            
            # 1. System message (metadata, not instructions)
            system_content = self._build_system_message()
            msgs.append({"role": "system", "content": system_content})
            
            # 2. Developer message (instructions)
            instructions = developer_instructions or system_prompt
            if instructions:
                developer_content = self._build_developer_message(instructions)
                msgs.append({"role": "developer", "content": developer_content})
            
            # 3. User message
            msgs.append({"role": "user", "content": prompt_or_messages})
            
            return msgs
        
        # Already a message sequence - validate and pass through
        msgs = list(prompt_or_messages)
        for m in msgs:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError("Each message must be a dict with 'role' and 'content' keys.")
        
        # For existing message sequences, ensure we have proper Harmony structure
        has_system = any(m.get("role") == "system" for m in msgs)
        has_developer = any(m.get("role") == "developer" for m in msgs)
        
        # If no system message, prepend one
        if not has_system:
            system_content = self._build_system_message()
            msgs.insert(0, {"role": "system", "content": system_content})
        
        # If no developer message but we have instructions, add one
        if not has_developer and (developer_instructions or system_prompt):
            instructions = developer_instructions or system_prompt
            developer_content = self._build_developer_message(instructions)
            # Insert after system message
            insert_idx = 1 if has_system else 0
            msgs.insert(insert_idx, {"role": "developer", "content": developer_content})
        
        return msgs
    
    def _build_system_message(self) -> str:
        """
        Build the system message with Harmony metadata.
        
        This contains model identity, reasoning settings, channels, and tools
        but NOT user instructions (those go in developer message).
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        parts = [
            self.config.model_identity,
            f"Knowledge cutoff: {self.config.knowledge_cutoff}",
            f"Current date: {current_date}",
            "",
            f"Reasoning: {self.config.reasoning_effort}",
            "",
            f"# Valid channels: {', '.join(self.config.required_channels)}. Channel must be included for every message.",
        ]
        
        # Add tool channel routing if we have function tools
        parts.append("Calls to these tools must go to the commentary channel: 'functions'.")
        
        # Add built-in tools if enabled
        if self.config.enable_builtin_python or self.config.enable_builtin_browser:
            parts.append("")
            parts.append("# Tools")
            parts.append("")
            
            if self.config.enable_builtin_python:
                parts.extend(self._get_python_tool_definition())
            
            if self.config.enable_builtin_browser:
                parts.extend(self._get_browser_tool_definition())
        
        return "\n".join(parts)
    
    def _build_developer_message(self, instructions: str) -> str:
        """
        Build the developer message with user instructions.
        
        This is what traditionally would be the "system prompt" - the
        actual instructions for how the model should behave.
        """
        return f"# Instructions\n\n{instructions}"
    
    def _get_python_tool_definition(self) -> List[str]:
        """Get the built-in Python tool definition for system message."""
        return [
            "## python",
            "",
            "Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).",
            "",
            "When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.",
            ""
        ]
    
    def _get_browser_tool_definition(self) -> List[str]:
        """Get the built-in browser tool definition for system message."""
        return [
            "## browser",
            "",
            "// Tool for browsing.",
            "// The `cursor` appears in brackets before each browsing display: `[{cursor}]`.",
            "// Cite information from the tool using the following format:",
            "// `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.",
            "// Do not quote more than 10 words directly from the tool output.",
            "// sources=web (default: web)",
            "namespace browser {",
            "",
            "// Searches for information related to `query` and displays `topn` results.",
            "type search = (_: {",
            "query: string,",
            "topn?: number, // default: 10",
            "source?: string,",
            "}) => any;",
            "",
            "// Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.",
            "// Valid link ids are displayed with the formatting: `【{id}†.*】`.",
            "// If `cursor` is not provided, the most recent page is implied.",
            "// If `id` is a string, it is treated as a fully qualified URL associated with `source`.",
            "// If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.",
            "// Use this function without `id` to scroll to a new location of an opened page.",
            "type open = (_: {",
            "id?: number | string, // default: -1",
            "cursor?: number, // default: -1",
            "loc?: number, // default: -1",
            "num_lines?: number, // default: -1",
            "view_source?: boolean, // default: false",
            "source?: string,",
            "}) => any;",
            "",
            "// Finds exact matches of `pattern` in the current page, or the page given by `cursor`.",
            "type find = (_: {",
            "pattern: string,",
            "cursor?: number, // default: -1",
            "}) => any;",
            "",
            "} // namespace browser",
            ""
        ]
    
    def extract_final_response(self, response_content: str) -> str:
        """
        Extract the final user-facing response from Harmony model output.
        
        Returns the full response including thinking content, rather than 
        just the final channel content. This allows users to see the 
        model's reasoning process.
        """
        # Return the full response content including thinking/reasoning
        # This preserves all thinking tokens and special markers for the user
        return response_content
