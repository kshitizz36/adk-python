# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from datetime import datetime
import base64
import json
from typing import Any, Dict, Optional, TYPE_CHECKING

from typing_extensions import override
from google.genai import types

from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class PreloadMemoryTool(BaseTool):
  """A tool that preloads the memory for the current user.
  
  Supports various content types including:
  - Text (default)
  - Images
  - JSON/structured data
  - Documents
  """

  def __init__(
      self,
      *,
      name: str = 'preload_memory',
      description: str = 'Preloads content into agent memory',
  ):
    # Name and description are provided for tool declaration
    super().__init__(name=name, description=description)

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    parts = tool_context.user_content.parts
    if not parts:
      return
      
    # Extract query from user content
    query = ""
    for part in parts:
      if part.text:
        query += part.text + " "
      elif hasattr(part, 'inline_data') and part.inline_data:
        # For images or other data, add a placeholder in the query
        mime_type = getattr(part.inline_data, 'mime_type', 'unknown')
        query += f"[User shared {mime_type} content] "
    
    query = query.strip()
    if not query:
      return
      
    response = tool_context.search_memory(query)
    if not response.memories:
      return
      
    memory_text = ''
    for memory in response.memories:
      time_str = datetime.fromtimestamp(memory.events[0].timestamp).isoformat()
      memory_text += f'Time: {time_str}\n'
      
      for event in memory.events:
        if not event.content or not event.content.parts:
          continue
          
        # Support multi-part content
        event_content = []
        for part in event.content.parts:
          if part.text:
            event_content.append(part.text)
          elif hasattr(part, 'inline_data') and part.inline_data:
            mime_type = getattr(part.inline_data, 'mime_type', 'unknown')
            event_content.append(f"[{mime_type} content]")
          elif hasattr(part, 'function_call') and part.function_call:
            func_name = getattr(part.function_call, 'name', 'unknown_function')
            event_content.append(f"[function call: {func_name}]")
            
        formatted_content = " ".join(event_content)
        if formatted_content:
          memory_text += f'{event.author}: {formatted_content}\n'
          
    si = f"""The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
{memory_text}
</PAST_CONVERSATIONS>
"""
    llm_request.append_instructions([si])

  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    """Gets the FunctionDeclaration for this tool when used directly."""
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to preload into memory. Can be text, JSON, or base64 encoded data.",
                },
                "content_type": {
                    "type": "string",
                    "description": "The type of content being preloaded.",
                    "enum": ["text", "json", "image", "document"],
                    "default": "text",
                },
                "key": {
                    "type": "string",
                    "description": "An optional key to associate with the content in memory.",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata to associate with the content.",
                },
            },
            "required": ["content"],
        },
    )

  async def run_async(
      self, *, args: Dict[str, Any], tool_context: ToolContext
  ) -> Dict[str, Any]:
    """Runs the tool with the given arguments and context.
    
    Args:
      args: The LLM-filled arguments.
      tool_context: The context of the tool.
      
    Returns:
      A dictionary containing the result of the operation.
    """
    content = args.get("content")
    content_type = args.get("content_type", "text")
    key = args.get("key")
    metadata = args.get("metadata", {})
    
    if not content:
      return {"status": "error", "message": "Content is required."}
    
    # Create a memory entry with the processed content
    memory_key = key or f"preloaded_{content_type}_{hash(content) % 10000}"
    
    # Create appropriate content part based on type
    try:
      memory_part = self._create_content_part(content, content_type)
      memory_content = types.Content(parts=[memory_part])
      
      # Add content to memory
      tool_context.add_to_memory(memory_key, memory_content, metadata)
      
      return {
          "status": "success",
          "message": f"Content successfully preloaded into memory with key '{memory_key}'.",
          "key": memory_key,
      }
    except Exception as e:
      return {"status": "error", "message": f"Failed to preload content: {str(e)}"}
  
  def _create_content_part(self, content: str, content_type: str) -> types.Part:
    """Creates the appropriate Part based on content type.
    
    Args:
      content: The raw content string.
      content_type: The type of the content.
      
    Returns:
      A Part object suitable for the content type.
    """
    if content_type == "text":
      return types.Part.from_text(content)
    
    elif content_type == "json":
      try:
        # Try to parse as JSON, but fall back to text
        json_content = json.loads(content)
        return types.Part.from_text(json.dumps(json_content, indent=2))
      except json.JSONDecodeError:
        return types.Part.from_text(content)
    
    elif content_type == "image":
      try:
        # Assume base64 encoded image
        image_data = base64.b64decode(content)
        return types.Part.from_data(
            data=image_data,
            mime_type="image/jpeg",  # Adjust based on actual image type if known
        )
      except Exception:
        # Fall back to treating as URL or other reference
        return types.Part.from_text(content)
    
    elif content_type == "document":
      try:
        # Try to decode as base64 for binary documents
        doc_data = base64.b64decode(content)
        return types.Part.from_data(
            data=doc_data,
            mime_type="application/octet-stream",  # Generic binary type
        )
      except Exception:
        # Fall back to treating as text
        return types.Part.from_text(content)
    
    # Default fallback
    return types.Part.from_text(content)


preload_memory_tool = PreloadMemoryTool()
