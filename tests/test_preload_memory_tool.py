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

import json
import base64
import unittest
from unittest import mock
from datetime import datetime

from google.genai import types

from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.tool_context import ToolContext
from google.adk.models.llm_request import LlmRequest

class TestPreloadMemoryTool(unittest.TestCase):
  
  def setUp(self):
    self.tool = PreloadMemoryTool()
    self.mock_tool_context = mock.MagicMock(spec=ToolContext)
    self.mock_tool_context.state = {}
    
    # Set up mock user content
    self.mock_tool_context.user_content = types.Content(
        parts=[types.Part.from_text("test query")]
    )
    
    # Set up mock memory response
    mock_memory = mock.MagicMock()
    mock_memory.events = []
    
    # Add a text event
    text_event = mock.MagicMock()
    text_event.author = "user"
    text_event.timestamp = datetime.now().timestamp()
    text_event.content = types.Content(
        parts=[types.Part.from_text("This is a text message")]
    )
    mock_memory.events.append(text_event)
    
    # Add an event with multiple parts
    multi_part_event = mock.MagicMock()
    multi_part_event.author = "assistant"
    multi_part_event.timestamp = datetime.now().timestamp()
    multi_part_event.content = types.Content(
        parts=[
            types.Part.from_text("This is text with an image"),
            mock.MagicMock(
                inline_data=mock.MagicMock(mime_type="image/jpeg")
            )
        ]
    )
    mock_memory.events.append(multi_part_event)
    
    # Set up mock memory search response
    mock_response = mock.MagicMock()
    mock_response.memories = [mock_memory]
    self.mock_tool_context.search_memory.return_value = mock_response
    
    # Set up mock LLM request
    self.mock_llm_request = mock.MagicMock(spec=LlmRequest)
    self.mock_llm_request.append_instructions = mock.MagicMock()
    
  async def test_process_llm_request(self):
    """Test that the tool correctly processes memory for an LLM request."""
    await self.tool.process_llm_request(
        tool_context=self.mock_tool_context,
        llm_request=self.mock_llm_request
    )
    
    # Check that search_memory was called with the expected query
    self.mock_tool_context.search_memory.assert_called_once_with("test query")
    
    # Check that append_instructions was called
    self.mock_llm_request.append_instructions.assert_called_once()
    
    # Check that the instructions contain the memory content
    instructions = self.mock_llm_request.append_instructions.call_args[0][0]
    self.assertIsInstance(instructions, list)
    self.assertEqual(len(instructions), 1)
    self.assertIn("user: This is a text message", instructions[0])
    self.assertIn("assistant: This is text with an image [image/jpeg content]", instructions[0])
    
  async def test_preload_text(self):
    """Test direct invocation to preload text content."""
    args = {
        "content": "This is a test content",
        "content_type": "text",
        "key": "test_key"
    }
    
    result = await self.tool.run_async(args=args, tool_context=self.mock_tool_context)
    
    self.assertEqual(result["status"], "success")
    self.assertEqual(result["key"], "test_key")
    self.mock_tool_context.add_to_memory.assert_called_once()
    
  async def test_preload_json(self):
    """Test direct invocation to preload JSON content."""
    test_json = {"name": "Test", "value": 123}
    args = {
        "content": json.dumps(test_json),
        "content_type": "json"
    }
    
    result = await self.tool.run_async(args=args, tool_context=self.mock_tool_context)
    
    self.assertEqual(result["status"], "success")
    self.mock_tool_context.add_to_memory.assert_called_once()
    
  async def test_preload_image(self):
    """Test direct invocation to preload image content."""
    # Create a simple base64 "image"
    mock_image_data = base64.b64encode(b"test image data").decode('utf-8')
    args = {
        "content": mock_image_data,
        "content_type": "image"
    }
    
    result = await self.tool.run_async(args=args, tool_context=self.mock_tool_context)
    
    self.assertEqual(result["status"], "success")
    self.mock_tool_context.add_to_memory.assert_called_once()
    
  async def test_missing_content(self):
    """Test error handling when content is missing."""
    args = {
        "content_type": "text"
    }
    
    result = await self.tool.run_async(args=args, tool_context=self.mock_tool_context)
    
    self.assertEqual(result["status"], "error")
    self.assertIn("Content is required", result["message"])

if __name__ == "__main__":
  unittest.main()
