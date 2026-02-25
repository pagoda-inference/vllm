# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda tool call JSON repair postprocessor."""

import json
from unittest.mock import MagicMock, patch

import pytest

from vllm.pagoda.tool_call_postprocessor import (
    StreamingToolCallAccumulator,
    ToolCallPostProcessor,
    ToolCallResult,
)


class TestToolCallPostProcessor:
    """Tests for ToolCallPostProcessor."""

    def setup_method(self):
        """Create a fresh postprocessor for each test."""
        self.pp = ToolCallPostProcessor()
        self.model = "test-model"
        self.tenant = "test-tenant"

    @patch("vllm.pagoda.tool_call_postprocessor.pagoda_tool_call_repair_total")
    def test_valid_json_passthrough(self, mock_metric):
        """Valid JSON → success=True, repaired=False."""
        raw = json.dumps([{"name": "get_weather", "arguments": {"city": "NYC"}}])
        result = self.pp.fix(raw, self.model, self.tenant)

        assert result.success is True
        assert result.repaired is False
        assert result.tool_calls == [{"name": "get_weather", "arguments": {"city": "NYC"}}]

    @patch("vllm.pagoda.tool_call_postprocessor.pagoda_tool_call_repair_total")
    def test_valid_json_single_object(self, mock_metric):
        """Single JSON object (not array) is normalized to list."""
        raw = json.dumps({"name": "search", "arguments": {"q": "test"}})
        result = self.pp.fix(raw, self.model, self.tenant)

        assert result.success is True
        assert result.repaired is False
        assert result.tool_calls == [{"name": "search", "arguments": {"q": "test"}}]

    @patch("vllm.pagoda.tool_call_postprocessor.pagoda_tool_call_repair_total")
    @patch("vllm.pagoda.tool_call_postprocessor.json5")
    def test_json5_repair_single_quotes(self, mock_json5, mock_metric):
        """json5 repairs single quotes → repaired=True."""
        raw = "{'name': 'search', 'arguments': {'q': 'test'}}"
        mock_json5.loads.return_value = {"name": "search", "arguments": {"q": "test"}}
        result = self.pp.fix(raw, self.model, self.tenant)

        assert result.success is True
        assert result.repaired is True
        mock_json5.loads.assert_called_once_with(raw)

    @patch("vllm.pagoda.tool_call_postprocessor.pagoda_tool_call_repair_total")
    @patch("vllm.pagoda.tool_call_postprocessor.json5")
    def test_json5_repair_trailing_comma(self, mock_json5, mock_metric):
        """json5 repairs trailing commas → repaired=True."""
        raw = '{"name": "search", "arguments": {"q": "test",},}'
        mock_json5.loads.return_value = {"name": "search", "arguments": {"q": "test"}}
        result = self.pp.fix(raw, self.model, self.tenant)

        assert result.success is True
        assert result.repaired is True

    @patch("vllm.pagoda.tool_call_postprocessor.pagoda_tool_call_repair_total")
    def test_deepseek_tool_call_extraction(self, mock_metric):
        """DeepSeek <tool_call> tag extraction."""
        inner = json.dumps({"name": "calc", "arguments": {"x": 1}})
        raw = f"Some text <tool_call>{inner}</tool_call> more text"
        result = self.pp.fix(raw, "deepseek-v2", self.tenant)

        assert result.success is True
        assert result.repaired is True
        assert result.tool_calls[0]["name"] == "calc"

    @patch("vllm.pagoda.tool_call_postprocessor.pagoda_tool_call_repair_total")
    def test_qwen_function_extraction(self, mock_metric):
        """Qwen ✿FUNCTION✿ format extraction."""
        inner = json.dumps({"name": "lookup", "arguments": {"id": 42}})
        raw = f"✿FUNCTION✿{inner}✿RESULT✿"
        result = self.pp.fix(raw, "qwen-72b", self.tenant)

        assert result.success is True
        assert result.repaired is True
        assert result.tool_calls[0]["name"] == "lookup"

    @patch("vllm.pagoda.tool_call_postprocessor.pagoda_tool_call_repair_total")
    def test_extract_json_from_text(self, mock_metric):
        """Extract embedded JSON from surrounding text."""
        obj = {"name": "fn", "arguments": {}}
        raw = f'Here is the result: {json.dumps(obj)} end of output'
        # json.loads on full string fails, but regex extraction works
        # Need to ensure strategy 1 fails — the full string isn't valid JSON
        result = self.pp.fix(raw, self.model, self.tenant)

        assert result.success is True
        assert result.repaired is True
        assert result.tool_calls[0]["name"] == "fn"

    @patch("vllm.pagoda.tool_call_postprocessor.pagoda_tool_call_repair_total")
    def test_fix_missing_closing_brace(self, mock_metric):
        """Fix missing closing braces."""
        raw = '{"name": "fn", "arguments": {"x": 1}'
        result = self.pp.fix(raw, self.model, self.tenant)

        assert result.success is True
        assert result.repaired is True
        assert result.tool_calls[0]["name"] == "fn"

    @patch("vllm.pagoda.tool_call_postprocessor.pagoda_tool_call_repair_total")
    def test_empty_input(self, mock_metric):
        """Empty input → success=False."""
        result = self.pp.fix("", self.model, self.tenant)
        assert result.success is False
        assert result.repaired is False
        assert "Empty" in result.error_message

    @patch("vllm.pagoda.tool_call_postprocessor.pagoda_tool_call_repair_total")
    def test_whitespace_only_input(self, mock_metric):
        """Whitespace-only input → success=False."""
        result = self.pp.fix("   \n  ", self.model, self.tenant)
        assert result.success is False
        assert result.repaired is False

    @patch("vllm.pagoda.tool_call_postprocessor.pagoda_tool_call_repair_total")
    def test_all_strategies_fail(self, mock_metric):
        """Completely unparseable input → success=False with error_message."""
        raw = "this is not json at all and has no braces"
        result = self.pp.fix(raw, self.model, self.tenant)

        assert result.success is False
        assert result.repaired is False
        assert result.error_message is not None


class TestStreamingToolCallAccumulator:
    """Tests for StreamingToolCallAccumulator."""

    def setup_method(self):
        """Create a fresh accumulator for each test."""
        self.pp = ToolCallPostProcessor()
        self.acc = StreamingToolCallAccumulator(
            model_name="test-model",
            tenant_id="test-tenant",
            postprocessor=self.pp,
        )

    def _make_chunk(self, tool_calls=None, finish_reason=None):
        """Helper to create a mock streaming chunk."""
        chunk = MagicMock()
        choice = MagicMock()
        choice.finish_reason = finish_reason
        delta = MagicMock()
        delta.tool_calls = tool_calls
        choice.delta = delta
        chunk.choices = [choice]
        return chunk

    def _make_tool_call_delta(self, index=0, tc_id=None, name=None, arguments=None):
        """Helper to create a mock tool call delta."""
        delta = MagicMock()
        delta.index = index
        delta.id = tc_id
        delta.type = "function" if tc_id else None
        func = MagicMock()
        func.name = name
        func.arguments = arguments
        delta.function = func
        return delta

    def test_non_tool_call_chunk_passthrough(self):
        """Non-tool-call chunks are returned as-is."""
        chunk = self._make_chunk(tool_calls=None, finish_reason=None)
        result = self.acc.feed_chunk(chunk)
        assert result is chunk

    def test_no_choices_passthrough(self):
        """Chunk with no choices is returned as-is."""
        chunk = MagicMock()
        chunk.choices = []
        result = self.acc.feed_chunk(chunk)
        assert result is chunk

    def test_tool_call_delta_accumulated(self):
        """Tool call delta chunks are accumulated, returning None."""
        tc = self._make_tool_call_delta(
            index=0, tc_id="call_1", name="fn",
            arguments='{"x":'
        )
        chunk = self._make_chunk(tool_calls=[tc])
        result = self.acc.feed_chunk(chunk)
        assert result is None
        assert self.acc._has_pending is True

    @patch("vllm.pagoda.tool_call_postprocessor.pagoda_tool_call_repair_total")
    def test_finalize_if_pending(self, mock_metric):
        """finalize_if_pending() repairs accumulated tool calls."""
        tc = self._make_tool_call_delta(
            index=0, tc_id="call_1", name="fn",
            arguments='{"x": 1}'
        )
        chunk = self._make_chunk(tool_calls=[tc])
        self.acc.feed_chunk(chunk)

        result = self.acc.finalize_if_pending()
        assert result is not None
        assert result.success is True

    def test_finalize_no_pending(self):
        """finalize_if_pending() returns None when nothing accumulated."""
        result = self.acc.finalize_if_pending()
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
