# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool call JSON repair for non-streaming and streaming responses.

Repair strategies (in priority order):
1. json.loads() — already valid
2. json5.loads() — handles single quotes, trailing commas, comments
3. Extract JSON from surrounding text (regex)
4. Fix missing closing braces
5. Model-specific extraction (DeepSeek, Qwen, LLaMA)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.pagoda.metrics import pagoda_tool_call_repair_total

logger = init_logger(__name__)

# Try to import json5; fall back to None
try:
    import json5
except ImportError:
    json5 = None  # type: ignore[assignment]
    logger.warning(
        "json5 not installed — some tool call repairs may be less reliable. "
        "Install with: pip install json5"
    )

# Regex patterns for JSON extraction
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")

# Model-specific patterns
_DEEPSEEK_TOOL_CALL_RE = re.compile(
    r"<tool_call>([\s\S]*?)</tool_call>", re.DOTALL
)
_QWEN_FUNCTION_RE = re.compile(
    r"✿FUNCTION✿([\s\S]*?)✿RESULT✿", re.DOTALL
)


@dataclass
class ToolCallResult:
    """Result of a tool call repair attempt."""
    success: bool
    repaired: bool  # True if repair was needed (not just passthrough)
    tool_calls: list[dict] | None = None
    error_message: str | None = None


class ToolCallPostProcessor:
    """Stateless tool call JSON repair for non-streaming responses."""

    def fix(
        self,
        raw_output: str,
        model_name: str,
        tenant_id: str = "__default__",
    ) -> ToolCallResult:
        """Attempt to parse/repair tool call JSON.

        Args:
            raw_output: JSON string of tool_calls array
            model_name: Model name for model-specific strategies
            tenant_id: For metrics labeling
        """
        if not raw_output or not raw_output.strip():
            return ToolCallResult(
                success=False,
                repaired=False,
                error_message="Empty tool call output",
            )

        # Strategy 1: Direct JSON parse
        parsed = self._try_json_loads(raw_output)
        if parsed is not None:
            pagoda_tool_call_repair_total.labels(
                tenant_id=tenant_id, model=model_name, status="success"
            ).inc()
            return ToolCallResult(
                success=True, repaired=False, tool_calls=self._normalize(parsed)
            )

        # Strategy 2: json5 parse
        parsed = self._try_json5(raw_output)
        if parsed is not None:
            pagoda_tool_call_repair_total.labels(
                tenant_id=tenant_id, model=model_name, status="repaired"
            ).inc()
            return ToolCallResult(
                success=True, repaired=True, tool_calls=self._normalize(parsed)
            )

        # Strategy 3: Model-specific extraction
        extracted = self._try_model_specific(raw_output, model_name)
        if extracted is not None:
            pagoda_tool_call_repair_total.labels(
                tenant_id=tenant_id, model=model_name, status="repaired"
            ).inc()
            return ToolCallResult(
                success=True, repaired=True, tool_calls=self._normalize(extracted)
            )

        # Strategy 4: Extract JSON from surrounding text
        extracted = self._try_extract_json(raw_output)
        if extracted is not None:
            pagoda_tool_call_repair_total.labels(
                tenant_id=tenant_id, model=model_name, status="repaired"
            ).inc()
            return ToolCallResult(
                success=True, repaired=True, tool_calls=self._normalize(extracted)
            )

        # Strategy 5: Fix missing closing braces
        fixed = self._try_fix_braces(raw_output)
        if fixed is not None:
            pagoda_tool_call_repair_total.labels(
                tenant_id=tenant_id, model=model_name, status="repaired"
            ).inc()
            return ToolCallResult(
                success=True, repaired=True, tool_calls=self._normalize(fixed)
            )

        # All strategies failed
        pagoda_tool_call_repair_total.labels(
            tenant_id=tenant_id, model=model_name, status="failed"
        ).inc()
        logger.warning(
            "Failed to repair tool call JSON for tenant=%s model=%s: %s",
            tenant_id,
            model_name,
            raw_output[:200],
        )
        return ToolCallResult(
            success=False,
            repaired=False,
            error_message=f"Failed to parse tool call output: {raw_output[:200]}",
        )

    @staticmethod
    def _try_json_loads(text: str) -> list | dict | None:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _try_json5(text: str) -> list | dict | None:
        if json5 is None:
            return None
        try:
            return json5.loads(text)
        except Exception:
            return None

    @staticmethod
    def _try_model_specific(
        text: str, model_name: str
    ) -> list | dict | None:
        model_lower = model_name.lower()

        # DeepSeek: <tool_call>...</tool_call>
        if "deepseek" in model_lower:
            matches = _DEEPSEEK_TOOL_CALL_RE.findall(text)
            if matches:
                results = []
                for m in matches:
                    try:
                        results.append(json.loads(m.strip()))
                    except json.JSONDecodeError:
                        pass
                if results:
                    return results

        # Qwen: ✿FUNCTION✿...✿RESULT✿
        if "qwen" in model_lower:
            matches = _QWEN_FUNCTION_RE.findall(text)
            if matches:
                results = []
                for m in matches:
                    try:
                        results.append(json.loads(m.strip()))
                    except json.JSONDecodeError:
                        pass
                if results:
                    return results

        return None

    @staticmethod
    def _try_extract_json(text: str) -> list | dict | None:
        """Extract JSON array or object from surrounding text."""
        # Try array first
        match = _JSON_ARRAY_RE.search(text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try object
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _try_fix_braces(text: str) -> list | dict | None:
        """Fix missing closing braces/brackets."""
        stripped = text.strip()
        open_braces = stripped.count("{") - stripped.count("}")
        open_brackets = stripped.count("[") - stripped.count("]")

        if open_braces <= 0 and open_brackets <= 0:
            return None

        fixed = stripped + ("}" * open_braces) + ("]" * open_brackets)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize(parsed: list | dict) -> list[dict]:
        """Ensure result is always a list of dicts."""
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        return [parsed]


class StreamingToolCallAccumulator:
    """Stateful per-request accumulator for streaming tool call repair.

    Accumulates tool_call delta chunks, then repairs the complete JSON
    when finish_reason="tool_calls" is received.
    """

    def __init__(
        self,
        model_name: str,
        tenant_id: str,
        postprocessor: ToolCallPostProcessor,
    ) -> None:
        self._model_name = model_name
        self._tenant_id = tenant_id
        self._postprocessor = postprocessor
        # Accumulated arguments per tool call index
        self._buffers: dict[int, dict] = {}
        # Track tool call metadata (id, name) per index
        self._tool_meta: dict[int, dict] = {}
        self._has_pending = False

    def feed_chunk(self, chunk: object) -> object | None:
        """Process a streaming chunk.

        - Non-tool-call chunks: returned as-is (transparent passthrough)
        - Tool-call delta chunks: accumulated internally, returns None
        - finish_reason="tool_calls" chunk: triggers repair, returns
          modified chunk with complete tool_calls

        Args:
            chunk: A ChatCompletionStreamResponse object

        Returns:
            The chunk to yield to client, or None to suppress
        """
        choices = getattr(chunk, "choices", None)
        if not choices:
            return chunk

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            return chunk

        tool_calls = getattr(delta, "tool_calls", None)

        # No tool calls in this chunk — passthrough
        if not tool_calls:
            # Check if this is the final chunk with finish_reason
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "tool_calls" and self._has_pending:
                return self._build_final_chunk(chunk)
            return chunk

        # Accumulate tool call deltas
        for tc_delta in tool_calls:
            idx = getattr(tc_delta, "index", 0)
            func = getattr(tc_delta, "function", None)

            if idx not in self._buffers:
                self._buffers[idx] = {"arguments": ""}
                self._tool_meta[idx] = {}

            # Capture metadata (id, name) from first delta
            tc_id = getattr(tc_delta, "id", None)
            if tc_id:
                self._tool_meta[idx]["id"] = tc_id
            tc_type = getattr(tc_delta, "type", None)
            if tc_type:
                self._tool_meta[idx]["type"] = tc_type
            if func:
                name = getattr(func, "name", None)
                if name:
                    self._tool_meta[idx]["name"] = name
                args = getattr(func, "arguments", None)
                if args:
                    self._buffers[idx]["arguments"] += args

        self._has_pending = True

        # Check finish_reason on this chunk
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "tool_calls":
            return self._build_final_chunk(chunk)

        # Suppress this chunk (accumulated internally)
        return None

    def finalize_if_pending(self) -> ToolCallResult | None:
        """Finalize any pending accumulated tool calls.

        Called when the generator is exhausted without a
        finish_reason="tool_calls" chunk.
        """
        if not self._has_pending:
            return None

        return self._repair_accumulated()

    def _repair_accumulated(self) -> ToolCallResult:
        """Repair all accumulated tool call arguments."""
        tool_calls = []
        for idx in sorted(self._buffers.keys()):
            meta = self._tool_meta.get(idx, {})
            args_str = self._buffers[idx]["arguments"]

            # Try to repair the arguments JSON
            result = self._postprocessor.fix(
                args_str, self._model_name, self._tenant_id
            )

            if result.success and result.tool_calls:
                # The fix() returns parsed JSON — re-serialize for the
                # OpenAI format where arguments is a string
                repaired_args = json.dumps(
                    result.tool_calls[0]
                    if len(result.tool_calls) == 1
                    else result.tool_calls
                )
            elif result.success:
                repaired_args = args_str
            else:
                return ToolCallResult(
                    success=False,
                    repaired=False,
                    error_message=result.error_message,
                )

            tool_calls.append(
                {
                    "id": meta.get("id", f"call_{idx}"),
                    "type": meta.get("type", "function"),
                    "function": {
                        "name": meta.get("name", ""),
                        "arguments": repaired_args,
                    },
                }
            )

        self._has_pending = False
        return ToolCallResult(
            success=True,
            repaired=True,
            tool_calls=tool_calls,
        )

    def _build_final_chunk(self, original_chunk: object) -> object:
        """Build a final chunk with repaired tool calls."""
        repair_result = self._repair_accumulated()

        if not repair_result.success:
            logger.warning(
                "Streaming tool call repair failed: %s",
                repair_result.error_message,
            )
            # Return original chunk unmodified on failure
            return original_chunk

        # Modify the chunk to include complete tool calls
        # We replace the delta's tool_calls with the repaired version
        from vllm.entrypoints.openai.engine.protocol import (
            DeltaFunctionCall,
            DeltaToolCall,
        )

        choices = getattr(original_chunk, "choices", [])
        if choices:
            delta = choices[0].delta
            repaired_tool_calls = []
            for i, tc in enumerate(repair_result.tool_calls or []):
                func = tc.get("function", {})
                repaired_tool_calls.append(
                    DeltaToolCall(
                        id=tc.get("id"),
                        type=tc.get("type", "function"),
                        index=i,
                        function=DeltaFunctionCall(
                            name=func.get("name"),
                            arguments=func.get("arguments"),
                        ),
                    )
                )
            delta.tool_calls = repaired_tool_calls

        return original_chunk
