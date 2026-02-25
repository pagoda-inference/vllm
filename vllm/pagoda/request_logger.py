# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured per-request JSON logging for Pagoda.

Emits one JSON log line per completed request to stdout, compatible with
standard log aggregation systems (ELK, Loki, CloudWatch, etc.).

Usage:
    logger = PagodaRequestLogger()
    logger.log(PagodaRequestLog(...))
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class PagodaRequestLog:
    """Structured log entry for a single completed request.

    All fields map to the spec in vllm-dev-prompt.md Phase 4 Task 13.
    """

    request_id: str
    tenant_id: str
    model: str
    priority: str  # high | normal | batch
    prompt_tokens: int = 0
    output_tokens: int = 0
    queue_wait_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0
    status: str = "success"  # success | error | rejected
    tool_call: bool = False
    tool_call_repaired: bool = False
    preempted: bool = False
    kv_cache_blocks_used: int = 0
    timestamp: str = field(default_factory=lambda: "")
    error_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )


class PagodaRequestLogger:
    """Emit structured JSON log lines for every completed request.

    Uses a dedicated Python logger so output can be routed independently
    of vLLM's main logger (e.g., to a separate file or stdout stream).
    """

    def __init__(self, logger_name: str = "pagoda.request_log") -> None:
        self._logger = logging.getLogger(logger_name)
        # Avoid duplicate handlers if instantiated multiple times
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
        # Don't propagate to root logger to avoid double-printing
        self._logger.propagate = False

    def log(self, entry: PagodaRequestLog) -> None:
        """Emit a single JSON log line for a completed request."""
        data = asdict(entry)
        # Drop None fields for cleaner output
        data = {k: v for k, v in data.items() if v is not None}
        self._logger.info(json.dumps(data, ensure_ascii=False))


class RequestTimer:
    """Convenience timer for tracking request lifecycle phases.

    Usage:
        timer = RequestTimer()
        # ... request queued ...
        timer.mark_inference_start()
        # ... inference runs ...
        timer.mark_done()
        timer.queue_wait_ms   # time in queue
        timer.inference_ms    # time in inference
        timer.total_ms        # total wall time
    """

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._inference_start: float | None = None
        self._end: float | None = None

    def mark_inference_start(self) -> None:
        """Mark the transition from queue wait to active inference."""
        self._inference_start = time.monotonic()

    def mark_done(self) -> None:
        """Mark request completion."""
        self._end = time.monotonic()

    @property
    def queue_wait_ms(self) -> float:
        """Milliseconds spent waiting in queue."""
        if self._inference_start is None:
            return 0.0
        return (self._inference_start - self._start) * 1000.0

    @property
    def inference_ms(self) -> float:
        """Milliseconds spent in inference."""
        if self._inference_start is None or self._end is None:
            return 0.0
        return (self._end - self._inference_start) * 1000.0

    @property
    def total_ms(self) -> float:
        """Total milliseconds from request arrival to completion."""
        end = self._end if self._end is not None else time.monotonic()
        return (end - self._start) * 1000.0
