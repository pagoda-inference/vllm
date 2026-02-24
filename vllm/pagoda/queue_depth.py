# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global queue depth protection."""

from __future__ import annotations

import threading

from vllm.logger import init_logger
from vllm.pagoda.metrics import (
    pagoda_queue_depth_current,
    pagoda_queue_rejected_total,
)

logger = init_logger(__name__)


class QueueDepthTracker:
    """Track global pending request count and reject when full."""

    def __init__(self, max_pending: int, warn_pct: float = 0.8) -> None:
        self._max_pending = max_pending
        self._warn_threshold = int(max_pending * warn_pct)
        self._current = 0
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to increment pending count. Returns False if queue is full."""
        with self._lock:
            if self._current >= self._max_pending:
                pagoda_queue_rejected_total.labels(reason="queue_full").inc()
                return False
            self._current += 1
            pagoda_queue_depth_current.set(self._current)

        if self._current >= self._warn_threshold:
            logger.warning(
                "Queue depth %d/%d exceeds warning threshold",
                self._current,
                self._max_pending,
            )
        return True

    def release(self) -> None:
        """Decrement pending count."""
        with self._lock:
            self._current = max(0, self._current - 1)
            pagoda_queue_depth_current.set(self._current)

    @property
    def current_depth(self) -> int:
        return self._current

    @property
    def max_pending(self) -> int:
        return self._max_pending
