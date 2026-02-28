# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda global queue depth protection."""

import logging
from unittest.mock import patch

import pytest

from vllm.pagoda.queue_depth import QueueDepthTracker


class TestQueueDepthTracker:
    """Tests for QueueDepthTracker."""

    def setup_method(self):
        """Create a fresh tracker for each test."""
        with patch("vllm.pagoda.queue_depth.pagoda_queue_depth_current"), \
             patch("vllm.pagoda.queue_depth.pagoda_queue_rejected_total"):
            self.tracker = QueueDepthTracker(max_pending=5, warn_pct=0.8)

    def test_acquire_increments_depth(self):
        """acquire() should increment current_depth."""
        with patch("vllm.pagoda.queue_depth.pagoda_queue_depth_current"), \
             patch("vllm.pagoda.queue_depth.pagoda_queue_rejected_total"):
            assert self.tracker.current_depth == 0
            assert self.tracker.acquire() is True
            assert self.tracker.current_depth == 1
            assert self.tracker.acquire() is True
            assert self.tracker.current_depth == 2

    def test_acquire_rejects_when_full(self):
        """acquire() returns False when max_pending is reached."""
        with patch("vllm.pagoda.queue_depth.pagoda_queue_depth_current"), \
             patch("vllm.pagoda.queue_depth.pagoda_queue_rejected_total"):
            for _ in range(5):
                assert self.tracker.acquire() is True
            assert self.tracker.current_depth == 5
            assert self.tracker.acquire() is False
            assert self.tracker.current_depth == 5

    def test_release_decrements_depth(self):
        """release() should decrement current_depth."""
        with patch("vllm.pagoda.queue_depth.pagoda_queue_depth_current"), \
             patch("vllm.pagoda.queue_depth.pagoda_queue_rejected_total"):
            self.tracker.acquire()
            self.tracker.acquire()
            assert self.tracker.current_depth == 2
            self.tracker.release()
            assert self.tracker.current_depth == 1

    def test_release_does_not_go_below_zero(self):
        """release() clamps to 0."""
        with patch("vllm.pagoda.queue_depth.pagoda_queue_depth_current"), \
             patch("vllm.pagoda.queue_depth.pagoda_queue_rejected_total"):
            self.tracker.release()
            assert self.tracker.current_depth == 0
            self.tracker.release()
            assert self.tracker.current_depth == 0

    def test_warn_threshold_triggers_log(self, caplog):
        """Exceeding warn_threshold logs a warning."""
        # Ensure the logger propagates to caplog
        import logging
        logger = logging.getLogger("vllm.pagoda.queue_depth")
        logger.propagate = True

        with patch("vllm.pagoda.queue_depth.pagoda_queue_depth_current"), \
             patch("vllm.pagoda.queue_depth.pagoda_queue_rejected_total"), \
             caplog.at_level(logging.WARNING, logger="vllm.pagoda.queue_depth"):
            # warn_pct=0.8, max_pending=5 → threshold = 4
            for _ in range(3):
                self.tracker.acquire()
            # 4th acquire crosses threshold (4 >= 4)
            self.tracker.acquire()

            # Check if warning was logged
            assert any("warning threshold" in r.message.lower()
                       for r in caplog.records), \
                f"Expected warning log, got: {[r.message for r in caplog.records]}"

    def test_max_pending_property(self):
        """max_pending property returns configured value."""
        assert self.tracker.max_pending == 5

    def test_acquire_release_cycle(self):
        """Full acquire-release cycle allows re-acquire."""
        with patch("vllm.pagoda.queue_depth.pagoda_queue_depth_current"), \
             patch("vllm.pagoda.queue_depth.pagoda_queue_rejected_total"):
            for _ in range(5):
                self.tracker.acquire()
            assert self.tracker.acquire() is False
            self.tracker.release()
            assert self.tracker.acquire() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
