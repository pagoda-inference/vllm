# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for Queue Depth protection (场景 2).

Verifies that requests exceeding max_pending_requests are rejected with 503
through the full middleware stack.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.pagoda.middleware import PagodaMiddleware
from vllm.pagoda.rate_limiter import RateLimitResult


def _make_scope(tenant_id="tenant-a"):
    """Create a minimal ASGI scope for /v1/ path."""
    return {
        "type": "http",
        "path": "/v1/chat/completions",
        "headers": [
            (b"x-tenant-id", tenant_id.encode()),
        ],
        "state": {},
    }


def _make_middleware(max_pending: int):
    """Create PagodaMiddleware with a specific max_pending_requests."""
    app = AsyncMock()
    config = MagicMock()
    config.get_queue_config.return_value = MagicMock(
        max_pending_requests=max_pending,
        warn_threshold_pct=0.8,
    )
    mass_client = AsyncMock()

    with patch("vllm.pagoda.middleware.TenantRateLimiter") as mock_rl_cls, \
         patch("vllm.pagoda.middleware.QueueDepthTracker"), \
         patch("vllm.pagoda.middleware.PagodaRequestLogger"):

        mock_rl = AsyncMock()
        mock_rl.acquire = AsyncMock(return_value=RateLimitResult(
            allowed=True, semaphore_ref=MagicMock(), priority="normal",
        ))
        mock_rl.release = AsyncMock()
        mock_rl_cls.return_value = mock_rl

        mw = PagodaMiddleware(
            app=app,
            config=config,
            mass_client=mass_client,
            trust_upstream_tenant_id=True,
        )

    # Replace queue_tracker with a real one (small limit) to test actual logic
    from vllm.pagoda.queue_depth import QueueDepthTracker
    with patch("vllm.pagoda.queue_depth.pagoda_queue_depth_current"), \
         patch("vllm.pagoda.queue_depth.pagoda_queue_rejected_total"):
        mw.queue_tracker = QueueDepthTracker(
            max_pending=max_pending, warn_pct=0.8,
        )

    return mw


class TestQueueDepthIntegration:
    """Integration tests: concurrent requests exceeding queue depth get 503."""

    @pytest.mark.asyncio
    @patch("vllm.pagoda.middleware.pagoda_request_rejected_total")
    @patch("vllm.pagoda.middleware.pagoda_request_total")
    @patch("vllm.pagoda.middleware.pagoda_request_latency_seconds")
    @patch("vllm.pagoda.middleware.pagoda_tenant_concurrent_requests")
    @patch("vllm.pagoda.queue_depth.pagoda_queue_depth_current")
    @patch("vllm.pagoda.queue_depth.pagoda_queue_rejected_total")
    async def test_concurrent_requests_exceed_queue_depth(
        self, mock_qr, mock_qd, mock_conc, mock_lat, mock_total, mock_rej,
    ):
        """Send max_pending+2 concurrent requests; extras get 503."""
        max_pending = 3
        mw = _make_middleware(max_pending=max_pending)

        # Slow app: blocks until event is set
        gate = asyncio.Event()

        async def slow_app(scope, receive, send_fn):
            await gate.wait()
            await send_fn({
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            })
            await send_fn({"type": "http.response.body", "body": b'{"ok":true}'})

        mw.app = slow_app

        total_requests = max_pending + 2
        captures = [[] for _ in range(total_requests)]

        async def make_send(idx):
            async def send(msg):
                captures[idx].append(msg)
            return send

        # Launch all requests concurrently
        tasks = []
        for i in range(total_requests):
            send_fn = await make_send(i)
            tasks.append(
                asyncio.create_task(
                    mw(_make_scope(f"tenant-{i}"), AsyncMock(), send_fn)
                )
            )

        # Let the event loop schedule all tasks
        await asyncio.sleep(0.05)

        # Release the gate so slow requests complete
        gate.set()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Collect statuses
        statuses = []
        for cap in captures:
            for msg in cap:
                if msg.get("type") == "http.response.start":
                    statuses.append(msg["status"])

        assert statuses.count(503) == 2, (
            f"Expected 2 rejections (503), got statuses: {statuses}"
        )
        assert statuses.count(200) == max_pending, (
            f"Expected {max_pending} successes (200), got statuses: {statuses}"
        )

    @pytest.mark.asyncio
    @patch("vllm.pagoda.middleware.pagoda_request_rejected_total")
    @patch("vllm.pagoda.middleware.pagoda_request_total")
    @patch("vllm.pagoda.middleware.pagoda_request_latency_seconds")
    @patch("vllm.pagoda.middleware.pagoda_tenant_concurrent_requests")
    @patch("vllm.pagoda.queue_depth.pagoda_queue_depth_current")
    @patch("vllm.pagoda.queue_depth.pagoda_queue_rejected_total")
    async def test_503_response_body_format(
        self, mock_qr, mock_qd, mock_conc, mock_lat, mock_total, mock_rej,
    ):
        """503 response has correct JSON error body."""
        mw = _make_middleware(max_pending=0)  # reject everything

        captured = []

        async def send(msg):
            captured.append(msg)

        await mw(_make_scope(), AsyncMock(), send)

        assert len(captured) == 2
        start = captured[0]
        assert start["status"] == 503

        body = json.loads(captured[1]["body"])
        assert body["error"]["code"] == 503
        assert body["error"]["type"] == "server_error"
        assert "queue" in body["error"]["message"].lower()

    @pytest.mark.asyncio
    @patch("vllm.pagoda.middleware.pagoda_request_rejected_total")
    @patch("vllm.pagoda.middleware.pagoda_request_total")
    @patch("vllm.pagoda.middleware.pagoda_request_latency_seconds")
    @patch("vllm.pagoda.middleware.pagoda_tenant_concurrent_requests")
    @patch("vllm.pagoda.queue_depth.pagoda_queue_depth_current")
    @patch("vllm.pagoda.queue_depth.pagoda_queue_rejected_total")
    async def test_queue_releases_after_completion_allows_new_requests(
        self, mock_qr, mock_qd, mock_conc, mock_lat, mock_total, mock_rej,
    ):
        """After a request completes, its queue slot is freed for new requests."""
        mw = _make_middleware(max_pending=1)

        async def fast_app(scope, receive, send_fn):
            await send_fn({
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            })
            await send_fn({"type": "http.response.body", "body": b""})

        mw.app = fast_app

        # First request: should succeed and release the slot
        cap1 = []
        async def send1(msg):
            cap1.append(msg)
        await mw(_make_scope(), AsyncMock(), send1)
        assert cap1[0]["status"] == 200

        # Second request: should also succeed (slot was freed)
        cap2 = []
        async def send2(msg):
            cap2.append(msg)
        await mw(_make_scope(), AsyncMock(), send2)
        assert cap2[0]["status"] == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
