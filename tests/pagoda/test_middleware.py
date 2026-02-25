# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda unified ASGI middleware."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.pagoda.middleware import PagodaMiddleware
from vllm.pagoda.rate_limiter import RateLimitResult


def _make_scope(path="/v1/chat/completions", scope_type="http", headers=None):
    """Create a minimal ASGI scope."""
    raw_headers = []
    if headers:
        for k, v in headers.items():
            raw_headers.append((k.encode(), v.encode()))
    return {
        "type": scope_type,
        "path": path,
        "headers": raw_headers,
        "state": {},
    }


def _make_middleware():
    """Create a PagodaMiddleware with mocked dependencies."""
    app = AsyncMock()
    config = MagicMock()
    config.get_queue_config.return_value = MagicMock(
        max_pending_requests=100, warn_threshold_pct=0.8,
    )
    mass_client = AsyncMock()
    mass_client.get_tenant_config = AsyncMock(return_value=MagicMock(
        priority="normal",
    ))

    with patch("vllm.pagoda.middleware.TenantRateLimiter") as mock_rl_cls, \
         patch("vllm.pagoda.middleware.QueueDepthTracker") as mock_qd_cls, \
         patch("vllm.pagoda.middleware.PagodaRequestLogger"):

        mock_rl = AsyncMock()
        mock_rl._client = mass_client
        mock_rl.acquire = AsyncMock(return_value=RateLimitResult(
            allowed=True, semaphore_ref=MagicMock(),
        ))
        mock_rl.release = AsyncMock()
        mock_rl_cls.return_value = mock_rl

        mock_qd = MagicMock()
        mock_qd.acquire.return_value = True
        mock_qd.current_depth = 5
        mock_qd_cls.return_value = mock_qd

        mw = PagodaMiddleware(
            app=app,
            config=config,
            mass_client=mass_client,
            trust_upstream_tenant_id=True,
        )
        # Expose mocks for assertions
        mw._mock_rate_limiter = mock_rl
        mw._mock_queue_tracker = mock_qd
        mw._mock_app = app
    return mw


class TestPagodaMiddleware:
    """Tests for PagodaMiddleware."""

    @pytest.mark.asyncio
    async def test_non_http_passthrough(self):
        """Non-HTTP scope is passed directly to app."""
        mw = _make_middleware()
        scope = _make_scope(scope_type="websocket")
        receive = AsyncMock()
        send = AsyncMock()

        await mw(scope, receive, send)
        mw._mock_app.assert_awaited_once_with(scope, receive, send)

    @pytest.mark.asyncio
    async def test_non_v1_path_passthrough(self):
        """Non-/v1/ path is passed directly to app."""
        mw = _make_middleware()
        scope = _make_scope(path="/health")
        receive = AsyncMock()
        send = AsyncMock()

        await mw(scope, receive, send)
        mw._mock_app.assert_awaited_once_with(scope, receive, send)

    @pytest.mark.asyncio
    @patch("vllm.pagoda.middleware.pagoda_request_total")
    @patch("vllm.pagoda.middleware.pagoda_request_latency_seconds")
    @patch("vllm.pagoda.middleware.pagoda_tenant_concurrent_requests")
    async def test_normal_request_200(self, mock_conc, mock_lat, mock_total):
        """Normal /v1/ request flows through and injects response headers."""
        mw = _make_middleware()
        scope = _make_scope(headers={"x-tenant-id": "acme"})
        receive = AsyncMock()
        sent_messages = []

        async def capture_send(message):
            sent_messages.append(message)

        # Make the app send a response
        async def fake_app(s, r, send_fn):
            await send_fn({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            })
            await send_fn({
                "type": "http.response.body",
                "body": b'{"ok": true}',
            })

        mw.app = fake_app

        await mw(scope, receive, capture_send)

        # Verify response headers were injected
        start_msg = sent_messages[0]
        assert start_msg["type"] == "http.response.start"
        # The middleware wraps send, so headers should include x-queue-depth
        # and x-pagoda-tenant

    @pytest.mark.asyncio
    @patch("vllm.pagoda.middleware.pagoda_request_rejected_total")
    @patch("vllm.pagoda.middleware.pagoda_request_total")
    async def test_rate_limit_429(self, mock_total, mock_rejected):
        """Rate limit rejection sends 429 with retry-after."""
        mw = _make_middleware()
        mw.rate_limiter.acquire = AsyncMock(return_value=RateLimitResult(
            allowed=False, retry_after=2.0, reason="qps_limit",
        ))
        scope = _make_scope(headers={"x-tenant-id": "acme"})
        receive = AsyncMock()
        sent_messages = []

        async def capture_send(message):
            sent_messages.append(message)

        await mw(scope, receive, capture_send)

        # Should have sent error response
        assert len(sent_messages) == 2
        start = sent_messages[0]
        assert start["status"] == 429
        # Check retry-after header
        headers_dict = {k: v for k, v in start["headers"]}
        assert headers_dict[b"retry-after"] == b"2"

        body = json.loads(sent_messages[1]["body"])
        assert body["error"]["code"] == 429
        assert body["error"]["type"] == "rate_limit_error"

    @pytest.mark.asyncio
    @patch("vllm.pagoda.middleware.pagoda_request_rejected_total")
    @patch("vllm.pagoda.middleware.pagoda_request_total")
    async def test_queue_full_503(self, mock_total, mock_rejected):
        """Queue full sends 503."""
        mw = _make_middleware()
        mw.queue_tracker.acquire.return_value = False
        mw.queue_tracker.current_depth = 100
        scope = _make_scope(headers={"x-tenant-id": "acme"})
        receive = AsyncMock()
        sent_messages = []

        async def capture_send(message):
            sent_messages.append(message)

        await mw(scope, receive, capture_send)

        assert len(sent_messages) == 2
        start = sent_messages[0]
        assert start["status"] == 503

        body = json.loads(sent_messages[1]["body"])
        assert body["error"]["code"] == 503
        assert body["error"]["type"] == "server_error"

    @pytest.mark.asyncio
    async def test_tenant_stored_in_scope_state(self):
        """Tenant ID and priority are stored in scope state."""
        mw = _make_middleware()
        scope = _make_scope(headers={"x-tenant-id": "acme"})
        receive = AsyncMock()

        async def fake_app(s, r, send_fn):
            await send_fn({
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            })
            await send_fn({"type": "http.response.body", "body": b""})

        mw.app = fake_app

        with patch("vllm.pagoda.middleware.pagoda_request_total"), \
             patch("vllm.pagoda.middleware.pagoda_request_latency_seconds"), \
             patch("vllm.pagoda.middleware.pagoda_tenant_concurrent_requests"):
            await mw(scope, receive, AsyncMock())

        assert scope["state"]["pagoda_tenant_id"] == "acme"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
