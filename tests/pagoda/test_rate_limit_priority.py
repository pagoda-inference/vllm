# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for RateLimitResult.priority field propagation (Issue #4).

Verifies that:
1. RateLimitResult carries the tenant priority from MASS config
2. Priority is set on both allowed and rejected results
3. Middleware stores priority in scope state for downstream use
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.pagoda.mass_client import MassApiClient, TenantConfig
from vllm.pagoda.rate_limiter import RateLimitResult, TenantRateLimiter


def _make_tenant_config(
    tenant_id: str = "t1",
    priority: str = "normal",
    qps_limit: float = 10.0,
    concurrent_limit: int = 5,
) -> TenantConfig:
    return TenantConfig(
        tenant_id=tenant_id,
        priority=priority,
        qps_limit=qps_limit,
        concurrent_limit=concurrent_limit,
        kv_cache_block_quota=100,
    )


class TestRateLimitResultPriority:
    """Tests for priority field on RateLimitResult."""

    @pytest.mark.asyncio
    @patch("vllm.pagoda.rate_limiter.pagoda_rate_limit_rejected_total")
    async def test_allowed_result_carries_priority(self, mock_metric):
        """Allowed result has priority from MASS config."""
        mock_mass = AsyncMock(spec=MassApiClient)
        mock_mass.get_tenant_config = AsyncMock(
            return_value=_make_tenant_config(priority="high"),
        )
        limiter = TenantRateLimiter(mock_mass)

        result = await limiter.acquire("t1")

        assert result.allowed is True
        assert result.priority == "high"

    @pytest.mark.asyncio
    @patch("vllm.pagoda.rate_limiter.pagoda_rate_limit_rejected_total")
    async def test_rejected_qps_result_carries_priority(self, mock_metric):
        """QPS-rejected result still has priority from MASS config."""
        mock_mass = AsyncMock(spec=MassApiClient)
        mock_mass.get_tenant_config = AsyncMock(
            return_value=_make_tenant_config(
                priority="batch", qps_limit=1.0, concurrent_limit=100,
            ),
        )
        limiter = TenantRateLimiter(mock_mass)

        # Exhaust the single token
        r1 = await limiter.acquire("t1")
        assert r1.allowed is True
        assert r1.priority == "batch"

        # Second call rejected
        r2 = await limiter.acquire("t1")
        assert r2.allowed is False
        assert r2.reason == "qps_limit"
        assert r2.priority == "batch"

    @pytest.mark.asyncio
    @patch("vllm.pagoda.rate_limiter.pagoda_rate_limit_rejected_total")
    async def test_rejected_concurrent_result_carries_priority(self, mock_metric):
        """Concurrency-rejected result still has priority from MASS config."""
        mock_mass = AsyncMock(spec=MassApiClient)
        mock_mass.get_tenant_config = AsyncMock(
            return_value=_make_tenant_config(
                priority="high", qps_limit=100.0, concurrent_limit=1,
            ),
        )
        limiter = TenantRateLimiter(mock_mass)

        r1 = await limiter.acquire("t1")
        assert r1.allowed is True

        r2 = await limiter.acquire("t1")
        assert r2.allowed is False
        assert r2.reason == "concurrent_limit"
        assert r2.priority == "high"

    @pytest.mark.asyncio
    @patch("vllm.pagoda.rate_limiter.pagoda_rate_limit_rejected_total")
    async def test_different_tenants_different_priorities(self, mock_metric):
        """Different tenants get their own priority values."""
        mock_mass = AsyncMock(spec=MassApiClient)

        async def config_by_tenant(tenant_id):
            priorities = {"t-high": "high", "t-batch": "batch"}
            return _make_tenant_config(
                tenant_id=tenant_id,
                priority=priorities.get(tenant_id, "normal"),
            )

        mock_mass.get_tenant_config = AsyncMock(side_effect=config_by_tenant)
        limiter = TenantRateLimiter(mock_mass)

        r_high = await limiter.acquire("t-high")
        r_batch = await limiter.acquire("t-batch")

        assert r_high.priority == "high"
        assert r_batch.priority == "batch"

    @pytest.mark.asyncio
    @patch("vllm.pagoda.rate_limiter.pagoda_rate_limit_rejected_total")
    async def test_default_priority_when_mass_returns_normal(self, mock_metric):
        """Normal priority is the default from MASS."""
        mock_mass = AsyncMock(spec=MassApiClient)
        mock_mass.get_tenant_config = AsyncMock(
            return_value=_make_tenant_config(priority="normal"),
        )
        limiter = TenantRateLimiter(mock_mass)

        result = await limiter.acquire("t1")
        assert result.priority == "normal"


class TestPriorityInMiddleware:
    """Tests for priority propagation through middleware to scope state."""

    @pytest.mark.asyncio
    @patch("vllm.pagoda.middleware.pagoda_request_total")
    @patch("vllm.pagoda.middleware.pagoda_request_latency_seconds")
    @patch("vllm.pagoda.middleware.pagoda_tenant_concurrent_requests")
    async def test_priority_stored_in_scope_state(
        self, mock_conc, mock_lat, mock_total,
    ):
        """Middleware stores tenant priority in scope['state']."""
        from vllm.pagoda.middleware import PagodaMiddleware

        app = AsyncMock()
        config = MagicMock()
        config.get_queue_config.return_value = MagicMock(
            max_pending_requests=100, warn_threshold_pct=0.8,
        )
        mass_client = AsyncMock()

        with patch("vllm.pagoda.middleware.TenantRateLimiter") as mock_rl_cls, \
             patch("vllm.pagoda.middleware.QueueDepthTracker") as mock_qd_cls, \
             patch("vllm.pagoda.middleware.PagodaRequestLogger"):

            mock_rl = AsyncMock()
            mock_rl.acquire = AsyncMock(return_value=RateLimitResult(
                allowed=True, semaphore_ref=MagicMock(), priority="high",
            ))
            mock_rl.release = AsyncMock()
            mock_rl_cls.return_value = mock_rl

            mock_qd = MagicMock()
            mock_qd.acquire.return_value = True
            mock_qd.current_depth = 5
            mock_qd_cls.return_value = mock_qd

            mw = PagodaMiddleware(
                app=app, config=config, mass_client=mass_client,
            )

        scope = {
            "type": "http",
            "path": "/v1/chat/completions",
            "headers": [(b"x-tenant-id", b"acme")],
            "state": {},
        }

        async def fake_app(s, r, send_fn):
            await send_fn({
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            })
            await send_fn({"type": "http.response.body", "body": b""})

        mw.app = fake_app

        await mw(scope, AsyncMock(), AsyncMock())

        assert scope["state"]["pagoda_tenant_priority"] == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
