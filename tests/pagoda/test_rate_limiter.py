# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda per-tenant QPS + concurrency rate limiter."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from vllm.pagoda.mass_client import MassApiClient, TenantConfig
from vllm.pagoda.rate_limiter import RateLimitResult, TenantRateLimiter


def _make_tenant_config(
    tenant_id: str = "t1",
    qps_limit: float = 10.0,
    concurrent_limit: int = 5,
) -> TenantConfig:
    return TenantConfig(
        tenant_id=tenant_id,
        priority="normal",
        qps_limit=qps_limit,
        concurrent_limit=concurrent_limit,
        kv_cache_block_quota=100,
    )


@pytest.fixture
def mock_mass_client():
    client = AsyncMock(spec=MassApiClient)
    client.get_tenant_config = AsyncMock(return_value=_make_tenant_config())
    return client


class TestTenantRateLimiter:
    """Tests for TenantRateLimiter."""

    @pytest.mark.asyncio
    @patch("vllm.pagoda.rate_limiter.pagoda_rate_limit_rejected_total")
    async def test_first_acquire_allowed(self, mock_metric, mock_mass_client):
        """First acquire should be allowed."""
        limiter = TenantRateLimiter(mock_mass_client)
        result = await limiter.acquire("t1")

        assert result.allowed is True
        assert result.semaphore_ref is not None
        assert result.reason is None

    @pytest.mark.asyncio
    @patch("vllm.pagoda.rate_limiter.pagoda_rate_limit_rejected_total")
    async def test_qps_limit_exceeded(self, mock_metric, mock_mass_client):
        """Exceeding QPS limit should reject with reason=qps_limit."""
        # Use very low QPS limit so we exhaust tokens quickly
        mock_mass_client.get_tenant_config = AsyncMock(
            return_value=_make_tenant_config(qps_limit=2.0, concurrent_limit=100)
        )
        limiter = TenantRateLimiter(mock_mass_client)

        # Acquire twice (bucket starts with 2.0 tokens)
        r1 = await limiter.acquire("t1")
        assert r1.allowed is True
        r2 = await limiter.acquire("t1")
        assert r2.allowed is True

        # Third acquire should fail (no tokens left, no time elapsed)
        r3 = await limiter.acquire("t1")
        assert r3.allowed is False
        assert r3.reason == "qps_limit"
        assert r3.retry_after is not None
        assert r3.retry_after > 0

    @pytest.mark.asyncio
    @patch("vllm.pagoda.rate_limiter.pagoda_rate_limit_rejected_total")
    async def test_concurrent_limit_exceeded(self, mock_metric, mock_mass_client):
        """Exceeding concurrent limit should reject with reason=concurrent_limit."""
        mock_mass_client.get_tenant_config = AsyncMock(
            return_value=_make_tenant_config(qps_limit=100.0, concurrent_limit=2)
        )
        limiter = TenantRateLimiter(mock_mass_client)

        r1 = await limiter.acquire("t1")
        assert r1.allowed is True
        r2 = await limiter.acquire("t1")
        assert r2.allowed is True

        # Third acquire should fail on concurrency
        r3 = await limiter.acquire("t1")
        assert r3.allowed is False
        assert r3.reason == "concurrent_limit"

    @pytest.mark.asyncio
    @patch("vllm.pagoda.rate_limiter.pagoda_rate_limit_rejected_total")
    async def test_release_allows_reacquire(self, mock_metric, mock_mass_client):
        """Releasing semaphore allows new acquire."""
        mock_mass_client.get_tenant_config = AsyncMock(
            return_value=_make_tenant_config(qps_limit=100.0, concurrent_limit=1)
        )
        limiter = TenantRateLimiter(mock_mass_client)

        r1 = await limiter.acquire("t1")
        assert r1.allowed is True

        # Concurrent limit reached
        r2 = await limiter.acquire("t1")
        assert r2.allowed is False

        # Release and retry
        await limiter.release("t1", r1.semaphore_ref)
        r3 = await limiter.acquire("t1")
        assert r3.allowed is True

    @pytest.mark.asyncio
    @patch("vllm.pagoda.rate_limiter.pagoda_rate_limit_rejected_total")
    async def test_different_tenants_independent(self, mock_metric, mock_mass_client):
        """Different tenants have independent buckets."""
        mock_mass_client.get_tenant_config = AsyncMock(
            return_value=_make_tenant_config(qps_limit=100.0, concurrent_limit=1)
        )
        limiter = TenantRateLimiter(mock_mass_client)

        r1 = await limiter.acquire("t1")
        assert r1.allowed is True

        # t1 is at concurrent limit, but t2 should still work
        r2 = await limiter.acquire("t2")
        assert r2.allowed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
