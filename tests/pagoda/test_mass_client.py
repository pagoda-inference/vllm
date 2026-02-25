# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda MASS platform API client."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.pagoda.mass_client import CacheEntry, MassApiClient, TenantConfig


def _make_defaults() -> TenantConfig:
    """Create default tenant config for testing."""
    return TenantConfig(
        tenant_id="__default__",
        priority="normal",
        qps_limit=10.0,
        concurrent_limit=5,
        kv_cache_block_quota=100,
    )


class TestMassApiClient:
    """Tests for MassApiClient."""

    def setup_method(self):
        """Create a fresh client for each test."""
        self.defaults = _make_defaults()
        self.client = MassApiClient(
            mass_api_url="http://mass:8080/api",
            timeout_seconds=2,
            ttl_seconds=300,
            stale_ttl_seconds=3600,
            defaults=self.defaults,
        )

    @pytest.mark.asyncio
    async def test_cache_hit_no_http(self):
        """Cache hit within TTL returns cached config without HTTP call."""
        cached = TenantConfig(
            tenant_id="t1", priority="high",
            qps_limit=20.0, concurrent_limit=10, kv_cache_block_quota=200,
        )
        self.client._cache["t1"] = CacheEntry(
            config=cached, fetched_at=time.time(),
        )

        with patch.object(self.client, "_fetch_from_mass") as mock_fetch:
            result = await self.client.get_tenant_config("t1")
            mock_fetch.assert_not_called()
        assert result is cached

    @pytest.mark.asyncio
    async def test_cache_expired_refreshes(self):
        """Expired cache triggers MASS API fetch."""
        old_config = TenantConfig(
            tenant_id="t1", priority="normal",
            qps_limit=10.0, concurrent_limit=5, kv_cache_block_quota=100,
        )
        self.client._cache["t1"] = CacheEntry(
            config=old_config, fetched_at=time.time() - 400,  # expired
        )
        new_config = TenantConfig(
            tenant_id="t1", priority="high",
            qps_limit=50.0, concurrent_limit=20, kv_cache_block_quota=500,
        )

        with patch.object(
            self.client, "_fetch_from_mass", new_callable=AsyncMock,
            return_value=new_config,
        ):
            result = await self.client.get_tenant_config("t1")
        assert result is new_config
        assert self.client._cache["t1"].config is new_config

    @pytest.mark.asyncio
    async def test_mass_unavailable_stale_cache(self):
        """MASS unavailable → serve from stale cache."""
        stale_config = TenantConfig(
            tenant_id="t1", priority="normal",
            qps_limit=10.0, concurrent_limit=5, kv_cache_block_quota=100,
        )
        # Expired past TTL but within stale TTL
        self.client._cache["t1"] = CacheEntry(
            config=stale_config, fetched_at=time.time() - 400,
        )

        with patch.object(
            self.client, "_fetch_from_mass", new_callable=AsyncMock,
            return_value=None,
        ), patch.object(self.client, "_alert_mass_unavailable"):
            result = await self.client.get_tenant_config("t1")
        assert result is stale_config
        assert self.client._cache["t1"].is_stale is True

    @pytest.mark.asyncio
    async def test_stale_cache_also_expired_returns_defaults(self):
        """Stale cache also expired → returns defaults."""
        old_config = TenantConfig(
            tenant_id="t1", priority="normal",
            qps_limit=10.0, concurrent_limit=5, kv_cache_block_quota=100,
        )
        # Expired past both TTL and stale TTL
        self.client._cache["t1"] = CacheEntry(
            config=old_config, fetched_at=time.time() - 4000,
        )

        with patch.object(
            self.client, "_fetch_from_mass", new_callable=AsyncMock,
            return_value=None,
        ), patch.object(self.client, "_alert_mass_unavailable"):
            result = await self.client.get_tenant_config("t1")
        assert result.tenant_id == "t1"
        assert result.priority == self.defaults.priority
        assert result.qps_limit == self.defaults.qps_limit

    @pytest.mark.asyncio
    async def test_404_returns_defaults(self):
        """MASS 404 → _fetch_from_mass returns default config."""
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await self.client._fetch_from_mass("unknown-tenant")

        assert result is not None
        assert result.tenant_id == "unknown-tenant"
        assert result.priority == self.defaults.priority

    @pytest.mark.asyncio
    async def test_non_200_returns_none(self):
        """Non-200/404 response → returns None (degradation)."""
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await self.client._fetch_from_mass("t1")

        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        """Timeout → returns None (degradation)."""
        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )
            result = await self.client._fetch_from_mass("t1")

        assert result is None

    def test_invalidate_clears_cache(self):
        """invalidate() removes tenant from cache."""
        self.client._cache["t1"] = CacheEntry(
            config=_make_defaults(), fetched_at=time.time(),
        )
        assert "t1" in self.client._cache
        self.client.invalidate("t1")
        assert "t1" not in self.client._cache

    def test_invalidate_nonexistent_no_error(self):
        """invalidate() on missing tenant doesn't raise."""
        self.client.invalidate("nonexistent")

    @pytest.mark.asyncio
    async def test_cache_miss_fetches_from_mass(self):
        """Cache miss triggers MASS API fetch."""
        new_config = TenantConfig(
            tenant_id="t1", priority="high",
            qps_limit=50.0, concurrent_limit=20, kv_cache_block_quota=500,
        )

        with patch.object(
            self.client, "_fetch_from_mass", new_callable=AsyncMock,
            return_value=new_config,
        ):
            result = await self.client.get_tenant_config("t1")
        assert result is new_config
        assert "t1" in self.client._cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
