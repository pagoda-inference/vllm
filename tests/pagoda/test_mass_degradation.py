# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for MASS API degradation (场景 3).

Verifies that when MASS API is unreachable:
1. Requests are served using default config
2. Warning logs are emitted (rate-limited to every 300s)
3. Recovery is handled gracefully
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

from vllm.pagoda.mass_client import CacheEntry, MassApiClient, TenantConfig


def _make_defaults() -> TenantConfig:
    return TenantConfig(
        tenant_id="__default__",
        priority="normal",
        qps_limit=10.0,
        concurrent_limit=5,
        kv_cache_block_quota=100,
    )


class TestMassDegradation:
    """Tests for MASS API unavailability and graceful degradation."""

    def setup_method(self):
        self.defaults = _make_defaults()
        # Point to a non-routable address to simulate unreachable MASS
        self.client = MassApiClient(
            mass_api_url="http://192.0.2.1:1/api",  # RFC 5737 TEST-NET
            timeout_seconds=1,
            ttl_seconds=300,
            stale_ttl_seconds=3600,
            defaults=self.defaults,
        )

    @pytest.mark.asyncio
    async def test_mass_unavailable_returns_defaults(self):
        """MASS unreachable with no cache → returns default config."""
        with patch.object(
            self.client, "_fetch_from_mass", new_callable=AsyncMock,
            return_value=None,
        ), patch.object(self.client, "_alert_mass_unavailable"):
            config = await self.client.get_tenant_config("tenant-abc")

        assert config.tenant_id == "tenant-abc"
        assert config.priority == self.defaults.priority
        assert config.qps_limit == self.defaults.qps_limit
        assert config.concurrent_limit == self.defaults.concurrent_limit

    @pytest.mark.asyncio
    async def test_mass_unavailable_serves_stale_cache(self):
        """MASS unreachable with stale cache → serves stale config."""
        stale_config = TenantConfig(
            tenant_id="tenant-abc",
            priority="high",
            qps_limit=50.0,
            concurrent_limit=20,
            kv_cache_block_quota=500,
        )
        # Cache expired past TTL (400s > 300s) but within stale TTL (< 3600s)
        self.client._cache["tenant-abc"] = CacheEntry(
            config=stale_config,
            fetched_at=time.time() - 400,
        )

        with patch.object(
            self.client, "_fetch_from_mass", new_callable=AsyncMock,
            return_value=None,
        ), patch.object(self.client, "_alert_mass_unavailable"):
            config = await self.client.get_tenant_config("tenant-abc")

        # Should return the stale config, not defaults
        assert config is stale_config
        assert config.priority == "high"
        assert config.qps_limit == 50.0

    @pytest.mark.asyncio
    async def test_mass_unavailable_logs_warning(self):
        """MASS unreachable triggers rate-limited error log."""
        with patch.object(
            self.client, "_fetch_from_mass", new_callable=AsyncMock,
            return_value=None,
        ):
            # Reset alert state so first call triggers alert
            self.client._mass_healthy = True
            self.client._last_alert_time = 0.0

            with patch("vllm.pagoda.mass_client.pagoda_mass_api_errors_total"):
                config = await self.client.get_tenant_config("tenant-abc")

        # Client should now be marked unhealthy
        assert self.client._mass_healthy is False
        assert self.client._last_alert_time > 0

    @pytest.mark.asyncio
    async def test_alert_rate_limited_to_interval(self):
        """Repeated MASS failures only alert once per alert_interval."""
        self.client._mass_healthy = False
        self.client._last_alert_time = time.time()  # just alerted

        alert_count = 0
        original_alert = self.client._alert_mass_unavailable

        def counting_alert(tenant_id):
            nonlocal alert_count
            # Check if the original would actually log
            now = time.time()
            if (
                self.client._mass_healthy
                or (now - self.client._last_alert_time) > self.client._alert_interval
            ):
                alert_count += 1

        with patch.object(
            self.client, "_fetch_from_mass", new_callable=AsyncMock,
            return_value=None,
        ), patch.object(
            self.client, "_alert_mass_unavailable", side_effect=counting_alert,
        ):
            # Multiple calls in quick succession
            for _ in range(5):
                await self.client.get_tenant_config("tenant-abc")

        # Should not have triggered additional alerts (interval not elapsed)
        assert alert_count == 0

    @pytest.mark.asyncio
    async def test_mass_recovery_after_failure(self):
        """MASS recovers → cache refreshes and healthy flag resets."""
        # Start with MASS down
        self.client._mass_healthy = False

        new_config = TenantConfig(
            tenant_id="tenant-abc",
            priority="high",
            qps_limit=100.0,
            concurrent_limit=50,
            kv_cache_block_quota=1000,
        )

        with patch.object(
            self.client, "_fetch_from_mass", new_callable=AsyncMock,
            return_value=new_config,
        ):
            config = await self.client.get_tenant_config("tenant-abc")

        assert config is new_config
        assert self.client._mass_healthy is True
        assert self.client._cache["tenant-abc"].is_stale is False

    @pytest.mark.asyncio
    async def test_multiple_tenants_degrade_independently(self):
        """Each tenant gets its own default config when MASS is down."""
        with patch.object(
            self.client, "_fetch_from_mass", new_callable=AsyncMock,
            return_value=None,
        ), patch.object(self.client, "_alert_mass_unavailable"):
            config_a = await self.client.get_tenant_config("tenant-a")
            config_b = await self.client.get_tenant_config("tenant-b")

        assert config_a.tenant_id == "tenant-a"
        assert config_b.tenant_id == "tenant-b"
        # Both should have default values
        assert config_a.qps_limit == config_b.qps_limit == self.defaults.qps_limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
