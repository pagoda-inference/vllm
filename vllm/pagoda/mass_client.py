# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MASS platform API client with TTL cache and graceful degradation."""

from __future__ import annotations

import asyncio
import os
import ssl
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class TenantConfig:
    """Tenant configuration fetched from MASS platform."""
    tenant_id: str
    priority: str           # high | normal | batch
    qps_limit: float
    concurrent_limit: int
    kv_cache_block_quota: int


@dataclass
class CacheEntry:
    """Cached tenant config with freshness tracking."""
    config: TenantConfig
    fetched_at: float       # time.time() when fetched
    is_stale: bool = False  # True when serving from degraded cache


class MassApiClient:
    """Fetch tenant config from MASS platform with local TTL cache.

    Degradation strategy when MASS is unavailable:
    1. Serve from stale cache (up to stale_ttl_seconds)
    2. Fall back to defaults
    3. Rate-limited alerting to avoid log storms
    """

    def __init__(
        self,
        mass_api_url: str,
        timeout_seconds: int = 2,
        ttl_seconds: int = 300,
        stale_ttl_seconds: int = 3600,
        defaults: TenantConfig | None = None,
        api_key: str | None = None,
        client_cert_path: str | None = None,
        client_key_path: str | None = None,
        ca_cert_path: str | None = None,
    ) -> None:
        self._mass_api_url = mass_api_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._ttl = ttl_seconds
        self._stale_ttl = stale_ttl_seconds
        self._defaults = defaults

        # Auth: Bearer token (prefer env var over config)
        self._api_key = api_key or os.environ.get("PAGODA_MASS_API_KEY")

        # Auth: mTLS client certificate
        self._ssl_context: ssl.SSLContext | None = None
        client_cert = client_cert_path or os.environ.get(
            "PAGODA_MASS_CLIENT_CERT"
        )
        client_key = client_key_path or os.environ.get(
            "PAGODA_MASS_CLIENT_KEY"
        )
        ca_cert = ca_cert_path or os.environ.get("PAGODA_MASS_CA_CERT")
        if client_cert and client_key:
            self._ssl_context = ssl.create_default_context(
                purpose=ssl.Purpose.SERVER_AUTH,
                cafile=ca_cert,
            )
            self._ssl_context.load_cert_chain(
                certfile=client_cert, keyfile=client_key
            )
            logger.info("MASS API client mTLS enabled (cert=%s)", client_cert)
        elif self._api_key:
            logger.info("MASS API client Bearer token auth enabled")
        else:
            logger.warning(
                "MASS API client has NO authentication configured. "
                "Set PAGODA_MASS_API_KEY or configure mTLS certificates."
            )

        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

        # Alert state: avoid log storms when MASS is down
        self._mass_healthy = True
        self._last_alert_time = 0.0
        self._alert_interval = 300  # max one alert per 300s

    async def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        """Get tenant config. Priority:
        1. Local cache (not expired)
        2. MASS platform API (cache miss or expired)
        3. Stale cache (MASS unavailable)
        4. Defaults
        """
        async with self._lock:
            entry = self._cache.get(tenant_id)
            now = time.time()

            # Cache hit and fresh
            if entry and (now - entry.fetched_at) < self._ttl:
                return entry.config

            # Cache miss or expired — try MASS
            config = await self._fetch_from_mass(tenant_id)

            if config is not None:
                self._cache[tenant_id] = CacheEntry(
                    config=config,
                    fetched_at=now,
                    is_stale=False,
                )
                if not self._mass_healthy:
                    logger.info(
                        "MASS platform recovered, tenant config cache "
                        "refreshing."
                    )
                    self._mass_healthy = True
                return config

            # MASS unavailable — try stale cache
            if entry and (now - entry.fetched_at) < self._stale_ttl:
                self._alert_mass_unavailable(tenant_id)
                entry.is_stale = True
                return entry.config

            # Stale cache also expired — use defaults
            self._alert_mass_unavailable(tenant_id)
            logger.warning(
                "No valid cache for tenant %s, falling back to defaults.",
                tenant_id,
            )
            return self._make_default_config(tenant_id)

    async def _fetch_from_mass(
        self, tenant_id: str, max_retries: int = 2
    ) -> Optional[TenantConfig]:
        """Fetch tenant config from MASS API with exponential backoff.

        Returns None on failure after all retries exhausted.
        """
        url = (
            f"{self._mass_api_url}/internal/tenants/{tenant_id}/config"
        )
        req_headers: dict[str, str] = {}
        if self._api_key:
            req_headers["Authorization"] = f"Bearer {self._api_key}"

        connector = (
            aiohttp.TCPConnector(ssl=self._ssl_context)
            if self._ssl_context
            else None
        )

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession(
                    timeout=self._timeout,
                    connector=connector,
                ) as session:
                    async with session.get(
                        url, headers=req_headers
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return TenantConfig(
                                tenant_id=tenant_id,
                                priority=data.get(
                                    "priority", self._defaults.priority
                                ),
                                qps_limit=data.get(
                                    "qps_limit", self._defaults.qps_limit
                                ),
                                concurrent_limit=data.get(
                                    "concurrent_limit",
                                    self._defaults.concurrent_limit,
                                ),
                                kv_cache_block_quota=data.get(
                                    "kv_cache_block_quota",
                                    self._defaults.kv_cache_block_quota,
                                ),
                            )
                        elif resp.status == 404:
                            logger.info(
                                "Tenant %s not found in MASS, using defaults.",
                                tenant_id,
                            )
                            return self._make_default_config(tenant_id)
                        else:
                            logger.warning(
                                "MASS API returned %d for tenant %s "
                                "(attempt %d/%d)",
                                resp.status,
                                tenant_id,
                                attempt + 1,
                                max_retries + 1,
                            )
            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError()
                logger.warning(
                    "MASS API timeout for tenant %s (attempt %d/%d)",
                    tenant_id,
                    attempt + 1,
                    max_retries + 1,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "MASS API error for tenant %s (attempt %d/%d): %s",
                    tenant_id,
                    attempt + 1,
                    max_retries + 1,
                    e,
                )

            # Exponential backoff before next retry
            if attempt < max_retries:
                await asyncio.sleep(0.1 * (2 ** attempt))

        logger.warning(
            "MASS API exhausted %d retries for tenant %s, last error: %s",
            max_retries + 1,
            tenant_id,
            last_error,
        )
        return None

    def _alert_mass_unavailable(self, tenant_id: str) -> None:
        """Rate-limited alert when MASS is down."""
        now = time.time()
        if (
            self._mass_healthy
            or (now - self._last_alert_time) > self._alert_interval
        ):
            logger.error(
                "MASS platform unavailable, serving tenant configs from "
                "stale cache. Triggered by tenant: %s",
                tenant_id,
            )
            from vllm.pagoda.metrics import pagoda_mass_api_errors_total

            pagoda_mass_api_errors_total.inc()
            self._mass_healthy = False
            self._last_alert_time = now

    def _make_default_config(self, tenant_id: str) -> TenantConfig:
        return TenantConfig(
            tenant_id=tenant_id,
            priority=self._defaults.priority,
            qps_limit=self._defaults.qps_limit,
            concurrent_limit=self._defaults.concurrent_limit,
            kv_cache_block_quota=self._defaults.kv_cache_block_quota,
        )

    def invalidate(self, tenant_id: str) -> None:
        """Invalidate cache for a tenant. For webhook-triggered refresh.

        Instead of removing the entry (which loses the stale fallback),
        we reset fetched_at to 0 so the next get_tenant_config() will
        treat it as expired and re-fetch from MASS, while still keeping
        the stale value available if MASS is unreachable.
        """
        entry = self._cache.get(tenant_id)
        if entry is not None:
            entry.fetched_at = 0.0
            entry.is_stale = True
