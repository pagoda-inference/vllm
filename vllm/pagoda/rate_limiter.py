# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-bucket QPS rate limiter + per-tenant concurrency limiter.

Design: config and state are separated so hot-reload updates config
parameters without destroying in-flight state.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.pagoda.config import PagodaConfig
from vllm.pagoda.metrics import pagoda_rate_limit_rejected_total

logger = init_logger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate-limit acquire attempt."""
    allowed: bool
    retry_after: float | None = None
    reason: str | None = None  # "qps_limit" | "concurrent_limit"
    semaphore_ref: asyncio.Semaphore | None = None


@dataclass
class _TenantBucket:
    """Per-tenant rate-limit state. Mutable — updated in place."""
    # Token bucket state
    tokens: float
    last_refill: float
    # Config (can be hot-updated)
    qps_limit: float
    # Concurrency
    semaphore: asyncio.Semaphore
    concurrent_limit: int = field(init=False)

    def __post_init__(self) -> None:
        self.concurrent_limit = self.semaphore._value  # type: ignore[attr-defined]


class TenantRateLimiter:
    """Per-tenant QPS + concurrency rate limiter."""

    def __init__(self, config: PagodaConfig) -> None:
        self._config = config
        self._buckets: dict[str, _TenantBucket] = {}
        self._lock = asyncio.Lock()

    def _get_or_create_bucket(self, tenant_id: str) -> _TenantBucket:
        """Get existing bucket or create one from config. Must hold _lock."""
        bucket = self._buckets.get(tenant_id)
        if bucket is not None:
            return bucket

        tc = self._config.get_tenant_config(tenant_id)
        bucket = _TenantBucket(
            tokens=tc.qps_limit,  # start full
            last_refill=time.monotonic(),
            qps_limit=tc.qps_limit,
            semaphore=asyncio.Semaphore(tc.concurrent_limit),
        )
        self._buckets[tenant_id] = bucket
        return bucket

    async def acquire(self, tenant_id: str) -> RateLimitResult:
        """Try to acquire a rate-limit slot.

        Returns RateLimitResult with allowed=True and semaphore_ref on success.
        The caller MUST call release() with the returned semaphore_ref.
        """
        async with self._lock:
            bucket = self._get_or_create_bucket(tenant_id)

            # --- QPS token bucket ---
            now = time.monotonic()
            elapsed = now - bucket.last_refill
            bucket.tokens = min(
                bucket.qps_limit,  # cap at burst size = qps_limit
                bucket.tokens + elapsed * bucket.qps_limit,
            )
            bucket.last_refill = now

            if bucket.tokens < 1.0:
                retry_after = (1.0 - bucket.tokens) / bucket.qps_limit
                pagoda_rate_limit_rejected_total.labels(
                    tenant_id=tenant_id, reason="qps_limit"
                ).inc()
                return RateLimitResult(
                    allowed=False,
                    retry_after=retry_after,
                    reason="qps_limit",
                )

            # --- Concurrency semaphore ---
            # Snapshot the current semaphore before trying to acquire
            sem_ref = bucket.semaphore
            acquired = sem_ref._value > 0  # type: ignore[attr-defined]
            if not acquired:
                pagoda_rate_limit_rejected_total.labels(
                    tenant_id=tenant_id, reason="concurrent_limit"
                ).inc()
                return RateLimitResult(
                    allowed=False,
                    retry_after=1.0,
                    reason="concurrent_limit",
                )

            # Consume 1 QPS token
            bucket.tokens -= 1.0
            # Acquire semaphore (non-blocking, we checked above)
            sem_ref.acquire_nowait()

        return RateLimitResult(
            allowed=True,
            semaphore_ref=sem_ref,
        )

    async def release(
        self, tenant_id: str, semaphore_ref: asyncio.Semaphore
    ) -> None:
        """Release the concurrency semaphore.

        MUST pass the same semaphore_ref returned by acquire() to ensure
        correct release even after config hot-reload replaces the semaphore.
        """
        semaphore_ref.release()

    def reload_config(self) -> None:
        """Update bucket configs from current PagodaConfig.

        Called after config file reload. Updates config params in-place;
        state (tokens, last_refill) is preserved.
        """
        for tenant_id, bucket in self._buckets.items():
            tc = self._config.get_tenant_config(tenant_id)

            # Update QPS limit — takes effect on next acquire()
            bucket.qps_limit = tc.qps_limit

            # Update concurrency limit — replace semaphore object
            if tc.concurrent_limit != bucket.concurrent_limit:
                logger.info(
                    "Tenant %s concurrent_limit changed %d -> %d",
                    tenant_id,
                    bucket.concurrent_limit,
                    tc.concurrent_limit,
                )
                bucket.semaphore = asyncio.Semaphore(tc.concurrent_limit)
                bucket.concurrent_limit = tc.concurrent_limit
