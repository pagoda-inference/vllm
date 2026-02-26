# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tenant KV cache block quota manager for Pagoda.

Tracks KV cache block usage per tenant and enforces quotas to prevent
resource monopolization. This is a synchronous implementation that reads
tenant config from Request objects rather than making async API calls.
"""

from __future__ import annotations

import threading
from collections import defaultdict

from vllm.logger import init_logger

logger = init_logger(__name__)


class BlockQuotaManager:
    """Manages per-tenant KV cache block quotas.

    Tracks current block usage per tenant and enforces quota limits.
    Quotas are read from tenant_config stored in Request objects.

    Thread-safety: Currently used from a single async event-loop thread,
    but a defensive lock is included for future multi-threaded callers.
    """

    def __init__(self) -> None:
        """Initialize the block quota manager."""
        self._lock = threading.Lock()
        self._usage: dict[str, int] = defaultdict(int)

    def can_allocate(self, tenant_id: str, num_blocks: int, quota: int) -> bool:
        """Check if a tenant can allocate the requested number of blocks.

        Args:
            tenant_id: Tenant identifier
            num_blocks: Number of blocks to allocate
            quota: Tenant's KV cache block quota

        Returns:
            True if allocation is within quota, False otherwise
        """
        with self._lock:
            current = self._usage[tenant_id]
            return current + num_blocks <= quota

    def try_allocate(
        self, tenant_id: str, num_blocks: int, quota: int
    ) -> bool:
        """Atomically check quota and allocate blocks if within limit.

        Args:
            tenant_id: Tenant identifier
            num_blocks: Number of blocks to allocate
            quota: Tenant's KV cache block quota

        Returns:
            True if allocation succeeded, False if it would exceed quota
        """
        with self._lock:
            current = self._usage[tenant_id]
            if current + num_blocks > quota:
                return False
            self._usage[tenant_id] = current + num_blocks
            return True

    def allocate(self, tenant_id: str, num_blocks: int) -> None:
        """Record block allocation for a tenant.

        Args:
            tenant_id: Tenant identifier
            num_blocks: Number of blocks allocated
        """
        with self._lock:
            self._usage[tenant_id] += num_blocks

    def release(self, tenant_id: str, num_blocks: int) -> None:
        """Record block release for a tenant.

        Args:
            tenant_id: Tenant identifier
            num_blocks: Number of blocks released
        """
        with self._lock:
            self._usage[tenant_id] -= num_blocks
            if self._usage[tenant_id] < 0:
                logger.warning(
                    "Tenant %s has negative block usage: %d. Resetting to 0.",
                    tenant_id,
                    self._usage[tenant_id],
                )
                self._usage[tenant_id] = 0

    def get_usage(self, tenant_id: str) -> int:
        """Get current block usage for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Current number of blocks allocated to the tenant
        """
        with self._lock:
            return self._usage[tenant_id]

    def get_all_usage(self) -> dict[str, int]:
        """Get current block usage for all tenants.

        Returns:
            Dictionary mapping tenant_id to current block count
        """
        with self._lock:
            return dict(self._usage)
