# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tenant-Aware ARC Offloading Manager for multi-tenant KV cache management.

Extends the standard ARC (Adaptive Replacement Cache) algorithm with:
  - Per-tenant T1/T2 tracking and independent ARC adaptation
  - Tenant priority-based eviction ordering
  - Tenant quota enforcement (min_blocks guarantee, max_blocks cap)
  - Global ARC coordination across tenants

Eviction Strategy:
  1. Evict from tenants exceeding max_blocks first
  2. Then evict from lowest-priority tenants first
  3. Within a tenant, use standard ARC eviction (T1 vs T2 based on target)
  4. Never evict below a tenant's min_blocks guarantee (if possible)
"""
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.pagoda.metrics import (
    pagoda_offload_blocks_used,
    pagoda_offload_evictions_total,
    pagoda_offload_hits_total,
    pagoda_offload_misses_total,
    pagoda_offload_stores_total,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus

logger = init_logger(__name__)


@dataclass
class TenantQuota:
    """Quota configuration for a single tenant."""
    min_blocks: int = 0       # guaranteed minimum blocks
    max_blocks: int = 0       # hard cap (0 = unlimited)
    priority: int = 0         # higher = less likely to be evicted
    weight: float = 1.0       # proportional share weight


@dataclass
class TenantState:
    """Per-tenant ARC state."""
    tenant_id: str
    quota: TenantQuota
    # ARC data structures per tenant
    t1: OrderedDict[BlockHash, BlockStatus] = field(
        default_factory=OrderedDict
    )
    t2: OrderedDict[BlockHash, BlockStatus] = field(
        default_factory=OrderedDict
    )
    b1: OrderedDict[BlockHash, None] = field(default_factory=OrderedDict)
    b2: OrderedDict[BlockHash, None] = field(default_factory=OrderedDict)
    target_t1_size: float = 0.0

    @property
    def num_cached_blocks(self) -> int:
        return len(self.t1) + len(self.t2)

    @property
    def is_over_max(self) -> bool:
        return (
            self.quota.max_blocks > 0
            and self.num_cached_blocks > self.quota.max_blocks
        )

    @property
    def is_at_min(self) -> bool:
        return self.num_cached_blocks <= self.quota.min_blocks


class TenantAwareARCOffloadingManager(OffloadingManager):
    """
    Multi-tenant ARC offloading manager.

    Each tenant gets its own T1/T2/B1/B2 lists and adaptive target.
    Eviction respects tenant priorities and quota constraints.

    Block-to-tenant mapping is maintained via a hash→tenant_id index
    so that all OffloadingManager interface methods (which only receive
    BlockHash) can route to the correct tenant state.
    """

    def __init__(
        self,
        backend: Backend,
        tenant_quotas: dict[str, TenantQuota] | None = None,
        enable_events: bool = False,
    ):
        self.backend = backend
        self.cache_capacity: int = self.backend.get_num_free_blocks()
        self.events: list[OffloadingEvent] | None = (
            [] if enable_events else None
        )

        # Per-tenant state
        self.tenants: dict[str, TenantState] = {}
        if tenant_quotas:
            for tid, quota in tenant_quotas.items():
                self.tenants[tid] = TenantState(
                    tenant_id=tid, quota=quota
                )

        # Global index: block_hash → tenant_id for fast routing
        self._block_tenant: dict[BlockHash, str] = {}

        # Default tenant for blocks without explicit tenant assignment
        self._default_tenant_id = "__default__"

    def register_tenant(
        self, tenant_id: str, quota: TenantQuota | None = None
    ):
        """Register a new tenant (or update quota for existing one)."""
        if tenant_id in self.tenants:
            if quota is not None:
                self.tenants[tenant_id].quota = quota
        else:
            self.tenants[tenant_id] = TenantState(
                tenant_id=tenant_id,
                quota=quota or TenantQuota(),
            )

    def _ensure_tenant(self, tenant_id: str) -> TenantState:
        if tenant_id not in self.tenants:
            self.register_tenant(tenant_id)
        return self.tenants[tenant_id]

    def _get_tenant_for_block(self, block_hash: BlockHash) -> TenantState:
        tid = self._block_tenant.get(block_hash, self._default_tenant_id)
        return self._ensure_tenant(tid)

    # ------------------------------------------------------------------
    # OffloadingManager interface
    # ------------------------------------------------------------------

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        block_hash_list = list(block_hashes)
        hit_count = 0
        for block_hash in block_hash_list:
            block = self._find_block(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        # Record per-tenant hit/miss metrics
        if block_hash_list:
            tid = self._block_tenant.get(
                block_hash_list[0], self._default_tenant_id
            )
            if hit_count > 0:
                pagoda_offload_hits_total.labels(tenant_id=tid).inc(
                    hit_count
                )
            miss_count = len(block_hash_list) - hit_count
            if miss_count > 0:
                pagoda_offload_misses_total.labels(tenant_id=tid).inc(
                    miss_count
                )
        return hit_count

    def prepare_load(
        self,
        block_hashes: Iterable[BlockHash],
        tenant_id: str | None = None,
    ) -> LoadStoreSpec:
        blocks = []
        block_hash_list = list(block_hashes)
        for block_hash in block_hash_list:
            block = self._find_block(block_hash)
            assert block is not None, (
                f"Block {block_hash!r} not found in cache"
            )
            assert block.is_ready, (
                f"Block {block_hash!r} is not ready for reading"
            )
            block.ref_cnt += 1
            blocks.append(block)
        return self.backend.get_load_store_spec(block_hash_list, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        for block_hash in reversed(list(block_hashes)):
            ts = self._get_tenant_for_block(block_hash)

            if block_hash in ts.t1:
                block = ts.t1.pop(block_hash)
                if not block.is_ready:
                    ts.t1[block_hash] = block
                else:
                    ts.t2[block_hash] = block

            elif block_hash in ts.t2:
                ts.t2.move_to_end(block_hash)

            elif block_hash in ts.b1:
                delta = max(
                    1, len(ts.b2) / max(len(ts.b1), 1)
                )
                ts.target_t1_size = min(
                    ts.target_t1_size + delta,
                    self.cache_capacity,
                )
                ts.b1.move_to_end(block_hash)

            elif block_hash in ts.b2:
                delta = max(
                    1, len(ts.b1) / max(len(ts.b2), 1)
                )
                ts.target_t1_size = max(ts.target_t1_size - delta, 0)
                ts.b2.move_to_end(block_hash)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for block_hash in block_hashes:
            block = self._find_block(block_hash)
            assert block is not None, (
                f"Block {block_hash!r} not found"
            )
            assert block.ref_cnt > 0, (
                f"Block {block_hash!r} ref_cnt is already 0"
            )
            block.ref_cnt -= 1

    def prepare_store(
        self,
        block_hashes: Iterable[BlockHash],
        tenant_id: str | None = None,
    ) -> PrepareStoreOutput | None:
        tid = tenant_id or self._default_tenant_id
        ts = self._ensure_tenant(tid)

        # Filter out already-cached blocks
        block_hashes_to_store = []
        for bh in block_hashes:
            if self._find_block(bh) is None:
                block_hashes_to_store.append(bh)

        if not block_hashes_to_store:
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=self.backend.get_load_store_spec([], []),
                block_hashes_evicted=[],
            )

        # Check tenant max_blocks cap
        if (
            ts.quota.max_blocks > 0
            and ts.num_cached_blocks + len(block_hashes_to_store)
            > ts.quota.max_blocks
        ):
            # Only store up to the cap
            allowed = max(
                0, ts.quota.max_blocks - ts.num_cached_blocks
            )
            if allowed == 0:
                return None
            block_hashes_to_store = block_hashes_to_store[:allowed]

        # Evict blocks to make room
        num_to_evict = (
            len(block_hashes_to_store)
            - self.backend.get_num_free_blocks()
        )

        to_evict: list[BlockHash] = []
        if num_to_evict > 0:
            to_evict = self._evict_blocks(num_to_evict, protect_tenant=tid)
            if len(to_evict) < num_to_evict:
                # Could not evict enough
                return None

        # Trim ghost lists across all tenants
        for t in self.tenants.values():
            for b in [t.b1, t.b2]:
                while len(b) > self.cache_capacity:
                    bh, _ = b.popitem(last=False)
                    # Remove from tenant mapping when evicted from ghost list
                    self._block_tenant.pop(bh, None)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=to_evict,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )

        # Allocate and insert into tenant's T1
        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store)

        for bh, block in zip(block_hashes_to_store, blocks):
            ts.t1[bh] = block
            self._block_tenant[bh] = tid
            # Remove from ghost lists
            ts.b1.pop(bh, None)
            ts.b2.pop(bh, None)

        store_spec = self.backend.get_load_store_spec(
            block_hashes_to_store, blocks
        )

        # Record metrics
        tier = self.backend.medium
        pagoda_offload_stores_total.labels(
            tenant_id=tid, tier=tier
        ).inc(len(block_hashes_to_store))
        pagoda_offload_blocks_used.labels(
            tenant_id=tid, tier=tier
        ).set(ts.num_cached_blocks)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def complete_store(
        self,
        block_hashes: Iterable[BlockHash],
        success: bool = True,
    ):
        stored_block_hashes: list[BlockHash] = []

        if success:
            for bh in block_hashes:
                block = self._find_block(bh)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(bh)
        else:
            for bh in block_hashes:
                ts = self._get_tenant_for_block(bh)
                block = ts.t1.pop(bh, None)
                if block is None:
                    block = ts.t2.pop(bh, None)
                if block is not None and not block.is_ready:
                    self.backend.free(block)
                    self._block_tenant.pop(bh, None)

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_block(self, block_hash: BlockHash) -> BlockStatus | None:
        """Find a block across all tenants."""
        tid = self._block_tenant.get(block_hash)
        if tid is None:
            return None
        ts = self.tenants.get(tid)
        if ts is None:
            return None
        return ts.t1.get(block_hash) or ts.t2.get(block_hash)

    def _evict_blocks(
        self,
        num_to_evict: int,
        protect_tenant: str | None = None,
    ) -> list[BlockHash]:
        """
        Evict blocks using tenant-aware priority ordering.

        Eviction order:
          1. Tenants over their max_blocks cap
          2. Lowest-priority tenants first
          3. Within a tenant: ARC-guided (T1 vs T2 based on target)
          4. Skip tenants at their min_blocks guarantee

        Args:
            num_to_evict: number of blocks to evict.
            protect_tenant: tenant_id to deprioritize for eviction
                (the tenant requesting the store).

        Returns:
            List of evicted block hashes.
        """
        evicted: list[BlockHash] = []

        # Phase 1: Evict from tenants over max_blocks
        if len(evicted) < num_to_evict:
            for ts in self._tenants_sorted_for_eviction(protect_tenant):
                if not ts.is_over_max:
                    continue
                while (
                    len(evicted) < num_to_evict and ts.is_over_max
                ):
                    bh = self._evict_one_from_tenant(ts)
                    if bh is None:
                        break
                    evicted.append(bh)

        # Phase 2: Evict from lowest-priority tenants, respecting min
        if len(evicted) < num_to_evict:
            for ts in self._tenants_sorted_for_eviction(protect_tenant):
                while len(evicted) < num_to_evict and not ts.is_at_min:
                    bh = self._evict_one_from_tenant(ts)
                    if bh is None:
                        break
                    evicted.append(bh)

        # Phase 3: If still not enough, evict even from min-guaranteed
        # (last resort — system is truly full)
        if len(evicted) < num_to_evict:
            for ts in self._tenants_sorted_for_eviction(protect_tenant):
                while len(evicted) < num_to_evict:
                    bh = self._evict_one_from_tenant(ts)
                    if bh is None:
                        break
                    evicted.append(bh)

        return evicted

    def _tenants_sorted_for_eviction(
        self, protect_tenant: str | None = None
    ) -> list[TenantState]:
        """
        Sort tenants for eviction: lowest priority first.
        The protected tenant (requesting the store) is placed last.
        """
        tenants = list(self.tenants.values())
        tenants.sort(key=lambda t: (
            # Protected tenant goes last
            1 if t.tenant_id == protect_tenant else 0,
            # Lower priority evicted first
            t.quota.priority,
            # Tenants with more blocks evicted first (within same priority)
            -t.num_cached_blocks,
        ))
        return tenants

    def _evict_one_from_tenant(
        self, ts: TenantState
    ) -> BlockHash | None:
        """
        Evict one block from a tenant using ARC policy.
        Returns the evicted block hash, or None if nothing evictable.
        """
        block_to_evict = None
        eviction_t = None
        eviction_b = None

        if len(ts.t1) >= int(ts.target_t1_size):
            # Try T1 first
            for bh, block in ts.t1.items():
                if block.ref_cnt == 0:
                    block_to_evict = (bh, block)
                    eviction_t = ts.t1
                    eviction_b = ts.b1
                    break

        if block_to_evict is None:
            # Try T2
            for bh, block in ts.t2.items():
                if block.ref_cnt == 0:
                    block_to_evict = (bh, block)
                    eviction_t = ts.t2
                    eviction_b = ts.b2
                    break

        if block_to_evict is None and eviction_t is None:
            # Try T1 as fallback (when target_t1_size was large)
            for bh, block in ts.t1.items():
                if block.ref_cnt == 0:
                    block_to_evict = (bh, block)
                    eviction_t = ts.t1
                    eviction_b = ts.b1
                    break

        if block_to_evict is None:
            return None

        bh, block = block_to_evict
        del eviction_t[bh]
        eviction_b[bh] = None
        self.backend.free(block)
        # NOTE: Do NOT remove from _block_tenant here - ghost blocks still
        # need tenant tracking for ARC adaptation when touched

        # Record eviction metrics
        tier = self.backend.medium
        pagoda_offload_evictions_total.labels(
            tenant_id=ts.tenant_id, tier=tier
        ).inc()
        pagoda_offload_blocks_used.labels(
            tenant_id=ts.tenant_id, tier=tier
        ).set(ts.num_cached_blocks)

        return bh

    # ------------------------------------------------------------------
    # Tenant stats (for monitoring / debugging)
    # ------------------------------------------------------------------

    def get_tenant_stats(self) -> dict[str, dict]:
        """Return per-tenant cache statistics."""
        stats = {}
        for tid, ts in self.tenants.items():
            stats[tid] = {
                "t1_size": len(ts.t1),
                "t2_size": len(ts.t2),
                "b1_size": len(ts.b1),
                "b2_size": len(ts.b2),
                "total_cached": ts.num_cached_blocks,
                "target_t1_size": ts.target_t1_size,
                "priority": ts.quota.priority,
                "min_blocks": ts.quota.min_blocks,
                "max_blocks": ts.quota.max_blocks,
            }
        return stats
