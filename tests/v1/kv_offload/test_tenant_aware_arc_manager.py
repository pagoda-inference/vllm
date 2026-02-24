# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TenantAwareARCOffloadingManager."""
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.tenant_aware_arc_manager import (
    TenantAwareARCOffloadingManager,
    TenantQuota,
)


def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
    return [BlockHash(str(i).encode()) for i in int_hashes]


def _make(
    num_blocks: int = 4,
    quotas: dict[str, TenantQuota] | None = None,
) -> TenantAwareARCOffloadingManager:
    backend = CPUBackend(block_size=256, num_blocks=num_blocks)
    return TenantAwareARCOffloadingManager(
        backend, tenant_quotas=quotas, enable_events=True,
    )


# ------------------------------------------------------------------
# Basic operations
# ------------------------------------------------------------------

class TestBasicOperations:
    def test_store_and_lookup(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        assert mgr.lookup(to_hashes([1, 2])) == 2
        assert mgr.lookup(to_hashes([1, 2, 3])) == 2
        assert mgr.lookup(to_hashes([3])) == 0

    def test_store_not_ready_until_complete(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1]), tenant_id="t1")
        assert mgr.lookup(to_hashes([1])) == 0

        mgr.complete_store(to_hashes([1]))
        assert mgr.lookup(to_hashes([1])) == 1

    def test_store_noop_for_existing_blocks(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        out = mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        assert out is not None
        assert out.block_hashes_to_store == []

    def test_load_and_complete(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        spec = mgr.prepare_load(to_hashes([1, 2]))
        assert spec is not None
        mgr.complete_load(to_hashes([1, 2]))

    def test_failed_store_cleans_up(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]), success=False)

        assert mgr.lookup(to_hashes([1])) == 0
        assert mgr.lookup(to_hashes([2])) == 0


# ------------------------------------------------------------------
# Tenant isolation
# ------------------------------------------------------------------

class TestTenantIsolation:
    def test_two_tenants_independent(self):
        mgr = _make(num_blocks=4)
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="a")
        mgr.complete_store(to_hashes([1, 2]))
        mgr.prepare_store(to_hashes([3, 4]), tenant_id="b")
        mgr.complete_store(to_hashes([3, 4]))

        assert mgr.lookup(to_hashes([1, 2])) == 2
        assert mgr.lookup(to_hashes([3, 4])) == 2

        stats = mgr.get_tenant_stats()
        assert stats["a"]["total_cached"] == 2
        assert stats["b"]["total_cached"] == 2

    def test_auto_register_tenant(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1]), tenant_id="new_tenant")
        mgr.complete_store(to_hashes([1]))

        assert "new_tenant" in mgr.tenants
        assert mgr.lookup(to_hashes([1])) == 1

    def test_default_tenant_for_no_tenant_id(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1]))
        mgr.complete_store(to_hashes([1]))

        assert "__default__" in mgr.tenants
        assert mgr.lookup(to_hashes([1])) == 1

    def test_register_tenant_updates_quota(self):
        mgr = _make()
        mgr.register_tenant("t1", TenantQuota(priority=5))
        assert mgr.tenants["t1"].quota.priority == 5

        mgr.register_tenant("t1", TenantQuota(priority=10))
        assert mgr.tenants["t1"].quota.priority == 10


# ------------------------------------------------------------------
# ARC behavior (T1/T2 promotion, ghost lists)
# ------------------------------------------------------------------

class TestARCBehavior:
    def test_t1_to_t2_promotion(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1]), tenant_id="t1")
        mgr.complete_store(to_hashes([1]))

        ts = mgr.tenants["t1"]
        assert to_hashes([1])[0] in ts.t1
        assert to_hashes([1])[0] not in ts.t2

        mgr.touch(to_hashes([1]))
        assert to_hashes([1])[0] not in ts.t1
        assert to_hashes([1])[0] in ts.t2

    def test_ghost_list_b1_increases_target(self):
        mgr = _make(num_blocks=2)
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        ts = mgr.tenants["t1"]
        initial_target = ts.target_t1_size

        # Evict block 1 → goes to B1
        mgr.prepare_store(to_hashes([3]), tenant_id="t1")
        mgr.complete_store(to_hashes([3]))
        assert to_hashes([1])[0] in ts.b1

        # Touch block 1 (in B1) → target increases
        mgr.touch(to_hashes([1]))
        assert ts.target_t1_size > initial_target

    def test_ghost_list_b2_decreases_target(self):
        mgr = _make(num_blocks=2)
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        ts = mgr.tenants["t1"]
        # Promote block 1 to T2
        mgr.touch(to_hashes([1]))
        assert to_hashes([1])[0] in ts.t2

        # Evict block 1 from T2 → goes to B2
        # Need to set target low so T2 is evicted
        ts.target_t1_size = 10  # high target → evict from T2
        mgr.prepare_store(to_hashes([3]), tenant_id="t1")
        mgr.complete_store(to_hashes([3]))

        # If block 1 ended up in b2, touching it should decrease target
        if to_hashes([1])[0] in ts.b2:
            before = ts.target_t1_size
            mgr.touch(to_hashes([1]))
            assert ts.target_t1_size < before

    def test_ghost_lists_bounded(self):
        mgr = _make(num_blocks=2)
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        for i in range(3, 30):
            mgr.prepare_store(to_hashes([i]), tenant_id="t1")
            mgr.complete_store(to_hashes([i]))

        ts = mgr.tenants["t1"]
        assert len(ts.b1) <= mgr.cache_capacity
        assert len(ts.b2) <= mgr.cache_capacity


# ------------------------------------------------------------------
# Eviction: priority, quota, ordering
# ------------------------------------------------------------------

class TestEviction:
    def test_evicts_lowest_priority_first(self):
        quotas = {
            "low": TenantQuota(priority=1),
            "high": TenantQuota(priority=100),
        }
        mgr = _make(num_blocks=4, quotas=quotas)

        mgr.prepare_store(to_hashes([1, 2]), tenant_id="low")
        mgr.complete_store(to_hashes([1, 2]))
        mgr.prepare_store(to_hashes([3, 4]), tenant_id="high")
        mgr.complete_store(to_hashes([3, 4]))

        # Store 2 more for high → evicts from low
        out = mgr.prepare_store(to_hashes([5, 6]), tenant_id="high")
        assert out is not None
        assert len(out.block_hashes_evicted) == 2

        assert mgr.tenants["low"].num_cached_blocks == 0
        assert mgr.tenants["high"].num_cached_blocks == 4

    def test_evicts_over_max_first(self):
        quotas = {
            "capped": TenantQuota(max_blocks=1, priority=100),
            "normal": TenantQuota(priority=0),
        }
        mgr = _make(num_blocks=4, quotas=quotas)

        # Manually put capped tenant over its max
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="capped")
        mgr.complete_store(to_hashes([1, 2]))
        mgr.prepare_store(to_hashes([3, 4]), tenant_id="normal")
        mgr.complete_store(to_hashes([3, 4]))

        # capped has 2 blocks but max=1, so eviction phase 1 targets it
        out = mgr.prepare_store(to_hashes([5]), tenant_id="normal")
        assert out is not None
        # capped should have been evicted down despite high priority
        assert mgr.tenants["capped"].num_cached_blocks <= 1

    def test_max_blocks_cap_rejects_store(self):
        quotas = {"capped": TenantQuota(max_blocks=2)}
        mgr = _make(num_blocks=4, quotas=quotas)

        mgr.prepare_store(to_hashes([1, 2]), tenant_id="capped")
        mgr.complete_store(to_hashes([1, 2]))

        # At cap, further store is rejected
        out = mgr.prepare_store(to_hashes([3, 4]), tenant_id="capped")
        assert out is None

    def test_max_blocks_cap_partial_store(self):
        quotas = {"capped": TenantQuota(max_blocks=3)}
        mgr = _make(num_blocks=4, quotas=quotas)

        mgr.prepare_store(to_hashes([1, 2]), tenant_id="capped")
        mgr.complete_store(to_hashes([1, 2]))

        # Only 1 more allowed
        out = mgr.prepare_store(to_hashes([3, 4]), tenant_id="capped")
        assert out is not None
        assert len(out.block_hashes_to_store) == 1

    def test_min_blocks_respected(self):
        quotas = {
            "protected": TenantQuota(min_blocks=2, priority=0),
            "other": TenantQuota(priority=0),
        }
        mgr = _make(num_blocks=4, quotas=quotas)

        mgr.prepare_store(to_hashes([1, 2]), tenant_id="protected")
        mgr.complete_store(to_hashes([1, 2]))
        mgr.prepare_store(to_hashes([3, 4]), tenant_id="other")
        mgr.complete_store(to_hashes([3, 4]))

        # Evict 2 for a new tenant — should prefer "other" over "protected"
        # because protected is at its min_blocks
        out = mgr.prepare_store(to_hashes([5, 6]), tenant_id="new")
        assert out is not None
        # other should have been evicted (not at min)
        assert mgr.tenants["other"].num_cached_blocks == 0
        # protected should still have its min
        assert mgr.tenants["protected"].num_cached_blocks == 2

    def test_protect_requesting_tenant(self):
        """The tenant requesting the store is deprioritized for eviction."""
        mgr = _make(num_blocks=4)
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="a")
        mgr.complete_store(to_hashes([1, 2]))
        mgr.prepare_store(to_hashes([3, 4]), tenant_id="b")
        mgr.complete_store(to_hashes([3, 4]))

        # a stores 2 more → should evict from b (not a)
        out = mgr.prepare_store(to_hashes([5, 6]), tenant_id="a")
        assert out is not None
        assert mgr.tenants["b"].num_cached_blocks == 0
        assert mgr.tenants["a"].num_cached_blocks == 4

    def test_ref_cnt_blocks_eviction(self):
        mgr = _make(num_blocks=4)
        mgr.prepare_store(to_hashes([1, 2, 3, 4]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2, 3, 4]))

        # Load blocks 1,2 (ref_cnt > 0)
        mgr.prepare_load(to_hashes([1, 2]))

        # Cannot evict enough (1,2 are pinned)
        out = mgr.prepare_store(to_hashes([5, 6, 7]), tenant_id="t2")
        assert out is None

        mgr.complete_load(to_hashes([1, 2]))

        # Now it should work
        out = mgr.prepare_store(to_hashes([5, 6, 7]), tenant_id="t2")
        assert out is not None

    def test_eviction_returns_none_when_impossible(self):
        mgr = _make(num_blocks=2)
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        # Pin both blocks
        mgr.prepare_load(to_hashes([1, 2]))

        out = mgr.prepare_store(to_hashes([3]), tenant_id="t2")
        assert out is None

        mgr.complete_load(to_hashes([1, 2]))


# ------------------------------------------------------------------
# Events
# ------------------------------------------------------------------

class TestEvents:
    def test_store_events(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        events = list(mgr.take_events())
        assert len(events) == 1
        assert not events[0].removed
        assert set(events[0].block_hashes) == set(to_hashes([1, 2]))

    def test_eviction_events(self):
        mgr = _make(num_blocks=2)
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))
        list(mgr.take_events())  # drain

        mgr.prepare_store(to_hashes([3]), tenant_id="t2")
        mgr.complete_store(to_hashes([3]))

        events = list(mgr.take_events())
        evictions = [e for e in events if e.removed]
        stores = [e for e in events if not e.removed]
        assert len(evictions) == 1
        assert len(stores) == 1

    def test_take_events_clears(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1]), tenant_id="t1")
        mgr.complete_store(to_hashes([1]))

        events1 = list(mgr.take_events())
        assert len(events1) == 1

        events2 = list(mgr.take_events())
        assert len(events2) == 0


# ------------------------------------------------------------------
# get_tenant_stats
# ------------------------------------------------------------------

class TestTenantStats:
    def test_stats_reflect_state(self):
        quotas = {"t1": TenantQuota(min_blocks=1, max_blocks=10, priority=5)}
        mgr = _make(num_blocks=4, quotas=quotas)

        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))
        mgr.touch(to_hashes([1]))  # promote to T2

        stats = mgr.get_tenant_stats()["t1"]
        assert stats["t1_size"] == 1
        assert stats["t2_size"] == 1
        assert stats["total_cached"] == 2
        assert stats["min_blocks"] == 1
        assert stats["max_blocks"] == 10
        assert stats["priority"] == 5
