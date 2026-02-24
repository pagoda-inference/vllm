# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Prometheus metrics instrumentation in TenantAwareARCOffloadingManager."""
import pytest

from vllm.pagoda.metrics import (
    pagoda_offload_blocks_used,
    pagoda_offload_evictions_total,
    pagoda_offload_hits_total,
    pagoda_offload_misses_total,
    pagoda_offload_stores_total,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.tenant_aware_arc_manager import (
    TenantAwareARCOffloadingManager,
    TenantQuota,
)


def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
    return [BlockHash(str(i).encode()) for i in int_hashes]


def _counter_value(counter, **labels) -> float:
    """Read the current value of a labeled Counter."""
    return counter.labels(**labels)._value.get()


def _gauge_value(gauge, **labels) -> float:
    """Read the current value of a labeled Gauge."""
    return gauge.labels(**labels)._value.get()


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Clear all label-children so each test starts from zero."""
    for metric in (
        pagoda_offload_hits_total,
        pagoda_offload_misses_total,
        pagoda_offload_stores_total,
        pagoda_offload_evictions_total,
        pagoda_offload_blocks_used,
    ):
        metric._metrics.clear()
    yield


def _make_manager(
    num_blocks: int = 4,
    tenant_quotas: dict[str, TenantQuota] | None = None,
) -> TenantAwareARCOffloadingManager:
    backend = CPUBackend(block_size=256, num_blocks=num_blocks)
    return TenantAwareARCOffloadingManager(
        backend, tenant_quotas=tenant_quotas, enable_events=False,
    )


# ------------------------------------------------------------------
# lookup metrics
# ------------------------------------------------------------------

class TestLookupMetrics:
    def test_all_hits(self):
        mgr = _make_manager()
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        hit = mgr.lookup(to_hashes([1, 2]))
        assert hit == 2
        assert _counter_value(pagoda_offload_hits_total, tenant_id="t1") == 2
        assert _counter_value(pagoda_offload_misses_total, tenant_id="t1") == 0

    def test_partial_hit(self):
        mgr = _make_manager()
        mgr.prepare_store(to_hashes([1]), tenant_id="t1")
        mgr.complete_store(to_hashes([1]))

        hit = mgr.lookup(to_hashes([1, 2]))
        assert hit == 1
        assert _counter_value(pagoda_offload_hits_total, tenant_id="t1") == 1
        assert _counter_value(pagoda_offload_misses_total, tenant_id="t1") == 1

    def test_all_miss(self):
        mgr = _make_manager()
        mgr.prepare_store(to_hashes([1]), tenant_id="t1")
        mgr.complete_store(to_hashes([1]))

        hit = mgr.lookup(to_hashes([99, 100]))
        assert hit == 0
        # block 99 is unknown, so tenant falls back to __default__
        assert _counter_value(
            pagoda_offload_misses_total, tenant_id="__default__"
        ) == 2

    def test_empty_lookup(self):
        mgr = _make_manager()
        hit = mgr.lookup(to_hashes([]))
        assert hit == 0
        # No labels should have been created
        assert len(pagoda_offload_hits_total._metrics) == 0
        assert len(pagoda_offload_misses_total._metrics) == 0

    def test_hits_accumulate(self):
        mgr = _make_manager()
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        mgr.lookup(to_hashes([1]))
        mgr.lookup(to_hashes([1, 2]))
        assert _counter_value(pagoda_offload_hits_total, tenant_id="t1") == 3


# ------------------------------------------------------------------
# store metrics
# ------------------------------------------------------------------

class TestStoreMetrics:
    def test_store_increments_counter_and_gauge(self):
        mgr = _make_manager()
        mgr.prepare_store(to_hashes([1, 2, 3]), tenant_id="t1")

        assert _counter_value(
            pagoda_offload_stores_total, tenant_id="t1", tier="CPU"
        ) == 3
        assert _gauge_value(
            pagoda_offload_blocks_used, tenant_id="t1", tier="CPU"
        ) == 3

    def test_store_noop_for_already_cached(self):
        mgr = _make_manager()
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        # Store same blocks again — should be a no-op
        out = mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        assert out is not None
        assert out.block_hashes_to_store == []
        # Counter should still be 2 from the first store
        assert _counter_value(
            pagoda_offload_stores_total, tenant_id="t1", tier="CPU"
        ) == 2

    def test_store_two_tenants(self):
        mgr = _make_manager(num_blocks=4)
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="a")
        mgr.prepare_store(to_hashes([3]), tenant_id="b")

        assert _counter_value(
            pagoda_offload_stores_total, tenant_id="a", tier="CPU"
        ) == 2
        assert _counter_value(
            pagoda_offload_stores_total, tenant_id="b", tier="CPU"
        ) == 1
        assert _gauge_value(
            pagoda_offload_blocks_used, tenant_id="a", tier="CPU"
        ) == 2
        assert _gauge_value(
            pagoda_offload_blocks_used, tenant_id="b", tier="CPU"
        ) == 1


# ------------------------------------------------------------------
# eviction metrics
# ------------------------------------------------------------------

class TestEvictionMetrics:
    def test_eviction_counted_per_tenant(self):
        mgr = _make_manager(num_blocks=2)
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        # Store block 3 for t2 — must evict 1 block from t1
        mgr.prepare_store(to_hashes([3]), tenant_id="t2")

        assert _counter_value(
            pagoda_offload_evictions_total, tenant_id="t1", tier="CPU"
        ) == 1
        # t1 gauge should drop to 1
        assert _gauge_value(
            pagoda_offload_blocks_used, tenant_id="t1", tier="CPU"
        ) == 1

    def test_eviction_respects_priority(self):
        """Lower-priority tenant is evicted first."""
        quotas = {
            "low": TenantQuota(priority=0),
            "high": TenantQuota(priority=10),
        }
        mgr = _make_manager(num_blocks=4, tenant_quotas=quotas)

        mgr.prepare_store(to_hashes([1, 2]), tenant_id="low")
        mgr.complete_store(to_hashes([1, 2]))
        mgr.prepare_store(to_hashes([3, 4]), tenant_id="high")
        mgr.complete_store(to_hashes([3, 4]))

        # Store 2 more for high — must evict 2 from low
        mgr.prepare_store(to_hashes([5, 6]), tenant_id="high")

        assert _counter_value(
            pagoda_offload_evictions_total, tenant_id="low", tier="CPU"
        ) == 2
        assert _counter_value(
            pagoda_offload_evictions_total, tenant_id="high", tier="CPU"
        ) == 0

    def test_eviction_updates_gauge_to_zero(self):
        mgr = _make_manager(num_blocks=2)
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="t1")
        mgr.complete_store(to_hashes([1, 2]))

        # Evict both blocks from t1
        mgr.prepare_store(to_hashes([3, 4]), tenant_id="t2")

        assert _gauge_value(
            pagoda_offload_blocks_used, tenant_id="t1", tier="CPU"
        ) == 0
        assert _gauge_value(
            pagoda_offload_blocks_used, tenant_id="t2", tier="CPU"
        ) == 2

    def test_max_blocks_cap_triggers_eviction_metrics(self):
        """When a tenant hits max_blocks, further stores for that tenant
        are capped — but evictions from *other* tenants still get counted."""
        quotas = {
            "capped": TenantQuota(max_blocks=2),
            "other": TenantQuota(),
        }
        mgr = _make_manager(num_blocks=4, tenant_quotas=quotas)

        mgr.prepare_store(to_hashes([1, 2]), tenant_id="capped")
        mgr.complete_store(to_hashes([1, 2]))
        mgr.prepare_store(to_hashes([3, 4]), tenant_id="other")
        mgr.complete_store(to_hashes([3, 4]))

        # capped tenant tries to store 2 more — should be rejected (at cap)
        out = mgr.prepare_store(to_hashes([5, 6]), tenant_id="capped")
        assert out is None

        # No eviction should have happened
        assert _counter_value(
            pagoda_offload_evictions_total, tenant_id="capped", tier="CPU"
        ) == 0
        assert _counter_value(
            pagoda_offload_evictions_total, tenant_id="other", tier="CPU"
        ) == 0


# ------------------------------------------------------------------
# multi-tenant end-to-end
# ------------------------------------------------------------------

class TestMultiTenantMetricsE2E:
    def test_full_lifecycle(self):
        """Store, lookup, evict across two tenants and verify all counters."""
        mgr = _make_manager(num_blocks=4)

        # Tenant A stores 2 blocks
        mgr.prepare_store(to_hashes([1, 2]), tenant_id="A")
        mgr.complete_store(to_hashes([1, 2]))

        # Tenant B stores 2 blocks
        mgr.prepare_store(to_hashes([3, 4]), tenant_id="B")
        mgr.complete_store(to_hashes([3, 4]))

        # Lookups
        assert mgr.lookup(to_hashes([1, 2])) == 2
        assert mgr.lookup(to_hashes([3, 4])) == 2
        assert mgr.lookup(to_hashes([1, 99])) == 1

        assert _counter_value(pagoda_offload_hits_total, tenant_id="A") == 3
        assert _counter_value(pagoda_offload_misses_total, tenant_id="A") == 1
        assert _counter_value(pagoda_offload_hits_total, tenant_id="B") == 2
        assert _counter_value(pagoda_offload_misses_total, tenant_id="B") == 0

        # Tenant A stores 2 more — evicts 2 from B (lower priority by default,
        # but same priority means more-blocks-first, and B is not protected)
        mgr.prepare_store(to_hashes([5, 6]), tenant_id="A")
        mgr.complete_store(to_hashes([5, 6]))

        assert _counter_value(
            pagoda_offload_stores_total, tenant_id="A", tier="CPU"
        ) == 4
        assert _gauge_value(
            pagoda_offload_blocks_used, tenant_id="A", tier="CPU"
        ) == 4
        # B had 2 evictions
        assert _counter_value(
            pagoda_offload_evictions_total, tenant_id="B", tier="CPU"
        ) == 2
        assert _gauge_value(
            pagoda_offload_blocks_used, tenant_id="B", tier="CPU"
        ) == 0
