# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TieredOffloadingManager (CPU hot → SSD cold)."""
import tempfile

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.backends.ssd import SSDBackend
from vllm.v1.kv_offload.tiered_manager import TieredOffloadingManager


def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
    return [BlockHash(str(i).encode()) for i in int_hashes]


def _make(
    cpu_blocks: int = 4,
    ssd_blocks: int = 4,
) -> TieredOffloadingManager:
    cpu = CPUBackend(block_size=256, num_blocks=cpu_blocks)
    with tempfile.TemporaryDirectory() as tmpdir:
        # We need the tmpdir to persist, so we use a class-level approach
        pass
    # For tests we create a fresh tmpdir that persists for the test
    import tempfile as _tf
    _tmpdir = _tf.mkdtemp()
    ssd = SSDBackend(block_size=256, num_blocks=ssd_blocks, ssd_path=_tmpdir)
    return TieredOffloadingManager(cpu, ssd, enable_events=True)


class TestTieredBasic:
    def test_store_and_lookup(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1, 2]))
        mgr.complete_store(to_hashes([1, 2]))

        assert mgr.lookup(to_hashes([1, 2])) == 2
        assert mgr.lookup(to_hashes([1, 2, 3])) == 2
        assert mgr.lookup(to_hashes([3])) == 0

    def test_store_not_ready_until_complete(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1]))
        assert mgr.lookup(to_hashes([1])) == 0

        mgr.complete_store(to_hashes([1]))
        assert mgr.lookup(to_hashes([1])) == 1

    def test_noop_for_existing(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1, 2]))
        mgr.complete_store(to_hashes([1, 2]))

        out = mgr.prepare_store(to_hashes([1, 2]))
        assert out is not None
        assert out.block_hashes_to_store == []

    def test_failed_store(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1, 2]))
        mgr.complete_store(to_hashes([1, 2]), success=False)

        assert mgr.lookup(to_hashes([1])) == 0


class TestTieredSpill:
    def test_eviction_spills_to_ssd(self):
        mgr = _make(cpu_blocks=2, ssd_blocks=4)
        mgr.prepare_store(to_hashes([1, 2]))
        mgr.complete_store(to_hashes([1, 2]))

        # Store 2 more → evicts [1, 2] from CPU → spills to SSD
        out = mgr.prepare_store(to_hashes([3, 4]))
        assert out is not None
        assert len(out.block_hashes_evicted) == 2
        mgr.complete_store(to_hashes([3, 4]))

        # Evicted blocks should be findable on SSD
        assert to_hashes([1])[0] in mgr.ssd_cache
        assert to_hashes([2])[0] in mgr.ssd_cache

        # Lookup should still find them (on SSD)
        assert mgr.lookup(to_hashes([1])) == 1
        assert mgr.lookup(to_hashes([3])) == 1

    def test_ssd_eviction_when_full(self):
        mgr = _make(cpu_blocks=2, ssd_blocks=2)
        # Fill CPU
        mgr.prepare_store(to_hashes([1, 2]))
        mgr.complete_store(to_hashes([1, 2]))

        # Evict to SSD (fills SSD)
        mgr.prepare_store(to_hashes([3, 4]))
        mgr.complete_store(to_hashes([3, 4]))
        assert len(mgr.ssd_cache) == 2

        # Evict again → SSD must evict oldest to make room
        mgr.prepare_store(to_hashes([5, 6]))
        mgr.complete_store(to_hashes([5, 6]))

        # SSD should still have at most 2 blocks
        assert len(mgr.ssd_cache) <= 2

    def test_lookup_checks_both_tiers(self):
        mgr = _make(cpu_blocks=2, ssd_blocks=4)
        mgr.prepare_store(to_hashes([1, 2]))
        mgr.complete_store(to_hashes([1, 2]))

        # Spill 1,2 to SSD
        mgr.prepare_store(to_hashes([3, 4]))
        mgr.complete_store(to_hashes([3, 4]))

        # 3,4 on CPU, 1,2 on SSD
        assert mgr.lookup(to_hashes([3])) == 1  # CPU hit
        assert mgr.lookup(to_hashes([1])) == 1  # SSD hit
        assert mgr.lookup(to_hashes([99])) == 0  # miss


class TestTieredLoad:
    def test_load_from_cpu(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1, 2]))
        mgr.complete_store(to_hashes([1, 2]))

        spec = mgr.prepare_load(to_hashes([1, 2]))
        assert spec is not None
        mgr.complete_load(to_hashes([1, 2]))

    def test_load_from_ssd(self):
        mgr = _make(cpu_blocks=2, ssd_blocks=4)
        mgr.prepare_store(to_hashes([1, 2]))
        mgr.complete_store(to_hashes([1, 2]))

        # Spill to SSD
        mgr.prepare_store(to_hashes([3, 4]))
        mgr.complete_store(to_hashes([3, 4]))

        # Load from SSD
        spec = mgr.prepare_load(to_hashes([1]))
        assert spec is not None
        mgr.complete_load(to_hashes([1]))

    def test_loaded_blocks_not_evictable(self):
        mgr = _make(cpu_blocks=4)
        mgr.prepare_store(to_hashes([1, 2, 3, 4]))
        mgr.complete_store(to_hashes([1, 2, 3, 4]))

        # Pin 1,2
        mgr.prepare_load(to_hashes([1, 2]))

        # Cannot evict enough
        out = mgr.prepare_store(to_hashes([5, 6, 7]))
        assert out is None

        mgr.complete_load(to_hashes([1, 2]))

        out = mgr.prepare_store(to_hashes([5, 6, 7]))
        assert out is not None


class TestTieredARC:
    def test_touch_promotes_t1_to_t2(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1]))
        mgr.complete_store(to_hashes([1]))

        assert to_hashes([1])[0] in mgr.cpu_t1
        mgr.touch(to_hashes([1]))
        assert to_hashes([1])[0] not in mgr.cpu_t1
        assert to_hashes([1])[0] in mgr.cpu_t2

    def test_touch_ssd_refreshes_lru(self):
        mgr = _make(cpu_blocks=2, ssd_blocks=4)
        mgr.prepare_store(to_hashes([1, 2]))
        mgr.complete_store(to_hashes([1, 2]))

        # Spill to SSD
        mgr.prepare_store(to_hashes([3, 4]))
        mgr.complete_store(to_hashes([3, 4]))

        # Touch SSD block
        mgr.touch(to_hashes([1]))
        # Block 1 should be at end of SSD LRU
        last_key = list(mgr.ssd_cache.keys())[-1]
        assert last_key == to_hashes([1])[0]

    def test_ghost_list_adaptation(self):
        mgr = _make(cpu_blocks=2)
        mgr.prepare_store(to_hashes([1, 2]))
        mgr.complete_store(to_hashes([1, 2]))

        initial_target = mgr.cpu_target_t1

        # Evict block 1 → B1
        mgr.prepare_store(to_hashes([3]))
        mgr.complete_store(to_hashes([3]))
        assert to_hashes([1])[0] in mgr.cpu_b1

        # Touch block 1 (in B1) → target increases
        mgr.touch(to_hashes([1]))
        assert mgr.cpu_target_t1 > initial_target


class TestTieredEvents:
    def test_store_event(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1]))
        mgr.complete_store(to_hashes([1]))

        events = list(mgr.take_events())
        stores = [e for e in events if not e.removed]
        assert len(stores) >= 1

    def test_eviction_event(self):
        mgr = _make(cpu_blocks=2, ssd_blocks=4)
        mgr.prepare_store(to_hashes([1, 2]))
        mgr.complete_store(to_hashes([1, 2]))
        list(mgr.take_events())  # drain

        mgr.prepare_store(to_hashes([3, 4]))
        mgr.complete_store(to_hashes([3, 4]))

        events = list(mgr.take_events())
        evictions = [e for e in events if e.removed]
        assert len(evictions) >= 1

    def test_take_events_clears(self):
        mgr = _make()
        mgr.prepare_store(to_hashes([1]))
        mgr.complete_store(to_hashes([1]))

        list(mgr.take_events())
        assert list(mgr.take_events()) == []


class TestTieredStats:
    def test_get_tier_stats(self):
        mgr = _make(cpu_blocks=2, ssd_blocks=4)
        mgr.prepare_store(to_hashes([1, 2]))
        mgr.complete_store(to_hashes([1, 2]))

        stats = mgr.get_tier_stats()
        assert stats["cpu"]["total_cached"] == 2
        assert stats["cpu"]["free_blocks"] == 0
        assert stats["ssd"]["total_cached"] == 0
        assert stats["ssd"]["free_blocks"] == 4

        # Spill to SSD
        mgr.prepare_store(to_hashes([3, 4]))
        mgr.complete_store(to_hashes([3, 4]))

        stats = mgr.get_tier_stats()
        assert stats["cpu"]["total_cached"] == 2
        assert stats["ssd"]["total_cached"] == 2
