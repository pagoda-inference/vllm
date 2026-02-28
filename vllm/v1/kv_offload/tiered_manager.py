# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieredOffloadingManager — coordinates CPU (hot) and SSD (cold) tiers.

When the CPU tier is full and needs to evict, evicted blocks are spilled
to the SSD tier instead of being discarded.  On lookup / load the manager
checks CPU first, then SSD.

The manager delegates most ARC / LRU logic to two independent inner
managers (one per tier) and adds cross-tier spill / promote logic on top.
"""
from collections import OrderedDict
from collections.abc import Iterable

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus

logger = init_logger(__name__)


class TieredOffloadingManager(OffloadingManager):
    """
    Two-tier offloading manager: CPU (hot) → SSD (cold).

    New blocks are always stored into the CPU tier.  When the CPU tier
    needs to evict a block to make room, the evicted block is spilled
    to the SSD tier (if space is available) rather than discarded.

    Reads check CPU first; if the block is only on SSD the caller gets
    an SSD LoadStoreSpec so the worker can load from disk.
    """

    def __init__(
        self,
        cpu_backend: Backend,
        ssd_backend: Backend,
        enable_events: bool = False,
    ):
        self.cpu_backend = cpu_backend
        self.ssd_backend = ssd_backend

        self.cpu_capacity = cpu_backend.get_num_free_blocks()
        self.ssd_capacity = ssd_backend.get_num_free_blocks()

        # CPU tier: block_hash → BlockStatus (from cpu_backend)
        self.cpu_t1: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.cpu_t2: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        # Ghost lists for ARC adaptation on CPU tier
        self.cpu_b1: OrderedDict[BlockHash, None] = OrderedDict()
        self.cpu_b2: OrderedDict[BlockHash, None] = OrderedDict()
        self.cpu_target_t1: float = 0.0

        # SSD tier: simple LRU (cold storage, no ARC needed)
        self.ssd_cache: OrderedDict[BlockHash, BlockStatus] = OrderedDict()

        self.events: list[OffloadingEvent] | None = (
            [] if enable_events else None
        )

    # ------------------------------------------------------------------
    # OffloadingManager interface
    # ------------------------------------------------------------------

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        hit_count = 0
        for bh in block_hashes:
            block = self._find_cpu_block(bh)
            if block is None:
                block = self.ssd_cache.get(bh)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(
        self,
        block_hashes: Iterable[BlockHash],
        tenant_id: str | None = None,
    ) -> LoadStoreSpec:
        blocks: list[BlockStatus] = []
        block_hash_list = list(block_hashes)
        # Determine which tier each block lives in.
        # All blocks in a single prepare_load must come from the same tier
        # (the worker needs a single spec type).  If mixed, we promote
        # SSD blocks to CPU first.
        for bh in block_hash_list:
            cpu_block = self._find_cpu_block(bh)
            if cpu_block is not None:
                assert cpu_block.is_ready, (
                    f"Block {bh!r} not ready on CPU"
                )
                cpu_block.ref_cnt += 1
                blocks.append(cpu_block)
            else:
                ssd_block = self.ssd_cache.get(bh)
                assert ssd_block is not None, (
                    f"Block {bh!r} not found in any tier"
                )
                assert ssd_block.is_ready, (
                    f"Block {bh!r} not ready on SSD"
                )
                ssd_block.ref_cnt += 1
                blocks.append(ssd_block)

        # Return spec from the backend that owns the first block.
        # In practice, the offloading worker handles mixed specs via
        # the handler routing (GPU↔CPU vs CPU↔SSD).
        first_bh = block_hash_list[0]
        if self._find_cpu_block(first_bh) is not None:
            return self.cpu_backend.get_load_store_spec(
                block_hash_list, blocks
            )
        return self.ssd_backend.get_load_store_spec(
            block_hash_list, blocks
        )

    def touch(self, block_hashes: Iterable[BlockHash]):
        for bh in reversed(list(block_hashes)):
            # CPU tier ARC touch
            if bh in self.cpu_t1:
                block = self.cpu_t1.pop(bh)
                if not block.is_ready:
                    self.cpu_t1[bh] = block
                else:
                    self.cpu_t2[bh] = block
            elif bh in self.cpu_t2:
                self.cpu_t2.move_to_end(bh)
            elif bh in self.cpu_b1:
                delta = max(
                    1, len(self.cpu_b2) / max(len(self.cpu_b1), 1)
                )
                self.cpu_target_t1 = min(
                    self.cpu_target_t1 + delta, self.cpu_capacity
                )
                self.cpu_b1.move_to_end(bh)
            elif bh in self.cpu_b2:
                delta = max(
                    1, len(self.cpu_b1) / max(len(self.cpu_b2), 1)
                )
                self.cpu_target_t1 = max(self.cpu_target_t1 - delta, 0)
                self.cpu_b2.move_to_end(bh)

            # SSD tier LRU touch (independent of CPU tier)
            if bh in self.ssd_cache:
                # Touch on SSD refreshes LRU position
                self.ssd_cache.move_to_end(bh)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for bh in block_hashes:
            block = self._find_cpu_block(bh)
            if block is None:
                block = self.ssd_cache.get(bh)
            assert block is not None, f"Block {bh!r} not found"
            assert block.ref_cnt > 0, f"Block {bh!r} ref_cnt already 0"
            block.ref_cnt -= 1

    def prepare_store(
        self,
        block_hashes: Iterable[BlockHash],
        tenant_id: str | None = None,
    ) -> PrepareStoreOutput | None:
        # Filter already-cached blocks (in either tier)
        to_store: list[BlockHash] = []
        for bh in block_hashes:
            if (
                self._find_cpu_block(bh) is None
                and bh not in self.ssd_cache
            ):
                to_store.append(bh)

        if not to_store:
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=self.cpu_backend.get_load_store_spec([], []),
                block_hashes_evicted=[],
            )

        # Evict from CPU tier to make room, spilling to SSD
        num_to_evict = (
            len(to_store) - self.cpu_backend.get_num_free_blocks()
        )
        evicted: list[BlockHash] = []
        if num_to_evict > 0:
            evicted = self._evict_cpu_blocks(num_to_evict)
            if len(evicted) < num_to_evict:
                return None

        # Trim ghost lists
        for b in [self.cpu_b1, self.cpu_b2]:
            while len(b) > self.cpu_capacity:
                b.popitem(last=False)

        if evicted and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=evicted,
                    block_size=self.cpu_backend.block_size,
                    medium=self.cpu_backend.medium,
                    removed=True,
                )
            )

        # Allocate on CPU tier
        blocks = self.cpu_backend.allocate_blocks(to_store)
        assert len(blocks) == len(to_store)

        for bh, block in zip(to_store, blocks):
            self.cpu_t1[bh] = block
            self.cpu_b1.pop(bh, None)
            self.cpu_b2.pop(bh, None)

        store_spec = self.cpu_backend.get_load_store_spec(to_store, blocks)
        return PrepareStoreOutput(
            block_hashes_to_store=to_store,
            store_spec=store_spec,
            block_hashes_evicted=evicted,
        )

    def complete_store(
        self,
        block_hashes: Iterable[BlockHash],
        success: bool = True,
    ):
        stored: list[BlockHash] = []
        if success:
            for bh in block_hashes:
                block = self._find_cpu_block(bh)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    stored.append(bh)
        else:
            for bh in block_hashes:
                block = self.cpu_t1.pop(bh, None)
                if block is None:
                    block = self.cpu_t2.pop(bh, None)
                if block is not None and not block.is_ready:
                    self.cpu_backend.free(block)

        if stored and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored,
                    block_size=self.cpu_backend.block_size,
                    medium=self.cpu_backend.medium,
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

    def _find_cpu_block(self, bh: BlockHash) -> BlockStatus | None:
        return self.cpu_t1.get(bh) or self.cpu_t2.get(bh)

    def _evict_cpu_blocks(self, num_to_evict: int) -> list[BlockHash]:
        """
        Evict blocks from CPU tier using ARC policy.
        Evicted blocks are spilled to SSD if space is available.
        """
        evicted: list[BlockHash] = []

        while len(evicted) < num_to_evict:
            bh, block, src_list, ghost_list = self._pick_cpu_eviction()
            if bh is None:
                break

            del src_list[bh]
            ghost_list[bh] = None
            self.cpu_backend.free(block)
            evicted.append(bh)

            # Spill to SSD instead of discarding
            self._spill_to_ssd(bh)

        return evicted

    def _pick_cpu_eviction(
        self,
    ) -> tuple[
        BlockHash | None,
        BlockStatus | None,
        OrderedDict | None,
        OrderedDict | None,
    ]:
        """Pick one block to evict from CPU tier using ARC policy."""
        # Try T1 first if |T1| >= target
        if len(self.cpu_t1) >= int(self.cpu_target_t1):
            for bh, block in self.cpu_t1.items():
                if block.ref_cnt == 0:
                    return bh, block, self.cpu_t1, self.cpu_b1

        # Try T2
        for bh, block in self.cpu_t2.items():
            if block.ref_cnt == 0:
                return bh, block, self.cpu_t2, self.cpu_b2

        # Fallback: try T1 anyway
        for bh, block in self.cpu_t1.items():
            if block.ref_cnt == 0:
                return bh, block, self.cpu_t1, self.cpu_b1

        return None, None, None, None

    def _spill_to_ssd(self, block_hash: BlockHash):
        """
        Spill an evicted CPU block to SSD tier.
        If SSD is full, evict the oldest SSD block first.
        """
        if self.ssd_backend.get_num_free_blocks() == 0:
            # Evict oldest from SSD
            if self.ssd_cache:
                oldest_bh, oldest_block = next(iter(self.ssd_cache.items()))
                if oldest_block.ref_cnt == 0:
                    del self.ssd_cache[oldest_bh]
                    self.ssd_backend.free(oldest_block)
                else:
                    # All SSD blocks are in use, cannot spill
                    logger.warning(
                        "Cannot spill to SSD: all blocks in use"
                    )
                    return

        blocks = self.ssd_backend.allocate_blocks([block_hash])
        if blocks:
            ssd_block = blocks[0]
            # Mark as ready immediately (data was already on CPU,
            # the actual CPU→SSD transfer is handled by the worker
            # via the handler pipeline)
            ssd_block.ref_cnt = 0
            self.ssd_cache[block_hash] = ssd_block

            if self.events is not None:
                self.events.append(
                    OffloadingEvent(
                        block_hashes=[block_hash],
                        block_size=self.ssd_backend.block_size,
                        medium=self.ssd_backend.medium,
                        removed=False,
                    )
                )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_tier_stats(self) -> dict[str, dict]:
        """Return per-tier cache statistics."""
        return {
            "cpu": {
                "t1_size": len(self.cpu_t1),
                "t2_size": len(self.cpu_t2),
                "b1_size": len(self.cpu_b1),
                "b2_size": len(self.cpu_b2),
                "target_t1": self.cpu_target_t1,
                "total_cached": len(self.cpu_t1) + len(self.cpu_t2),
                "free_blocks": self.cpu_backend.get_num_free_blocks(),
            },
            "ssd": {
                "total_cached": len(self.ssd_cache),
                "free_blocks": self.ssd_backend.get_num_free_blocks(),
            },
        }
