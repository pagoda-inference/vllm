# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from collections.abc import Iterable
from pathlib import Path

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.mediums import SSDLoadStoreSpec

logger = init_logger(__name__)


class SSDBlockStatus(BlockStatus):
    _fields_ = BlockStatus._fields_ + [(  # type: ignore
        "block_id", ctypes.c_int64
    )]

    def __init__(self, block_id: int):
        super().__init__()
        self.block_id = block_id


class SSDBackend(Backend):
    """
    SSD backend for KV cache offloading.

    Stores KV cache blocks as individual files on SSD/NVMe storage.
    Uses simple file I/O for data transfer.

    File naming: {ssd_path}/block_{block_id}.bin
    """

    def __init__(
        self,
        block_size: int,
        num_blocks: int,
        ssd_path: str | Path,
    ):
        super().__init__(
            block_size=block_size,
            medium=SSDLoadStoreSpec.medium()
        )

        self.ssd_path = Path(ssd_path)
        self.ssd_path.mkdir(parents=True, exist_ok=True)

        self.num_blocks: int = num_blocks
        self.num_allocated_blocks: int = 0
        self.allocated_blocks_free_list: list[int] = []

        logger.info(
            "Initialized SSD backend at %s with %d blocks (block_size=%d)",
            self.ssd_path,
            num_blocks,
            block_size,
        )

    def get_num_free_blocks(self):
        return (
            len(self.allocated_blocks_free_list)
            + self.num_blocks
            - self.num_allocated_blocks
        )

    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        num_fresh_blocks = min(
            len(block_hashes), self.num_blocks - self.num_allocated_blocks
        )
        num_reused_blocks = len(block_hashes) - num_fresh_blocks
        assert len(self.allocated_blocks_free_list) >= num_reused_blocks

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh_blocks):
            blocks.append(SSDBlockStatus(self.num_allocated_blocks))
            self.num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused_blocks):
            block_id = self.allocated_blocks_free_list.pop()
            blocks.append(SSDBlockStatus(block_id))

        return blocks

    def free(self, block: BlockStatus):
        assert isinstance(block, SSDBlockStatus)
        # Delete the file if it exists
        file_path = self._get_block_path(block.block_id)
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(
                    "Failed to delete SSD block file %s: %s. "
                    "Block %d will NOT be reused to prevent corruption.",
                    file_path,
                    e,
                    block.block_id,
                )
                return
        self.allocated_blocks_free_list.append(block.block_id)

    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        file_paths = [
            str(self._get_block_path(block.block_id))
            for block in blocks
        ]
        return SSDLoadStoreSpec(
            block_ids=[block.block_id for block in blocks],
            file_paths=file_paths,
        )

    def _get_block_path(self, block_id: int) -> Path:
        return self.ssd_path / f"block_{block_id}.bin"
