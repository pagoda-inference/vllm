# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SSD free() deletion failure handling (Issue #9).

Verifies that when file deletion fails during free():
1. The block is NOT returned to the free list (prevents corruption)
2. Successful deletion does return the block to the free list
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.backends.ssd import SSDBackend, SSDBlockStatus


def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
    return [BlockHash(str(i).encode()) for i in int_hashes]


class TestSsdFreeFailure:
    """Tests for SSD free() when file deletion fails."""

    def test_free_delete_failure_does_not_recycle_block(self):
        """Block is NOT added to free list when unlink() fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SSDBackend(
                block_size=256, num_blocks=4, ssd_path=tmpdir,
            )
            blocks = backend.allocate_blocks(to_hashes([1]))
            block = blocks[0]

            # Create the file so unlink is attempted
            file_path = Path(tmpdir) / f"block_{block.block_id}.bin"
            file_path.write_bytes(b"\x00" * 256)

            # Make unlink fail
            with patch.object(Path, "unlink", side_effect=PermissionError("denied")):
                backend.free(block)

            # Block should NOT be in the free list
            assert block.block_id not in backend.allocated_blocks_free_list

    def test_free_delete_success_recycles_block(self):
        """Block IS added to free list when unlink() succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SSDBackend(
                block_size=256, num_blocks=4, ssd_path=tmpdir,
            )
            blocks = backend.allocate_blocks(to_hashes([1]))
            block = blocks[0]

            # Create the file
            file_path = Path(tmpdir) / f"block_{block.block_id}.bin"
            file_path.write_bytes(b"\x00" * 256)

            backend.free(block)

            # Block should be in the free list
            assert block.block_id in backend.allocated_blocks_free_list
            # File should be deleted
            assert not file_path.exists()

    def test_free_nonexistent_file_still_recycles(self):
        """Block is recycled when file doesn't exist (already cleaned up)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SSDBackend(
                block_size=256, num_blocks=4, ssd_path=tmpdir,
            )
            blocks = backend.allocate_blocks(to_hashes([1]))
            block = blocks[0]

            # Don't create the file — simulate a block that was never stored
            backend.free(block)

            # Should still be recycled (no file to delete = no error)
            assert block.block_id in backend.allocated_blocks_free_list

    def test_free_failure_preserves_other_blocks(self):
        """Failed free of one block doesn't affect other blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SSDBackend(
                block_size=256, num_blocks=4, ssd_path=tmpdir,
            )
            blocks = backend.allocate_blocks(to_hashes([1, 2]))

            # Create files for both blocks
            for b in blocks:
                fp = Path(tmpdir) / f"block_{b.block_id}.bin"
                fp.write_bytes(b"\x00" * 256)

            # Fail on first block
            with patch.object(Path, "unlink", side_effect=PermissionError("denied")):
                backend.free(blocks[0])

            # Second block should free normally
            backend.free(blocks[1])

            assert blocks[0].block_id not in backend.allocated_blocks_free_list
            assert blocks[1].block_id in backend.allocated_blocks_free_list

    def test_free_count_after_mixed_success_failure(self):
        """get_num_free_blocks reflects correct count after mixed free results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SSDBackend(
                block_size=256, num_blocks=4, ssd_path=tmpdir,
            )
            blocks = backend.allocate_blocks(to_hashes([1, 2, 3]))
            assert backend.get_num_free_blocks() == 1  # 4 - 3

            # Create files
            for b in blocks:
                fp = Path(tmpdir) / f"block_{b.block_id}.bin"
                fp.write_bytes(b"\x00" * 256)

            # Free block 0: fail
            with patch.object(Path, "unlink", side_effect=OSError("disk error")):
                backend.free(blocks[0])

            # Free block 1: success
            backend.free(blocks[1])

            # 1 original free + 1 successfully freed = 2
            # (block 0 failed, so not recycled)
            assert backend.get_num_free_blocks() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
