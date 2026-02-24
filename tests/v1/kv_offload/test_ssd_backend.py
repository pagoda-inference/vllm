# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SSDBackend."""
import tempfile
from pathlib import Path

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.backends.ssd import SSDBackend, SSDBlockStatus
from vllm.v1.kv_offload.mediums import SSDLoadStoreSpec


def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
    return [BlockHash(str(i).encode()) for i in int_hashes]


class TestSSDBackend:
    def test_init_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kv_cache"
            SSDBackend(block_size=256, num_blocks=4, ssd_path=path)
            assert path.exists()

    def test_free_blocks_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SSDBackend(block_size=256, num_blocks=4, ssd_path=tmpdir)
            assert backend.get_num_free_blocks() == 4

            backend.allocate_blocks(to_hashes([1, 2]))
            assert backend.get_num_free_blocks() == 2

    def test_allocate_and_free(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SSDBackend(block_size=256, num_blocks=4, ssd_path=tmpdir)
            blocks = backend.allocate_blocks(to_hashes([1, 2]))
            assert len(blocks) == 2
            assert all(isinstance(b, SSDBlockStatus) for b in blocks)
            assert blocks[0].block_id == 0
            assert blocks[1].block_id == 1

            backend.free(blocks[0])
            assert backend.get_num_free_blocks() == 3

    def test_reuse_freed_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SSDBackend(block_size=256, num_blocks=2, ssd_path=tmpdir)
            blocks = backend.allocate_blocks(to_hashes([1, 2]))
            backend.free(blocks[0])  # free block_id=0

            new_blocks = backend.allocate_blocks(to_hashes([3]))
            assert new_blocks[0].block_id == 0  # reused

    def test_get_load_store_spec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SSDBackend(block_size=256, num_blocks=4, ssd_path=tmpdir)
            blocks = backend.allocate_blocks(to_hashes([1, 2]))

            spec = backend.get_load_store_spec(to_hashes([1, 2]), blocks)
            assert isinstance(spec, SSDLoadStoreSpec)
            assert len(spec.file_paths) == 2
            assert "block_0.bin" in spec.file_paths[0]
            assert "block_1.bin" in spec.file_paths[1]

    def test_free_deletes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SSDBackend(block_size=256, num_blocks=4, ssd_path=tmpdir)
            blocks = backend.allocate_blocks(to_hashes([1]))

            # Create the file to simulate a completed store
            file_path = Path(tmpdir) / "block_0.bin"
            file_path.write_bytes(b"\x00" * 256)
            assert file_path.exists()

            backend.free(blocks[0])
            assert not file_path.exists()

    def test_medium(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SSDBackend(block_size=256, num_blocks=4, ssd_path=tmpdir)
            assert backend.medium == "SSD"
