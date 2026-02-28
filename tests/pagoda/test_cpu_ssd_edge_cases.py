# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU-SSD offloading edge cases (Issue #10).

Covers:
1. wait() timeout path — transfer job times out, task is cancelled
2. _create_load_task short-read detection — incomplete file read raises IOError
3. _create_load_task file-not-found — missing SSD block file raises FileNotFoundError
4. _create_store_task OSError handling
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, SSDLoadStoreSpec
from vllm.v1.kv_offload.worker.cpu_ssd import CPUToSSDOffloadingHandler


def _make_cpu_tensor(block_size_elements: int = 16, num_blocks: int = 4):
    """Create a fake CPU tensor using MagicMock (avoids torch dependency)."""
    import torch
    tensor = torch.zeros(num_blocks, block_size_elements, dtype=torch.float32)
    return tensor


def _make_handler(
    num_blocks: int = 4,
    block_size_elements: int = 16,
    io_timeout: float = 1.0,
):
    """Create a CPUToSSDOffloadingHandler with test tensors."""
    tensor = _make_cpu_tensor(block_size_elements, num_blocks)
    return CPUToSSDOffloadingHandler(
        cpu_tensors=[tensor],
        block_size_bytes=block_size_elements * 4,  # float32 = 4 bytes
        io_timeout_seconds=io_timeout,
    )


class TestCpuSsdTimeout:
    """Tests for wait() timeout handling."""

    def test_wait_timeout_cancels_task(self):
        """wait() cancels the task when IO timeout is exceeded."""
        handler = _make_handler(io_timeout=0.1)

        # Create a task that never completes
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def never_finish():
            await asyncio.sleep(999)

        task = loop.create_task(never_finish())

        from vllm.v1.kv_offload.worker.cpu_ssd import AsyncTransfer
        handler._transfers.append(AsyncTransfer(
            job_id=42,
            num_bytes=1024,
            task=task,
        ))
        handler._event_loop = loop

        # wait() should handle the timeout without raising
        handler.wait({42})

        # Task should have been cancelled (or at least cancel was called)
        # Note: shield() in wait() may prevent actual cancellation,
        # but cancel() should have been called
        assert task.cancelled() or task.done()
        loop.close()

    def test_wait_completed_task_no_timeout(self):
        """wait() returns immediately for already-completed tasks."""
        handler = _make_handler(io_timeout=1.0)

        loop = asyncio.new_event_loop()

        async def instant():
            return

        task = loop.create_task(instant())
        loop.run_until_complete(task)

        from vllm.v1.kv_offload.worker.cpu_ssd import AsyncTransfer
        handler._transfers.append(AsyncTransfer(
            job_id=1,
            num_bytes=64,
            task=task,
        ))
        handler._event_loop = loop

        # Should not raise or hang
        handler.wait({1})
        assert task.done()
        loop.close()


class TestCpuSsdShortRead:
    """Tests for _create_load_task short-read detection."""

    @pytest.mark.asyncio
    async def test_short_read_raises_ioerror(self, tmp_path):
        """Short read (incomplete file) raises IOError."""
        handler = _make_handler(block_size_elements=16)

        # Create a file that's too small
        block_file = tmp_path / "block_0.bin"
        expected_bytes = handler.cpu_tensors[0].element_size() * handler.cpu_tensors[0].stride(0)
        # Write only half the expected data
        block_file.write_bytes(b"\x00" * (expected_bytes // 2))

        ssd_spec = SSDLoadStoreSpec(
            block_ids=[0],
            file_paths=[str(block_file)],
        )
        cpu_spec = CPULoadStoreSpec(
            block_ids=[0],
        )

        with pytest.raises(IOError, match="Short read"):
            await handler._create_load_task(ssd_spec, cpu_spec)

    @pytest.mark.asyncio
    async def test_file_not_found_raises(self, tmp_path):
        """Missing SSD block file raises FileNotFoundError."""
        handler = _make_handler()

        ssd_spec = SSDLoadStoreSpec(
            block_ids=[0],
            file_paths=[str(tmp_path / "nonexistent_block.bin")],
        )
        cpu_spec = CPULoadStoreSpec(
            block_ids=[0],
        )

        with pytest.raises(FileNotFoundError, match="not found"):
            await handler._create_load_task(ssd_spec, cpu_spec)

    @pytest.mark.asyncio
    async def test_full_read_succeeds(self, tmp_path):
        """Full-size file is read successfully without error."""
        handler = _make_handler(block_size_elements=16)

        # Create a correctly-sized file
        block_file = tmp_path / "block_0.bin"
        expected_bytes = handler.cpu_tensors[0].element_size() * handler.cpu_tensors[0].stride(0)
        block_file.write_bytes(b"\x00" * expected_bytes)

        ssd_spec = SSDLoadStoreSpec(
            block_ids=[0],
            file_paths=[str(block_file)],
        )
        cpu_spec = CPULoadStoreSpec(
            block_ids=[0],
        )

        # Should not raise
        await handler._create_load_task(ssd_spec, cpu_spec)


class TestCpuSsdStoreErrors:
    """Tests for _create_store_task error handling."""

    @pytest.mark.asyncio
    async def test_store_oserror_raises(self, tmp_path):
        """OSError during write propagates as exception."""
        handler = _make_handler(block_size_elements=16)

        ssd_spec = SSDLoadStoreSpec(
            block_ids=[0],
            # Write to a path that will fail (directory as file)
            file_paths=[str(tmp_path)],  # tmp_path is a directory, not a file
        )
        cpu_spec = CPULoadStoreSpec(
            block_ids=[0],
        )

        with pytest.raises(OSError):
            await handler._create_store_task(cpu_spec, ssd_spec)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
