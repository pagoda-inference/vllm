# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import aiofiles
import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, SSDLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


@dataclass
class AsyncTransfer:
    job_id: int
    num_bytes: int
    task: asyncio.Task


class CPUToSSDOffloadingHandler(OffloadingHandler):
    """
    Handler for CPU ↔ SSD transfers using async file I/O.

    This handler manages data transfer between CPU pinned memory
    and SSD storage using aiofiles for async I/O operations.
    """

    def __init__(
        self,
        cpu_tensors: list[torch.Tensor],
        block_size_bytes: int,
    ):
        """
        Initialize CPU to SSD offloading handler.

        Args:
            cpu_tensors: List of CPU tensors (pinned memory) to transfer.
            block_size_bytes: Size of each block in bytes.
        """
        self.cpu_tensors = cpu_tensors
        self.block_size_bytes = block_size_bytes
        self.total_block_size_bytes = sum(
            tensor.element_size() * tensor.stride(0)
            for tensor in cpu_tensors
        )

        # Track ongoing transfers
        self._transfers: deque[AsyncTransfer] = deque()
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for async operations."""
        if self._event_loop is None:
            try:
                self._event_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._event_loop)
        return self._event_loop

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        """
        Start an async transfer from CPU to SSD or SSD to CPU.

        Args:
            job_id: Unique identifier for this transfer job.
            transfer_spec: Tuple of (src_spec, dst_spec).

        Returns:
            True if transfer was successfully started.
        """
        src_spec, dst_spec = transfer_spec

        # Determine transfer direction
        if isinstance(src_spec, CPULoadStoreSpec) and isinstance(dst_spec, SSDLoadStoreSpec):
            # CPU → SSD (store)
            task = self._create_store_task(src_spec, dst_spec)
        elif isinstance(src_spec, SSDLoadStoreSpec) and isinstance(dst_spec, CPULoadStoreSpec):
            # SSD → CPU (load)
            task = self._create_load_task(src_spec, dst_spec)
        else:
            logger.error(
                "Invalid transfer spec types: %s → %s",
                type(src_spec).__name__,
                type(dst_spec).__name__,
            )
            return False

        # Schedule the task
        loop = self._get_event_loop()
        async_task = loop.create_task(task)

        num_blocks = len(src_spec.block_ids)
        num_bytes = num_blocks * self.total_block_size_bytes

        self._transfers.append(
            AsyncTransfer(
                job_id=job_id,
                num_bytes=num_bytes,
                task=async_task,
            )
        )

        return True

    async def _create_store_task(
        self,
        cpu_spec: CPULoadStoreSpec,
        ssd_spec: SSDLoadStoreSpec,
    ):
        """Store blocks from CPU to SSD."""
        for cpu_block_id, file_path in zip(cpu_spec.block_ids, ssd_spec.file_paths):
            # Read data from CPU tensor
            data_chunks = []
            for tensor in self.cpu_tensors:
                block_data = tensor[cpu_block_id]
                data_chunks.append(block_data.cpu().numpy().tobytes())

            # Write to SSD file
            async with aiofiles.open(file_path, 'wb') as f:
                for chunk in data_chunks:
                    await f.write(chunk)

    async def _create_load_task(
        self,
        ssd_spec: SSDLoadStoreSpec,
        cpu_spec: CPULoadStoreSpec,
    ):
        """Load blocks from SSD to CPU."""
        for file_path, cpu_block_id in zip(ssd_spec.file_paths, cpu_spec.block_ids):
            # Check if file exists
            if not Path(file_path).exists():
                logger.error("SSD block file not found: %s", file_path)
                raise FileNotFoundError(f"SSD block file not found: {file_path}")

            # Read from SSD file
            async with aiofiles.open(file_path, 'rb') as f:
                for tensor in self.cpu_tensors:
                    chunk_size = tensor.element_size() * tensor.stride(0)
                    data = await f.read(chunk_size)
                    # Copy data to CPU tensor
                    tensor[cpu_block_id].copy_(
                        torch.frombuffer(data, dtype=tensor.dtype).reshape(
                            tensor[cpu_block_id].shape
                        )
                    )

    def get_finished(self) -> list[TransferResult]:
        """Check for completed transfers and return their results."""
        results: list[TransferResult] = []

        while self._transfers and self._transfers[0].task.done():
            transfer = self._transfers.popleft()

            try:
                # Check if task completed successfully
                transfer.task.result()
                success = True
            except Exception as e:
                logger.error(
                    "Transfer job %d failed: %s",
                    transfer.job_id,
                    e
                )
                success = False

            result = TransferResult(
                job_id=transfer.job_id,
                success=success,
                transfer_size=transfer.num_bytes,
                transfer_time=0.0,  # TODO: Add timing if needed
                transfer_type=("CPU", "SSD"),
            )
            results.append(result)

        return results

    def wait(self, job_ids: set[int]):
        """Wait for specific transfer jobs to complete."""
        loop = self._get_event_loop()

        for transfer in self._transfers:
            if transfer.job_id in job_ids:
                if not transfer.task.done():
                    loop.run_until_complete(transfer.task)
