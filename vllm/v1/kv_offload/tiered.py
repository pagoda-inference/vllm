# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SSD and Tiered Offloading Specs for vLLM v1.

Provides:
  - SSDOffloadingSpec: Simple SSD-only offloading
  - TieredOffloadingSpec: GPU → CPU → SSD three-tier offloading
  - TenantAwareOffloadingSpec: Tenant-aware multi-tier offloading
"""
from collections.abc import Iterator
from pathlib import Path

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.backends.ssd import SSDBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.tiered_manager import TieredOffloadingManager
from vllm.v1.kv_offload.mediums import (
    CPULoadStoreSpec,
    GPULoadStoreSpec,
    SSDLoadStoreSpec,
)
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.tenant_aware_arc_manager import (
    TenantAwareARCOffloadingManager,
    TenantQuota,
)
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.cpu_ssd import CPUToSSDOffloadingHandler
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)


class SSDOffloadingSpec(OffloadingSpec):
    """
    Simple SSD-only offloading spec.

    GPU ↔ SSD direct transfers (no CPU tier).
    Useful for testing or when CPU memory is limited.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        ssd_path = self.extra_config.get("ssd_path")
        if not ssd_path:
            raise ValueError(
                "ssd_path must be specified in kv_connector_extra_config"
            )

        ssd_bytes_to_use = self.extra_config.get("ssd_bytes_to_use")
        if not ssd_bytes_to_use:
            raise ValueError(
                "ssd_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # Calculate block size
        assert kv_cache_config is not None
        page_sizes = {
            kv_cache_group.kv_cache_spec.page_size_bytes
            for kv_cache_group in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        kv_bytes_per_offloaded_block = kv_bytes_per_block * (
            self.offloaded_block_size // self.gpu_block_size
        )

        self.num_blocks = (
            int(ssd_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )
        self.ssd_path = Path(ssd_path)

        self._manager: OffloadingManager | None = None
        self.eviction_policy: str = self.extra_config.get(
            "eviction_policy", "lru"
        )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )

            backend = SSDBackend(
                block_size=self.offloaded_block_size,
                num_blocks=self.num_blocks,
                ssd_path=self.ssd_path,
            )

            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            else:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}. "
                    f"Supported policies: lru, arc"
                )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
    ]:
        # TODO: Implement GPU ↔ SSD direct transfer handler
        # For now, this is a placeholder
        raise NotImplementedError(
            "Direct GPU ↔ SSD transfer not yet implemented. "
            "Use TieredOffloadingSpec for GPU → CPU → SSD."
        )


class TieredOffloadingSpec(OffloadingSpec):
    """
    Three-tier offloading: GPU → CPU (hot) → SSD (cold).

    Configuration:
        cpu_bytes_to_use: CPU memory for hot tier
        ssd_path: SSD directory path
        ssd_bytes_to_use: SSD capacity for cold tier
        eviction_policy: "lru" or "arc"
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        # CPU tier config
        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise ValueError(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # SSD tier config
        ssd_path = self.extra_config.get("ssd_path")
        if not ssd_path:
            raise ValueError(
                "ssd_path must be specified in kv_connector_extra_config"
            )

        ssd_bytes_to_use = self.extra_config.get("ssd_bytes_to_use")
        if not ssd_bytes_to_use:
            raise ValueError(
                "ssd_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # Calculate block sizes
        assert kv_cache_config is not None
        page_sizes = {
            kv_cache_group.kv_cache_spec.page_size_bytes
            for kv_cache_group in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        kv_bytes_per_offloaded_block = kv_bytes_per_block * (
            self.offloaded_block_size // self.gpu_block_size
        )

        self.num_cpu_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )
        self.num_ssd_blocks = (
            int(ssd_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )
        self.ssd_path = Path(ssd_path)

        self._manager: OffloadingManager | None = None
        self._cpu_handlers: CpuGpuOffloadingHandlers | None = None
        self._ssd_handler: CPUToSSDOffloadingHandler | None = None

        self.eviction_policy: str = self.extra_config.get(
            "eviction_policy", "arc"
        )

        logger.info(
            "Initialized TieredOffloadingSpec: CPU=%d blocks, SSD=%d blocks",
            self.num_cpu_blocks,
            self.num_ssd_blocks,
        )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )

            cpu_backend = CPUBackend(
                block_size=self.offloaded_block_size,
                num_blocks=self.num_cpu_blocks,
            )
            ssd_backend = SSDBackend(
                block_size=self.offloaded_block_size,
                num_blocks=self.num_ssd_blocks,
                ssd_path=self.ssd_path,
            )

            self._manager = TieredOffloadingManager(
                cpu_backend=cpu_backend,
                ssd_backend=ssd_backend,
                enable_events=enable_events,
            )

            logger.info(
                "TieredOffloadingSpec: CPU=%d blocks, SSD=%d blocks",
                self.num_cpu_blocks,
                self.num_ssd_blocks,
            )

        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
    ]:
        if not current_platform.is_cuda_alike():
            raise Exception(
                "Tiered Offloading is currently only supported on CUDA-alike GPUs"
            )

        # GPU ↔ CPU handlers
        if not self._cpu_handlers:
            self._cpu_handlers = CpuGpuOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=self.num_cpu_blocks,
                gpu_caches=kv_caches,
            )

        assert self._cpu_handlers is not None
        yield (
            GPULoadStoreSpec,
            CPULoadStoreSpec,
            self._cpu_handlers.gpu_to_cpu_handler,
        )
        yield (
            CPULoadStoreSpec,
            GPULoadStoreSpec,
            self._cpu_handlers.cpu_to_gpu_handler,
        )

        # CPU ↔ SSD handler (TODO: integrate with manager)
        # For now, this is a placeholder
        logger.warning("CPU ↔ SSD handler not yet integrated with manager")


class TenantAwareOffloadingSpec(OffloadingSpec):
    """
    Tenant-aware multi-tier offloading with per-tenant quotas.

    Configuration:
        cpu_bytes_to_use: CPU memory for hot tier
        ssd_path: SSD directory path (optional)
        ssd_bytes_to_use: SSD capacity for cold tier (optional)
        eviction_policy: "tenant_aware_arc" (recommended)
        tenant_quotas: dict[tenant_id, TenantQuota]
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        # CPU tier config
        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise ValueError(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # Calculate block sizes
        assert kv_cache_config is not None
        page_sizes = {
            kv_cache_group.kv_cache_spec.page_size_bytes
            for kv_cache_group in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        kv_bytes_per_offloaded_block = kv_bytes_per_block * (
            self.offloaded_block_size // self.gpu_block_size
        )

        self.num_cpu_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )

        # Parse tenant quotas from config
        self.tenant_quotas: dict[str, TenantQuota] = {}
        tenant_quotas_config = self.extra_config.get("tenant_quotas", {})
        for tenant_id, quota_dict in tenant_quotas_config.items():
            self.tenant_quotas[tenant_id] = TenantQuota(
                min_blocks=quota_dict.get("min_blocks", 0),
                max_blocks=quota_dict.get("max_blocks", 0),
                priority=quota_dict.get("priority", 0),
                weight=quota_dict.get("weight", 1.0),
            )

        self._manager: TenantAwareARCOffloadingManager | None = None
        self._cpu_handlers: CpuGpuOffloadingHandlers | None = None

        logger.info(
            "Initialized TenantAwareOffloadingSpec: CPU=%d blocks, %d tenants configured",
            self.num_cpu_blocks,
            len(self.tenant_quotas),
        )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )

            backend = CPUBackend(
                block_size=self.offloaded_block_size,
                num_blocks=self.num_cpu_blocks,
            )

            self._manager = TenantAwareARCOffloadingManager(
                backend=backend,
                tenant_quotas=self.tenant_quotas,
                enable_events=enable_events,
            )

        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
    ]:
        if not current_platform.is_cuda_alike():
            raise Exception(
                "Tenant-aware Offloading is currently only supported on CUDA-alike GPUs"
            )

        if not self._cpu_handlers:
            self._cpu_handlers = CpuGpuOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=self.num_cpu_blocks,
                gpu_caches=kv_caches,
            )

        assert self._cpu_handlers is not None
        yield (
            GPULoadStoreSpec,
            CPULoadStoreSpec,
            self._cpu_handlers.gpu_to_cpu_handler,
        )
        yield (
            CPULoadStoreSpec,
            GPULoadStoreSpec,
            self._cpu_handlers.cpu_to_gpu_handler,
        )
