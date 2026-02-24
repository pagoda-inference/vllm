# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pagoda configuration management with YAML loading."""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from vllm.logger import init_logger
from vllm.pagoda.mass_client import MassApiClient, TenantConfig

logger = init_logger(__name__)


@dataclass
class QueueConfig:
    """Global queue depth configuration."""
    max_pending_requests: int = 1000
    warn_threshold_pct: float = 0.8


class PagodaConfig:
    """Load pagoda_config.yaml and initialize MassApiClient."""

    def __init__(self, config_path: str) -> None:
        self._config_path = config_path

        try:
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.error("Pagoda config file not found: %s", config_path)
            raw = {}
        except yaml.YAMLError as e:
            logger.error("Failed to parse pagoda config: %s", e)
            raw = {}

        # Parse queue config
        queue_raw = raw.get("queue", {})
        self.queue = QueueConfig(
            max_pending_requests=int(
                queue_raw.get("max_pending_requests", 1000)
            ),
            warn_threshold_pct=float(
                queue_raw.get("warn_threshold_pct", 0.8)
            ),
        )

        # Parse defaults for fallback
        defaults_raw = raw.get("defaults", {})
        defaults = TenantConfig(
            tenant_id="__default__",
            priority=defaults_raw.get("priority", "normal"),
            qps_limit=float(defaults_raw.get("qps_limit", 10.0)),
            concurrent_limit=int(defaults_raw.get("concurrent_limit", 5)),
            kv_cache_block_quota=int(
                defaults_raw.get("kv_cache_block_quota", 100)
            ),
        )

        # Initialize MASS API client
        mass_cfg = raw.get("mass_api", {})
        self.mass_client = MassApiClient(
            mass_api_url=mass_cfg.get("url", "http://localhost:8080/api"),
            timeout_seconds=mass_cfg.get("timeout_seconds", 2),
            ttl_seconds=mass_cfg.get("tenant_config_ttl_seconds", 300),
            stale_ttl_seconds=mass_cfg.get("stale_cache_ttl_seconds", 3600),
            defaults=defaults,
        )

        logger.info("Loaded pagoda config from %s", config_path)

    def get_queue_config(self) -> QueueConfig:
        """Get queue depth config."""
        return self.queue
