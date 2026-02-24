# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pagoda configuration management with YAML loading and hot reload."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from threading import Lock

import yaml

from vllm.logger import init_logger

logger = init_logger(__name__)

_RELOAD_INTERVAL_SEC = 60.0


@dataclass
class TenantConfig:
    """Per-tenant configuration."""
    priority: str = "normal"
    qps_limit: float = 10.0
    concurrent_limit: int = 5
    prompt_template: str | None = None


@dataclass
class QueueConfig:
    """Global queue depth configuration."""
    max_pending_requests: int = 1000
    warn_threshold_pct: float = 0.8


@dataclass
class PagodaConfigData:
    """Parsed configuration data."""
    api_keys: dict[str, str] = field(default_factory=dict)
    tenants: dict[str, TenantConfig] = field(default_factory=dict)
    defaults: TenantConfig = field(default_factory=TenantConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)


class PagodaConfig:
    """Load and manage pagoda_config.yaml with file-watch hot reload."""

    def __init__(self, config_path: str) -> None:
        self._config_path = config_path
        self._lock = Lock()
        self._last_mtime: float = 0.0
        self._last_check_time: float = 0.0
        self._data = PagodaConfigData()
        self._load()

    def _load(self) -> None:
        """Load config from YAML file."""
        try:
            with open(self._config_path) as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.error("Pagoda config file not found: %s", self._config_path)
            return
        except yaml.YAMLError as e:
            logger.error("Failed to parse pagoda config: %s", e)
            return

        self._last_mtime = os.path.getmtime(self._config_path)
        self._last_check_time = time.monotonic()

        # Parse api_keys
        api_keys = raw.get("api_keys", {})

        # Parse defaults
        defaults_raw = raw.get("defaults", {})
        defaults = TenantConfig(
            priority=defaults_raw.get("priority", "normal"),
            qps_limit=float(defaults_raw.get("qps_limit", 10.0)),
            concurrent_limit=int(defaults_raw.get("concurrent_limit", 5)),
            prompt_template=defaults_raw.get("prompt_template"),
        )

        # Parse tenants
        tenants: dict[str, TenantConfig] = {}
        for tid, tcfg in raw.get("tenants", {}).items():
            tenants[tid] = TenantConfig(
                priority=tcfg.get("priority", defaults.priority),
                qps_limit=float(tcfg.get("qps_limit", defaults.qps_limit)),
                concurrent_limit=int(
                    tcfg.get("concurrent_limit", defaults.concurrent_limit)
                ),
                prompt_template=tcfg.get("prompt_template"),
            )

        # Parse queue
        queue_raw = raw.get("queue", {})
        queue = QueueConfig(
            max_pending_requests=int(
                queue_raw.get("max_pending_requests", 1000)
            ),
            warn_threshold_pct=float(
                queue_raw.get("warn_threshold_pct", 0.8)
            ),
        )

        with self._lock:
            self._data = PagodaConfigData(
                api_keys=api_keys,
                tenants=tenants,
                defaults=defaults,
                queue=queue,
            )

        logger.info(
            "Loaded pagoda config: %d api_keys, %d tenants",
            len(api_keys),
            len(tenants),
        )

    def _maybe_reload(self) -> None:
        """Check file mtime and reload if changed (throttled)."""
        now = time.monotonic()
        if now - self._last_check_time < _RELOAD_INTERVAL_SEC:
            return
        self._last_check_time = now

        try:
            mtime = os.path.getmtime(self._config_path)
        except OSError:
            return

        if mtime > self._last_mtime:
            logger.info("Pagoda config file changed, reloading...")
            self._load()

    def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        """Get config for a specific tenant, falling back to defaults."""
        self._maybe_reload()
        with self._lock:
            return self._data.tenants.get(tenant_id, self._data.defaults)

    def get_defaults(self) -> TenantConfig:
        """Get default tenant config."""
        self._maybe_reload()
        with self._lock:
            return self._data.defaults

    def get_queue_config(self) -> QueueConfig:
        """Get queue depth config."""
        self._maybe_reload()
        with self._lock:
            return self._data.queue

    def resolve_api_key(self, api_key: str) -> str | None:
        """Map API key to tenant_id. Returns None if not found."""
        self._maybe_reload()
        with self._lock:
            return self._data.api_keys.get(api_key)

    @property
    def data(self) -> PagodaConfigData:
        """Access current config data (triggers reload check)."""
        self._maybe_reload()
        with self._lock:
            return self._data

    def reload(self) -> None:
        """Force reload config from file."""
        self._load()
