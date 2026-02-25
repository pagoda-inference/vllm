# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda configuration loading."""

import os

import pytest
import yaml

from vllm.pagoda.config import PagodaConfig, QueueConfig, SchedulerConfig


class TestQueueConfig:
    """Tests for QueueConfig dataclass defaults."""

    def test_defaults(self):
        """Default values should match spec."""
        cfg = QueueConfig()
        assert cfg.max_pending_requests == 1000
        assert cfg.warn_threshold_pct == 0.8


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass defaults."""

    def test_defaults(self):
        """Default values should match spec."""
        cfg = SchedulerConfig()
        assert cfg.batch_max_wait_seconds == 30.0
        assert cfg.length_buckets == [64, 256, 1024, 4096]

    def test_length_buckets_none_uses_default(self):
        """Explicit None for length_buckets triggers __post_init__ default."""
        cfg = SchedulerConfig(length_buckets=None)
        assert cfg.length_buckets == [64, 256, 1024, 4096]

    def test_custom_length_buckets(self):
        """Custom length_buckets should be preserved."""
        cfg = SchedulerConfig(length_buckets=[32, 128])
        assert cfg.length_buckets == [32, 128]


class TestPagodaConfig:
    """Tests for PagodaConfig YAML loading."""

    def _write_yaml(self, tmp_path, data):
        """Helper to write a YAML config file."""
        path = tmp_path / "pagoda_config.yaml"
        path.write_text(yaml.dump(data))
        return str(path)

    def test_normal_load(self, tmp_path):
        """YAML with all sections loads correctly."""
        data = {
            "queue": {
                "max_pending_requests": 500,
                "warn_threshold_pct": 0.9,
            },
            "scheduler": {
                "batch_max_wait_seconds": 15.0,
                "length_buckets": [32, 128, 512],
            },
            "defaults": {
                "priority": "high",
                "qps_limit": 20.0,
                "concurrent_limit": 10,
                "kv_cache_block_quota": 200,
            },
            "mass_api": {
                "url": "http://mass:8080/api",
                "timeout_seconds": 5,
                "tenant_config_ttl_seconds": 600,
                "stale_cache_ttl_seconds": 7200,
            },
        }
        path = self._write_yaml(tmp_path, data)
        cfg = PagodaConfig(path)

        assert cfg.queue.max_pending_requests == 500
        assert cfg.queue.warn_threshold_pct == 0.9
        assert cfg.scheduler.batch_max_wait_seconds == 15.0
        assert cfg.scheduler.length_buckets == [32, 128, 512]
        assert cfg.mass_client._defaults.priority == "high"
        assert cfg.mass_client._defaults.qps_limit == 20.0
        assert cfg.mass_client._defaults.concurrent_limit == 10
        assert cfg.mass_client._defaults.kv_cache_block_quota == 200

    def test_file_not_found_uses_defaults(self, tmp_path):
        """Missing config file should use defaults without raising."""
        path = str(tmp_path / "nonexistent.yaml")
        cfg = PagodaConfig(path)

        assert cfg.queue.max_pending_requests == 1000
        assert cfg.queue.warn_threshold_pct == 0.8
        assert cfg.scheduler.batch_max_wait_seconds == 30.0

    def test_malformed_yaml_uses_defaults(self, tmp_path):
        """Malformed YAML should use defaults without raising."""
        path = tmp_path / "bad.yaml"
        path.write_text("{{invalid: yaml: [")
        cfg = PagodaConfig(str(path))

        assert cfg.queue.max_pending_requests == 1000
        assert cfg.scheduler.batch_max_wait_seconds == 30.0

    def test_custom_fields_override_defaults(self, tmp_path):
        """Only specified fields override; others keep defaults."""
        data = {
            "queue": {"max_pending_requests": 2000},
        }
        path = self._write_yaml(tmp_path, data)
        cfg = PagodaConfig(path)

        assert cfg.queue.max_pending_requests == 2000
        # warn_threshold_pct not specified → default
        assert cfg.queue.warn_threshold_pct == 0.8
        # scheduler not specified → all defaults
        assert cfg.scheduler.batch_max_wait_seconds == 30.0

    def test_get_queue_config(self, tmp_path):
        """get_queue_config() returns the queue config."""
        path = self._write_yaml(tmp_path, {})
        cfg = PagodaConfig(path)
        qcfg = cfg.get_queue_config()
        assert isinstance(qcfg, QueueConfig)
        assert qcfg is cfg.queue

    def test_get_scheduler_config(self, tmp_path):
        """get_scheduler_config() returns the scheduler config."""
        path = self._write_yaml(tmp_path, {})
        cfg = PagodaConfig(path)
        scfg = cfg.get_scheduler_config()
        assert isinstance(scfg, SchedulerConfig)
        assert scfg is cfg.scheduler

    def test_empty_yaml_file(self, tmp_path):
        """Empty YAML file (parses as None) should use defaults."""
        path = tmp_path / "empty.yaml"
        path.write_text("")
        cfg = PagodaConfig(str(path))

        assert cfg.queue.max_pending_requests == 1000
        assert cfg.scheduler.batch_max_wait_seconds == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
