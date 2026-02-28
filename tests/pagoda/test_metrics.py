# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda Prometheus metrics definitions."""

import pytest
from prometheus_client import Counter, Gauge, Histogram

from vllm.pagoda import metrics as m


class TestMetricsDefined:
    """Verify all expected metrics exist and have correct types/labels."""

    # --- Request-level metrics ---

    def test_request_total(self):
        assert isinstance(m.pagoda_request_total, Counter)
        assert m.pagoda_request_total._labelnames == (
            "tenant_id", "user_id", "model", "status",
        )

    def test_request_latency_seconds(self):
        assert isinstance(m.pagoda_request_latency_seconds, Histogram)
        assert m.pagoda_request_latency_seconds._labelnames == (
            "tenant_id", "user_id", "model",
        )

    def test_request_queue_wait_seconds(self):
        assert isinstance(m.pagoda_request_queue_wait_seconds, Histogram)
        assert m.pagoda_request_queue_wait_seconds._labelnames == (
            "tenant_id", "model",
        )

    def test_request_rejected_total(self):
        assert isinstance(m.pagoda_request_rejected_total, Counter)
        assert m.pagoda_request_rejected_total._labelnames == (
            "tenant_id", "user_id", "reason",
        )

    # --- Rate limiting ---

    def test_rate_limit_rejected_total(self):
        assert isinstance(m.pagoda_rate_limit_rejected_total, Counter)

    # --- Queue depth ---

    def test_queue_rejected_total(self):
        assert isinstance(m.pagoda_queue_rejected_total, Counter)

    def test_queue_depth_current(self):
        assert isinstance(m.pagoda_queue_depth_current, Gauge)

    # --- Concurrency ---

    def test_tenant_concurrent_requests(self):
        assert isinstance(m.pagoda_tenant_concurrent_requests, Gauge)

    # --- Tool call ---

    def test_tool_call_repair_total(self):
        assert isinstance(m.pagoda_tool_call_repair_total, Counter)

    def test_tool_call_format_errors(self):
        assert isinstance(m.pagoda_tool_call_format_errors, Counter)
        assert "error_type" in m.pagoda_tool_call_format_errors._labelnames

    # --- KV cache ---

    def test_kv_cache_blocks_used(self):
        assert isinstance(m.pagoda_kv_cache_blocks_used, Gauge)

    def test_kv_cache_blocks_total(self):
        assert isinstance(m.pagoda_kv_cache_blocks_total, Gauge)

    def test_quota_exceeded_total(self):
        assert isinstance(m.pagoda_quota_exceeded_total, Counter)

    # --- Offloading ---

    def test_offload_blocks_used(self):
        assert isinstance(m.pagoda_offload_blocks_used, Gauge)
        assert "tier" in m.pagoda_offload_blocks_used._labelnames

    def test_offload_evictions_total(self):
        assert isinstance(m.pagoda_offload_evictions_total, Counter)

    def test_offload_hits_total(self):
        assert isinstance(m.pagoda_offload_hits_total, Counter)

    def test_offload_misses_total(self):
        assert isinstance(m.pagoda_offload_misses_total, Counter)

    # --- System-level ---

    def test_gpu_utilization_percent(self):
        assert isinstance(m.pagoda_gpu_utilization_percent, Gauge)
        assert m.pagoda_gpu_utilization_percent._labelnames == ("device",)

    def test_gpu_memory_used_gb(self):
        assert isinstance(m.pagoda_gpu_memory_used_gb, Gauge)

    def test_batch_size(self):
        assert isinstance(m.pagoda_batch_size, Histogram)

    def test_tokens_per_second(self):
        assert isinstance(m.pagoda_tokens_per_second, Gauge)

    # --- Scheduling ---

    def test_preemption_total(self):
        assert isinstance(m.pagoda_preemption_total, Counter)
        assert "reason" in m.pagoda_preemption_total._labelnames

    def test_swap_in_total(self):
        assert isinstance(m.pagoda_swap_in_total, Counter)

    def test_swap_out_total(self):
        assert isinstance(m.pagoda_swap_out_total, Counter)

    # --- Prefix cache ---

    def test_prefix_cache_hit_rate(self):
        assert isinstance(m.pagoda_prefix_cache_hit_rate, Gauge)
        assert m.pagoda_prefix_cache_hit_rate._labelnames == ("model",)
