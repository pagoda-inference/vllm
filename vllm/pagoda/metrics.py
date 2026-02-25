# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for Pagoda multi-tenant features.

All Pagoda metrics are defined here and incremented in the relevant modules.
Metrics are exposed via vLLM's existing /metrics endpoint.

Naming convention: pagoda_<domain>_<metric_name>_<unit>
"""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Request-level metrics (Task 12)
# ---------------------------------------------------------------------------

pagoda_request_total = Counter(
    "pagoda_request_total",
    "Total requests by tenant, model, and status",
    ["tenant_id", "model", "status"],  # status: success | error | rejected
)

pagoda_request_latency_seconds = Histogram(
    "pagoda_request_latency_seconds",
    "End-to-end request latency in seconds",
    ["tenant_id", "model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

pagoda_request_queue_wait_seconds = Histogram(
    "pagoda_request_queue_wait_seconds",
    "Time spent waiting in queue before inference starts",
    ["tenant_id", "model"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

pagoda_request_rejected_total = Counter(
    "pagoda_request_rejected_total",
    "Total requests rejected (unified across all rejection reasons)",
    ["tenant_id", "reason"],
    # reason: qps_limit | queue_full | block_quota | preempted | concurrent_limit
)

# ---------------------------------------------------------------------------
# Rate limiting (Phase 1 — kept for backward compat, also feeds unified metric)
# ---------------------------------------------------------------------------

pagoda_rate_limit_rejected_total = Counter(
    "pagoda_rate_limit_rejected_total",
    "Total requests rejected due to rate limit (429)",
    ["tenant_id", "reason"],  # reason: qps_limit | concurrent_limit
)

# ---------------------------------------------------------------------------
# Queue depth
# ---------------------------------------------------------------------------

pagoda_queue_rejected_total = Counter(
    "pagoda_queue_rejected_total",
    "Total requests rejected due to queue full (503)",
    ["reason"],
)

pagoda_queue_depth_current = Gauge(
    "pagoda_queue_depth_current",
    "Current number of pending requests in queue",
)

# ---------------------------------------------------------------------------
# Per-tenant concurrency
# ---------------------------------------------------------------------------

pagoda_tenant_concurrent_requests = Gauge(
    "pagoda_tenant_concurrent_requests",
    "Current concurrent requests per tenant",
    ["tenant_id"],
)

# ---------------------------------------------------------------------------
# MASS platform availability
# ---------------------------------------------------------------------------

pagoda_mass_api_errors_total = Counter(
    "pagoda_mass_api_errors_total",
    "Total times MASS API was unavailable (served from stale cache or defaults)",
)

# ---------------------------------------------------------------------------
# Tool call metrics
# ---------------------------------------------------------------------------

pagoda_tool_call_repair_total = Counter(
    "pagoda_tool_call_repair_total",
    "Total tool call repair attempts",
    ["tenant_id", "model", "status"],  # status: success | repaired | failed
)

pagoda_tool_call_format_errors = Counter(
    "pagoda_tool_call_format_errors",
    "Total tool call format errors by error type",
    ["tenant_id", "model", "error_type"],
    # error_type: trailing_comma | single_quotes | missing_brace | extra_text | unknown
)

# ---------------------------------------------------------------------------
# KV cache quota metrics
# ---------------------------------------------------------------------------

pagoda_quota_exceeded_total = Counter(
    "pagoda_quota_exceeded_total",
    "Total requests rejected due to KV cache block quota exceeded",
    ["tenant_id"],
)

pagoda_kv_cache_blocks_used = Gauge(
    "pagoda_kv_cache_blocks_used",
    "Current KV cache blocks used per tenant",
    ["tenant_id"],
)

pagoda_kv_cache_blocks_total = Gauge(
    "pagoda_kv_cache_blocks_total",
    "Total KV cache blocks available in the system",
)

# ---------------------------------------------------------------------------
# KV cache offloading metrics
# ---------------------------------------------------------------------------

pagoda_offload_blocks_used = Gauge(
    "pagoda_offload_blocks_used",
    "Current offloaded blocks per tenant and tier",
    ["tenant_id", "tier"],  # tier: cpu, ssd
)

pagoda_offload_evictions_total = Counter(
    "pagoda_offload_evictions_total",
    "Total blocks evicted from offload cache per tenant",
    ["tenant_id", "tier"],
)

pagoda_offload_stores_total = Counter(
    "pagoda_offload_stores_total",
    "Total blocks stored to offload cache per tenant",
    ["tenant_id", "tier"],
)

pagoda_offload_hits_total = Counter(
    "pagoda_offload_hits_total",
    "Total offload cache lookup hits per tenant",
    ["tenant_id"],
)

pagoda_offload_misses_total = Counter(
    "pagoda_offload_misses_total",
    "Total offload cache lookup misses per tenant",
    ["tenant_id"],
)

# ---------------------------------------------------------------------------
# System-level metrics (GPU, batch, throughput)
# ---------------------------------------------------------------------------

pagoda_gpu_utilization_percent = Gauge(
    "pagoda_gpu_utilization_percent",
    "GPU compute utilization percentage",
    ["device"],
)

pagoda_gpu_memory_used_gb = Gauge(
    "pagoda_gpu_memory_used_gb",
    "GPU memory used in GB",
    ["device"],
)

pagoda_batch_size = Histogram(
    "pagoda_batch_size",
    "Number of requests in each scheduled batch",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

pagoda_tokens_per_second = Gauge(
    "pagoda_tokens_per_second",
    "Token generation throughput per model",
    ["model"],
)

# ---------------------------------------------------------------------------
# Scheduling metrics (preemption, swap)
# ---------------------------------------------------------------------------

pagoda_preemption_total = Counter(
    "pagoda_preemption_total",
    "Total preemption events",
    ["reason"],  # reason: memory | priority
)

pagoda_swap_in_total = Counter(
    "pagoda_swap_in_total",
    "Total swap-in operations (CPU → GPU)",
)

pagoda_swap_out_total = Counter(
    "pagoda_swap_out_total",
    "Total swap-out operations (GPU → CPU)",
)

# ---------------------------------------------------------------------------
# Prefix cache
# ---------------------------------------------------------------------------

pagoda_prefix_cache_hit_rate = Gauge(
    "pagoda_prefix_cache_hit_rate",
    "Prefix cache hit rate (0.0 - 1.0) per model",
    ["model"],
)
