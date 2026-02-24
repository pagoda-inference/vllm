# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for Pagoda multi-tenant features.

Metrics are defined here and incremented in the relevant modules.
"""

from prometheus_client import Counter, Gauge

# Rate limiting
pagoda_rate_limit_rejected_total = Counter(
    "pagoda_rate_limit_rejected_total",
    "Total requests rejected due to rate limit (429)",
    ["tenant_id", "reason"],  # reason: qps_limit | concurrent_limit
)

# Queue depth
pagoda_queue_rejected_total = Counter(
    "pagoda_queue_rejected_total",
    "Total requests rejected due to queue full (503)",
    ["reason"],
)

pagoda_queue_depth_current = Gauge(
    "pagoda_queue_depth_current",
    "Current number of pending requests in queue",
)

# Per-tenant concurrency
pagoda_tenant_concurrent_requests = Gauge(
    "pagoda_tenant_concurrent_requests",
    "Current concurrent requests per tenant",
    ["tenant_id"],
)

# MASS platform availability
pagoda_mass_api_errors_total = Counter(
    "pagoda_mass_api_errors_total",
    "Total times MASS API was unavailable (served from stale cache or defaults)",
)

# Tool call repair
pagoda_tool_call_repair_total = Counter(
    "pagoda_tool_call_repair_total",
    "Total tool call repair attempts",
    ["tenant_id", "model", "status"],  # status: success | repaired | failed
)

# KV cache quota metrics
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

# KV cache offloading metrics
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
