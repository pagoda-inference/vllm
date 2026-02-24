# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 1 Prometheus metrics for Pagoda multi-tenant features.

Metrics are defined here and incremented in the relevant modules.
The Prometheus endpoint exposure is deferred to Phase 4.
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
