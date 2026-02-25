# Pagoda Multi-Tenant Serving

## Introduction

Pagoda is a multi-tenant serving layer for vLLM. It adds tenant isolation, per-tenant rate limiting, priority-aware scheduling, tool-call JSON repair, and structured request logging — all without modifying vLLM's core inference path.

Pagoda is designed for shared GPU clusters where multiple teams or customers send requests to the same vLLM instance. Each tenant gets its own QPS limit, concurrency limit, KV-cache block quota, and scheduling priority, configured dynamically via the MASS platform API.

## Enabling Pagoda

Pass a YAML config file to the vLLM server with `--pagoda-config`:

```bash
vllm serve meta-llama/Llama-3-8B-Instruct --pagoda-config pagoda_config.yaml
```

## Configuration

A minimal `pagoda_config.yaml`:

```yaml
queue:
  max_pending_requests: 1000    # global queue depth cap
  warn_threshold_pct: 0.8       # log warning at 80% capacity

scheduler:
  batch_max_wait_seconds: 30.0  # max wait for batch requests
  length_buckets: [64, 256, 1024, 4096]

defaults:                        # fallback when MASS is unavailable
  priority: normal               # high | normal | batch
  qps_limit: 10.0
  concurrent_limit: 5
  kv_cache_block_quota: 100

mass_api:
  url: http://mass:8080/api
  timeout_seconds: 2
  tenant_config_ttl_seconds: 300
  stale_cache_ttl_seconds: 3600
```

All sections are optional. Missing values fall back to the defaults shown above.

## Tenant Resolution

Pagoda resolves the tenant from the `X-Tenant-ID` HTTP header. When `trust_upstream_tenant_id` is enabled (the default), the header value is used directly. When disabled, all requests are assigned to the `__default__` tenant.

Requests without the header are always assigned to `__default__`.

## Rate Limiting

Each tenant is subject to two independent limits, fetched from the MASS platform API:

- **QPS limit** — token-bucket rate limiter. Requests exceeding the limit receive HTTP 429 with a `Retry-After` header.
- **Concurrency limit** — semaphore-based. Requests exceeding the limit also receive HTTP 429.

## Queue Depth Protection

A global queue depth tracker rejects requests with HTTP 503 when the total pending request count exceeds `max_pending_requests`. A warning is logged when the queue crosses `warn_threshold_pct`.

## Priority Scheduling

Tenants are assigned a priority (`high`, `normal`, or `batch`) via MASS. The scheduler uses this priority along with sequence-length bucketing to order requests within each scheduling cycle.

## Tool Call JSON Repair

Pagoda includes a postprocessor that attempts to repair malformed tool-call JSON from model output. It applies a cascade of strategies:

1. Standard `json.loads`
2. `json5` relaxed parsing (single quotes, trailing commas)
3. Model-specific extraction (DeepSeek `<tool_call>` tags, Qwen `✿FUNCTION✿` format)
4. Regex-based JSON extraction from surrounding text
5. Bracket-balancing repair for truncated output

Repairs are tracked via the `pagoda_tool_call_repair_total` Prometheus metric.

## Structured Request Logging

Every completed request emits a single JSON log line to a dedicated logger (`pagoda.request_log`), containing:

- `request_id`, `tenant_id`, `model`, `priority`
- `prompt_tokens`, `output_tokens`
- `queue_wait_ms`, `inference_ms`, `total_ms`
- `status` (`success` | `error` | `rejected`)
- `tool_call`, `tool_call_repaired`, `preempted`
- `kv_cache_blocks_used`, `timestamp`

These logs are compatible with standard aggregation systems (ELK, Loki, CloudWatch).

## Prometheus Metrics

Pagoda exports the following metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `pagoda_request_total` | Counter | Total requests by tenant, priority, status |
| `pagoda_request_latency_seconds` | Histogram | End-to-end request latency by tenant |
| `pagoda_request_rejected_total` | Counter | Rejected requests by tenant and reason |
| `pagoda_queue_depth_current` | Gauge | Current global queue depth |
| `pagoda_queue_rejected_total` | Counter | Requests rejected by queue depth cap |
| `pagoda_rate_limit_rejected_total` | Counter | Requests rejected by rate limiter |
| `pagoda_tool_call_repair_total` | Counter | Tool call repairs by model and outcome |
| `pagoda_tenant_concurrent_requests` | Gauge | Active concurrent requests per tenant |

## Graceful Degradation

When the MASS platform API is unavailable:

1. Cached tenant configs continue to be served until `stale_cache_ttl_seconds` expires.
2. After stale expiry, the `defaults` section from the YAML config is used.
3. An alert is emitted so operators can investigate.

No requests are dropped due to MASS unavailability alone.

## Architecture

```
Request → PagodaMiddleware
            ├── TenantResolver (X-Tenant-ID header)
            ├── MassApiClient (fetch tenant config, cached)
            ├── TenantRateLimiter (QPS + concurrency)
            ├── QueueDepthTracker (global cap)
            ├── vLLM inference (unmodified)
            ├── ToolCallPostProcessor (JSON repair)
            └── PagodaRequestLogger (structured JSON)
```

All Pagoda code lives under `vllm/pagoda/` and is tested in `tests/pagoda/`.
