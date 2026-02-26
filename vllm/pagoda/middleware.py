# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified Pagoda ASGI middleware for rate limiting + queue depth + observability."""

from __future__ import annotations

import json
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm.logger import init_logger
from vllm.pagoda.config import PagodaConfig
from vllm.pagoda.mass_client import MassApiClient
from vllm.pagoda.metrics import (
    pagoda_request_latency_seconds,
    pagoda_request_rejected_total,
    pagoda_request_total,
    pagoda_tenant_concurrent_requests,
)
from vllm.pagoda.queue_depth import QueueDepthTracker
from vllm.pagoda.rate_limiter import TenantRateLimiter
from vllm.pagoda.request_logger import (
    PagodaRequestLog,
    PagodaRequestLogger,
    RequestTimer,
)
from vllm.pagoda.tenant import TenantResolver, sanitize_metric_label

logger = init_logger(__name__)


class PagodaMiddleware:
    """Combined ASGI middleware: tenant resolution → rate limit → queue depth.

    Follows the same pattern as vLLM's ScalingMiddleware and
    AuthenticationMiddleware.
    """

    def __init__(
        self,
        app: ASGIApp,
        config: PagodaConfig,
        mass_client: MassApiClient,
        trust_upstream_tenant_id: bool = True,
    ) -> None:
        self.app = app
        self.config = config
        self.tenant_resolver = TenantResolver(
            trust_upstream=trust_upstream_tenant_id
        )
        self.rate_limiter = TenantRateLimiter(mass_client)
        queue_cfg = config.get_queue_config()
        self.queue_tracker = QueueDepthTracker(
            max_pending=queue_cfg.max_pending_requests,
            warn_pct=queue_cfg.warn_threshold_pct,
        )
        self.request_logger = PagodaRequestLogger()

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Only apply to /v1/ API paths
        path: str = scope.get("path", "")
        if not path.startswith("/v1/"):
            await self.app(scope, receive, send)
            return

        # --- Tenant resolution ---
        headers = Headers(scope=scope)
        identity = self.tenant_resolver.resolve(headers)
        tenant_id = identity.tenant_id
        user_id = identity.user_id
        uid = sanitize_metric_label(user_id)

        # --- Rate limiting (also fetches tenant priority from MASS) ---
        result = await self.rate_limiter.acquire(tenant_id, user_id=user_id)
        tenant_priority_str = result.priority

        # Store tenant_id, user_id, and priority in scope state for downstream
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["pagoda_tenant_id"] = tenant_id
        scope["state"]["pagoda_user_id"] = user_id
        scope["state"]["pagoda_tenant_priority"] = tenant_priority_str

        if not result.allowed:
            pagoda_request_rejected_total.labels(
                tenant_id=tenant_id or "__unknown__",
                user_id=uid,
                reason=result.reason or "rate_limit",
            ).inc()
            pagoda_request_total.labels(
                tenant_id=tenant_id or "__unknown__",
                user_id=uid,
                model="__unknown__",
                status="rejected",
            ).inc()
            self.request_logger.log(PagodaRequestLog(
                request_id=scope.get("state", {}).get("request_id", ""),
                tenant_id=tenant_id or "__unknown__",
                user_id=user_id,
                model="__unknown__",
                priority=tenant_priority_str or "normal",
                status="rejected",
                error_reason=f"rate_limit:{result.reason}",
            ))
            await self._send_error(
                send,
                status=429,
                message=f"Rate limit exceeded: {result.reason}",
                headers={
                    "retry-after": str(int(result.retry_after or 1)),
                    "x-pagoda-tenant": tenant_id or "__unknown__",
                },
            )
            return

        # --- Queue depth ---
        if not self.queue_tracker.acquire():
            # Release rate limiter since we're rejecting
            await self.rate_limiter.release(tenant_id, result.semaphore_ref)
            pagoda_request_rejected_total.labels(
                tenant_id=tenant_id or "__unknown__",
                user_id=uid,
                reason="queue_full",
            ).inc()
            pagoda_request_total.labels(
                tenant_id=tenant_id or "__unknown__",
                user_id=uid,
                model="__unknown__",
                status="rejected",
            ).inc()
            self.request_logger.log(PagodaRequestLog(
                request_id=scope.get("state", {}).get("request_id", ""),
                tenant_id=tenant_id or "__unknown__",
                user_id=user_id,
                model="__unknown__",
                priority=tenant_priority_str or "normal",
                status="rejected",
                error_reason="queue_full",
            ))
            await self._send_error(
                send,
                status=503,
                message="Server overloaded: queue depth limit reached",
                headers={
                    "x-queue-depth": str(self.queue_tracker.current_depth),
                },
            )
            return

        # --- Process request with try/finally for guaranteed cleanup ---
        timer = RequestTimer()
        pagoda_tenant_concurrent_requests.labels(
            tenant_id=tenant_id, user_id=uid,
        ).inc()

        # Track response status from downstream
        response_status: int = 200
        released = False

        # Wrap send to inject X-Queue-Depth header and capture status
        async def send_with_headers(message: Message) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message.get("status", 200)
                response_headers = MutableHeaders(scope=message)
                response_headers.append(
                    "x-queue-depth", str(self.queue_tracker.current_depth)
                )
                response_headers.append(
                    "x-pagoda-tenant", tenant_id or "__unknown__"
                )
            await send(message)

        try:
            timer.mark_inference_start()
            await self.app(scope, receive, send_with_headers)
        except Exception:
            response_status = 500
            logger.exception(
                "Unhandled error processing request for tenant=%s user=%s",
                tenant_id,
                user_id,
            )
            raise
        finally:
            timer.mark_done()
            if not released:
                released = True
                await self.rate_limiter.release(
                    tenant_id, result.semaphore_ref
                )
                self.queue_tracker.release()
            pagoda_tenant_concurrent_requests.labels(
                tenant_id=tenant_id, user_id=uid,
            ).dec()

            # Determine status label
            status_label = (
                "success" if 200 <= response_status < 400 else "error"
            )
            model = scope.get("state", {}).get("model", "__unknown__")

            # Record metrics
            pagoda_request_total.labels(
                tenant_id=tenant_id or "__unknown__",
                user_id=uid,
                model=model,
                status=status_label,
            ).inc()
            pagoda_request_latency_seconds.labels(
                tenant_id=tenant_id or "__unknown__",
                user_id=uid,
                model=model,
            ).observe(timer.total_ms / 1000.0)

            # Emit structured log
            self.request_logger.log(PagodaRequestLog(
                request_id=scope.get("state", {}).get("request_id", ""),
                tenant_id=tenant_id or "__unknown__",
                user_id=user_id,
                model=model,
                priority=tenant_priority_str or "normal",
                total_ms=timer.total_ms,
                queue_wait_ms=timer.queue_wait_ms,
                inference_ms=timer.inference_ms,
                status=status_label,
            ))

    @staticmethod
    async def _send_error(
        send: Send,
        status: int,
        message: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Send a JSON error response."""
        body = json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "server_error" if status >= 500 else "rate_limit_error",
                    "code": status,
                }
            }
        ).encode()

        raw_headers: list[tuple[bytes, bytes]] = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        if headers:
            for k, v in headers.items():
                raw_headers.append((k.encode(), v.encode()))

        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": raw_headers,
            }
        )
        await send({"type": "http.response.body", "body": body})
