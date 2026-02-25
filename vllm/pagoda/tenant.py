# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tenant and user identification from request headers."""

from __future__ import annotations

from dataclasses import dataclass

from starlette.datastructures import Headers

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_TENANT = "__default__"


@dataclass
class ResolvedIdentity:
    """Result of tenant + user resolution from request headers."""

    tenant_id: str
    user_id: str | None = None

    @property
    def rate_limit_key(self) -> str:
        """Composite key for rate limiting and quota enforcement.

        Returns ``tenant_id:user_id`` when user_id is present,
        otherwise plain ``tenant_id``.
        """
        if self.user_id:
            return f"{self.tenant_id}:{self.user_id}"
        return self.tenant_id


class TenantResolver:
    """Extract tenant_id and optional user_id from request headers.

    Primary mode: trust upstream gateway's X-Tenant-ID / X-User-ID headers.
    Fallback: return __default__ tenant when header is missing.
    """

    def __init__(self, trust_upstream: bool = True) -> None:
        self._trust_upstream = trust_upstream

    def resolve(self, headers: Headers) -> ResolvedIdentity:
        """Resolve tenant_id and user_id from headers.

        Returns a ``ResolvedIdentity``.  Missing X-Tenant-ID → "__default__".
        Missing X-User-ID → None (rate limiting falls back to tenant level).
        """
        if self._trust_upstream:
            tenant_id = headers.get("x-tenant-id", _DEFAULT_TENANT)
            user_id = headers.get("x-user-id") or None
            return ResolvedIdentity(tenant_id=tenant_id, user_id=user_id)

        # Fallback for local testing without gateway
        return ResolvedIdentity(tenant_id=_DEFAULT_TENANT)
