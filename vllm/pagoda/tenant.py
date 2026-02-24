# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tenant identification from request headers."""

from __future__ import annotations

from starlette.datastructures import Headers

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_TENANT = "__default__"


class TenantResolver:
    """Extract tenant_id from request headers.

    Primary mode: trust upstream gateway's X-Tenant-ID header.
    Fallback: return __default__ when header is missing.
    """

    def __init__(self, trust_upstream: bool = True) -> None:
        self._trust_upstream = trust_upstream

    def resolve(self, headers: Headers) -> str:
        """Resolve tenant_id from headers.

        Returns tenant_id string. Missing header → "__default__".
        """
        if self._trust_upstream:
            return headers.get("x-tenant-id", _DEFAULT_TENANT)

        # Fallback for local testing without gateway
        return _DEFAULT_TENANT
