# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tenant identification from request headers."""

from __future__ import annotations

from starlette.datastructures import Headers

from vllm.logger import init_logger
from vllm.pagoda.config import PagodaConfig

logger = init_logger(__name__)

_DEFAULT_TENANT = "__default__"


class TenantResolver:
    """Extract tenant_id from request headers.

    Two modes:
    1. API Key mapping (default): Authorization header → api_keys lookup
    2. Trust upstream: X-Tenant-ID header directly (gateway already authed)
    """

    def __init__(
        self,
        config: PagodaConfig,
        trust_upstream_tenant_id: bool = False,
    ) -> None:
        self._config = config
        self._trust_upstream = trust_upstream_tenant_id

    def resolve(self, headers: Headers) -> str | None:
        """Resolve tenant_id from headers.

        Returns:
            tenant_id string, or None if authentication failed
            (API key mode only — unknown key → None → 401).
            In trust-upstream mode, missing header → "__default__".
        """
        if self._trust_upstream:
            return headers.get("x-tenant-id", _DEFAULT_TENANT)

        # API key mapping mode
        auth = headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            # No bearer token — let vLLM's own AuthenticationMiddleware
            # handle the 401. We return default so middleware doesn't block.
            return _DEFAULT_TENANT

        api_key = auth[7:].strip()  # strip "Bearer "
        tenant_id = self._config.resolve_api_key(api_key)
        if tenant_id is None:
            # Key not in pagoda config — could be a valid vLLM api-key
            # that just isn't mapped to a tenant. Use default.
            return _DEFAULT_TENANT

        return tenant_id
