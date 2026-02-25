# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda tenant resolution from request headers."""

import pytest
from starlette.datastructures import Headers

from vllm.pagoda.tenant import TenantResolver


class TestTenantResolver:
    """Tests for TenantResolver."""

    def test_trust_upstream_with_header(self):
        """trust_upstream=True + X-Tenant-ID header → returns header value."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers(headers=[(b"x-tenant-id", b"acme-corp")])
        assert resolver.resolve(headers) == "acme-corp"

    def test_trust_upstream_without_header(self):
        """trust_upstream=True + no header → returns __default__."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers(headers=[])
        assert resolver.resolve(headers) == "__default__"

    def test_no_trust_upstream_always_default(self):
        """trust_upstream=False → always returns __default__ even with header."""
        resolver = TenantResolver(trust_upstream=False)
        headers = Headers(headers=[(b"x-tenant-id", b"acme-corp")])
        assert resolver.resolve(headers) == "__default__"

    def test_no_trust_upstream_no_header(self):
        """trust_upstream=False + no header → returns __default__."""
        resolver = TenantResolver(trust_upstream=False)
        headers = Headers(headers=[])
        assert resolver.resolve(headers) == "__default__"

    def test_default_trust_upstream_is_true(self):
        """Default trust_upstream should be True."""
        resolver = TenantResolver()
        headers = Headers(headers=[(b"x-tenant-id", b"tenant-x")])
        assert resolver.resolve(headers) == "tenant-x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
