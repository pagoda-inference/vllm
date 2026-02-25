# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda tenant resolution from request headers."""

import pytest
from starlette.datastructures import Headers

from vllm.pagoda.tenant import ResolvedIdentity, TenantResolver


class TestTenantResolver:
    """Tests for TenantResolver."""

    def test_trust_upstream_with_header(self):
        """trust_upstream=True + X-Tenant-ID header → returns header value."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers(headers=[(b"x-tenant-id", b"acme-corp")])
        identity = resolver.resolve(headers)
        assert isinstance(identity, ResolvedIdentity)
        assert identity.tenant_id == "acme-corp"
        assert identity.user_id is None

    def test_trust_upstream_without_header(self):
        """trust_upstream=True + no header → returns __default__."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers(headers=[])
        identity = resolver.resolve(headers)
        assert identity.tenant_id == "__default__"
        assert identity.user_id is None

    def test_no_trust_upstream_always_default(self):
        """trust_upstream=False → always returns __default__ even with header."""
        resolver = TenantResolver(trust_upstream=False)
        headers = Headers(headers=[(b"x-tenant-id", b"acme-corp")])
        identity = resolver.resolve(headers)
        assert identity.tenant_id == "__default__"
        assert identity.user_id is None

    def test_no_trust_upstream_no_header(self):
        """trust_upstream=False + no header → returns __default__."""
        resolver = TenantResolver(trust_upstream=False)
        headers = Headers(headers=[])
        identity = resolver.resolve(headers)
        assert identity.tenant_id == "__default__"

    def test_default_trust_upstream_is_true(self):
        """Default trust_upstream should be True."""
        resolver = TenantResolver()
        headers = Headers(headers=[(b"x-tenant-id", b"tenant-x")])
        assert resolver.resolve(headers).tenant_id == "tenant-x"

    def test_user_id_extracted_from_header(self):
        """X-User-ID header is extracted when trust_upstream=True."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers(headers=[
            (b"x-tenant-id", b"acme"),
            (b"x-user-id", b"user-42"),
        ])
        identity = resolver.resolve(headers)
        assert identity.tenant_id == "acme"
        assert identity.user_id == "user-42"

    def test_user_id_none_when_missing(self):
        """Missing X-User-ID → user_id is None."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers(headers=[(b"x-tenant-id", b"acme")])
        assert resolver.resolve(headers).user_id is None

    def test_user_id_none_when_no_trust(self):
        """trust_upstream=False → user_id is always None."""
        resolver = TenantResolver(trust_upstream=False)
        headers = Headers(headers=[
            (b"x-tenant-id", b"acme"),
            (b"x-user-id", b"user-42"),
        ])
        identity = resolver.resolve(headers)
        assert identity.tenant_id == "__default__"
        assert identity.user_id is None

    def test_empty_user_id_treated_as_none(self):
        """Empty X-User-ID header → user_id is None."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers(headers=[
            (b"x-tenant-id", b"acme"),
            (b"x-user-id", b""),
        ])
        assert resolver.resolve(headers).user_id is None


class TestResolvedIdentity:
    """Tests for ResolvedIdentity."""

    def test_rate_limit_key_tenant_only(self):
        """rate_limit_key is just tenant_id when user_id is None."""
        identity = ResolvedIdentity(tenant_id="acme")
        assert identity.rate_limit_key == "acme"

    def test_rate_limit_key_with_user(self):
        """rate_limit_key is tenant_id:user_id when user_id is set."""
        identity = ResolvedIdentity(tenant_id="acme", user_id="u1")
        assert identity.rate_limit_key == "acme:u1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
