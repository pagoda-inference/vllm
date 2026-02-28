# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda tenant resolution from request headers."""

import pytest
from starlette.datastructures import Headers

from vllm.pagoda.tenant import (
    ResolvedIdentity,
    TenantResolver,
    sanitize_metric_label,
    _MAX_ID_LENGTH,
)


class TestTenantResolver:
    """Tests for TenantResolver."""

    def test_trust_upstream_with_header(self):
        """trust_upstream=True + X-Tenant-ID header → returns header value."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers([(b"x-tenant-id", b"acme-corp")])
        identity = resolver.resolve(headers)
        assert isinstance(identity, ResolvedIdentity)
        assert identity.tenant_id == "acme-corp"
        assert identity.user_id is None

    def test_trust_upstream_without_header(self):
        """trust_upstream=True + no header → returns __default__."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers([])
        identity = resolver.resolve(headers)
        assert identity.tenant_id == "__default__"
        assert identity.user_id is None

    def test_no_trust_upstream_always_default(self):
        """trust_upstream=False → always returns __default__ even with header."""
        resolver = TenantResolver(trust_upstream=False)
        headers = Headers([(b"x-tenant-id", b"acme-corp")])
        identity = resolver.resolve(headers)
        assert identity.tenant_id == "__default__"
        assert identity.user_id is None

    def test_no_trust_upstream_no_header(self):
        """trust_upstream=False + no header → returns __default__."""
        resolver = TenantResolver(trust_upstream=False)
        headers = Headers([])
        identity = resolver.resolve(headers)
        assert identity.tenant_id == "__default__"

    def test_default_trust_upstream_is_true(self):
        """Default trust_upstream should be True."""
        resolver = TenantResolver()
        headers = Headers([(b"x-tenant-id", b"tenant-x")])
        assert resolver.resolve(headers).tenant_id == "tenant-x"

    def test_user_id_extracted_from_header(self):
        """X-User-ID header is extracted when trust_upstream=True."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers([
            (b"x-tenant-id", b"acme"),
            (b"x-user-id", b"user-42"),
        ])
        identity = resolver.resolve(headers)
        assert identity.tenant_id == "acme"
        assert identity.user_id == "user-42"

    def test_user_id_none_when_missing(self):
        """Missing X-User-ID → user_id is None."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers([(b"x-tenant-id", b"acme")])
        assert resolver.resolve(headers).user_id is None

    def test_user_id_none_when_no_trust(self):
        """trust_upstream=False → user_id is always None."""
        resolver = TenantResolver(trust_upstream=False)
        headers = Headers([
            (b"x-tenant-id", b"acme"),
            (b"x-user-id", b"user-42"),
        ])
        identity = resolver.resolve(headers)
        assert identity.tenant_id == "__default__"
        assert identity.user_id is None

    def test_empty_user_id_treated_as_none(self):
        """Empty X-User-ID header → user_id is None."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers([
            (b"x-tenant-id", b"acme"),
            (b"x-user-id", b""),
        ])
        assert resolver.resolve(headers).user_id is None


    def test_user_id_rejected_when_too_long(self):
        """X-User-ID exceeding max length → user_id is None."""
        resolver = TenantResolver(trust_upstream=True)
        long_id = "u" * (_MAX_ID_LENGTH + 1)
        headers = Headers([
            (b"x-tenant-id", b"acme"),
            (b"x-user-id", long_id.encode()),
        ])
        assert resolver.resolve(headers).user_id is None

    def test_user_id_rejected_when_invalid_chars(self):
        """X-User-ID with special characters → user_id is None."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers([
            (b"x-tenant-id", b"acme"),
            (b"x-user-id", b"user;DROP TABLE"),
        ])
        assert resolver.resolve(headers).user_id is None

    def test_tenant_id_rejected_when_too_long(self):
        """X-Tenant-ID exceeding max length → falls back to __default__."""
        resolver = TenantResolver(trust_upstream=True)
        long_id = "t" * (_MAX_ID_LENGTH + 1)
        headers = Headers([
            (b"x-tenant-id", long_id.encode()),
        ])
        assert resolver.resolve(headers).tenant_id == "__default__"

    def test_tenant_id_rejected_when_invalid_chars(self):
        """X-Tenant-ID with special characters → falls back to __default__."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers([
            (b"x-tenant-id", b"tenant\x00evil"),
        ])
        assert resolver.resolve(headers).tenant_id == "__default__"

    def test_valid_ids_with_allowed_special_chars(self):
        """IDs with dots, underscores, colons, @ are accepted."""
        resolver = TenantResolver(trust_upstream=True)
        headers = Headers([
            (b"x-tenant-id", b"org.acme_corp:prod"),
            (b"x-user-id", b"user@example.com"),
        ])
        identity = resolver.resolve(headers)
        assert identity.tenant_id == "org.acme_corp:prod"
        assert identity.user_id == "user@example.com"


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


class TestSanitizeMetricLabel:
    """Tests for sanitize_metric_label."""

    def test_none_returns_empty(self):
        assert sanitize_metric_label(None) == ""

    def test_empty_returns_empty(self):
        assert sanitize_metric_label("") == ""

    def test_valid_id_passes_through(self):
        assert sanitize_metric_label("user-42") == "user-42"

    def test_valid_id_with_special_chars(self):
        assert sanitize_metric_label("user@org.com:main") == "user@org.com:main"

    def test_too_long_returns_sentinel(self):
        assert sanitize_metric_label("x" * (_MAX_ID_LENGTH + 1)) == "__invalid__"

    def test_invalid_chars_returns_sentinel(self):
        assert sanitize_metric_label("user;DROP TABLE") == "__invalid__"

    def test_max_length_exactly_passes(self):
        val = "a" * _MAX_ID_LENGTH
        assert sanitize_metric_label(val) == val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
