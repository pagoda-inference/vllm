# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tenant and user identification from request headers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from starlette.datastructures import Headers

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_TENANT = "__default__"

# Validation constraints for header-sourced identifiers.
# Only alphanumeric, hyphens, underscores, dots, and colons are allowed.
_VALID_ID_RE = re.compile(r"^[a-zA-Z0-9._:@\-]+$")
_MAX_ID_LENGTH = 128

# Sentinel used in Prometheus labels when the raw user_id is invalid.
_INVALID_USER_LABEL = "__invalid__"


def sanitize_metric_label(raw: str | None) -> str:
    """Return a safe string for use as a Prometheus label value.

    * None / empty → ""
    * Valid identifier (matches ``_VALID_ID_RE``, within length) → passed through
    * Otherwise → ``__invalid__`` to cap cardinality
    """
    if not raw:
        return ""
    if len(raw) > _MAX_ID_LENGTH:
        logger.warning(
            "Metric label value too long (%d chars), replacing with sentinel",
            len(raw),
        )
        return _INVALID_USER_LABEL
    if not _VALID_ID_RE.match(raw):
        logger.warning(
            "Metric label value contains invalid characters, "
            "replacing with sentinel"
        )
        return _INVALID_USER_LABEL
    return raw


def _validate_header_id(value: str | None, header_name: str) -> str | None:
    """Validate and sanitize a header-sourced identifier.

    Returns the value unchanged if valid, or None if it fails validation.
    """
    if not value:
        return None
    if len(value) > _MAX_ID_LENGTH:
        logger.warning(
            "%s too long (%d chars), ignoring", header_name, len(value)
        )
        return None
    if not _VALID_ID_RE.match(value):
        logger.warning(
            "%s contains invalid characters, ignoring", header_name
        )
        return None
    return value


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

    All header values are validated against length and character constraints
    to prevent cardinality explosion in downstream Prometheus metrics.
    """

    def __init__(self, trust_upstream: bool = True) -> None:
        self._trust_upstream = trust_upstream

    def resolve(self, headers: Headers) -> ResolvedIdentity:
        """Resolve tenant_id and user_id from headers.

        Returns a ``ResolvedIdentity``.  Missing X-Tenant-ID → "__default__".
        Missing or invalid X-User-ID → None (rate limiting falls back to
        tenant level).
        """
        if self._trust_upstream:
            raw_tenant = headers.get("x-tenant-id") or None
            tenant_id = (
                _validate_header_id(raw_tenant, "X-Tenant-ID")
                or _DEFAULT_TENANT
            )
            raw_user = headers.get("x-user-id") or None
            user_id = _validate_header_id(raw_user, "X-User-ID")
            return ResolvedIdentity(tenant_id=tenant_id, user_id=user_id)

        # Fallback for local testing without gateway
        return ResolvedIdentity(tenant_id=_DEFAULT_TENANT)
