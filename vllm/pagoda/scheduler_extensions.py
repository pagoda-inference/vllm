# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pagoda scheduler extensions for multi-tenant priority scheduling.

This module provides helper functions for mapping tenant priorities to
scheduler priority values and determining when batch requests should be
promoted to prevent starvation.
"""

from __future__ import annotations

import time

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)


def map_tenant_priority(tenant_priority_str: str) -> int:
    """Map tenant priority string to scheduler priority integer.

    Lower priority values are scheduled first in the priority queue.

    Args:
        tenant_priority_str: Tenant priority level ("high", "normal", or "batch")

    Returns:
        Priority integer value:
        - "high" → 0 (highest priority)
        - "normal" → 100 (medium priority)
        - "batch" → 200 (lowest priority)
    """
    priority_map = {
        "high": 0,
        "normal": 100,
        "batch": 200,
    }
    return priority_map.get(tenant_priority_str, 100)  # Default to normal


def should_promote_batch_request(
    request: Request,
    max_wait_seconds: float,
) -> bool:
    """Check if a batch-priority request should be promoted to normal priority.

    Batch requests are promoted after waiting for max_wait_seconds to prevent
    starvation. This ensures that even low-priority requests eventually get
    scheduled.

    Args:
        request: The request to check
        max_wait_seconds: Maximum wait time before promotion

    Returns:
        True if the request should be promoted, False otherwise
    """
    if request.tenant_priority_str != "batch":
        return False

    wait_time = time.time() - request.queued_at
    return wait_time >= max_wait_seconds
