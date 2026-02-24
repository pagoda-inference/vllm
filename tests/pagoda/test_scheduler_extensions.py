# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda multi-priority scheduler extensions."""

import time

import pytest

from vllm.pagoda.scheduler_extensions import (
    map_tenant_priority,
    should_promote_batch_request,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request


def test_map_tenant_priority():
    """Test tenant priority string to integer mapping."""
    # Test standard priority levels
    assert map_tenant_priority("high") == 0
    assert map_tenant_priority("normal") == 100
    assert map_tenant_priority("batch") == 200

    # Test default fallback for unknown priority
    assert map_tenant_priority("unknown") == 100
    assert map_tenant_priority("") == 100
    assert map_tenant_priority(None) == 100


def test_should_promote_batch_request():
    """Test batch request promotion logic."""
    sampling_params = SamplingParams(max_tokens=10)

    # Create a batch-priority request
    request = Request(
        request_id="test-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        pooling_params=None,
        tenant_id="tenant-1",
        tenant_priority_str="batch",
    )

    # Should not promote immediately
    assert not should_promote_batch_request(request, max_wait_seconds=1.0)

    # Wait and check promotion
    time.sleep(1.1)
    assert should_promote_batch_request(request, max_wait_seconds=1.0)


def test_should_not_promote_non_batch_requests():
    """Test that non-batch requests are not promoted."""
    sampling_params = SamplingParams(max_tokens=10)

    # Create high-priority request
    high_request = Request(
        request_id="test-high",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        pooling_params=None,
        tenant_id="tenant-1",
        tenant_priority_str="high",
    )

    # Create normal-priority request
    normal_request = Request(
        request_id="test-normal",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        pooling_params=None,
        tenant_id="tenant-2",
        tenant_priority_str="normal",
    )

    # Wait to ensure time has passed
    time.sleep(0.1)

    # Neither should be promoted
    assert not should_promote_batch_request(high_request, max_wait_seconds=0.05)
    assert not should_promote_batch_request(normal_request, max_wait_seconds=0.05)


def test_request_priority_ordering():
    """Test that requests are ordered correctly by priority and length_bucket."""
    sampling_params = SamplingParams(max_tokens=10)

    # Create requests with different priorities
    high_req = Request(
        request_id="high",
        prompt_token_ids=[1] * 10,
        sampling_params=sampling_params,
        pooling_params=None,
        priority=0,  # high
        tenant_priority_str="high",
    )

    normal_req = Request(
        request_id="normal",
        prompt_token_ids=[1] * 10,
        sampling_params=sampling_params,
        pooling_params=None,
        priority=100,  # normal
        tenant_priority_str="normal",
    )

    batch_req = Request(
        request_id="batch",
        prompt_token_ids=[1] * 10,
        sampling_params=sampling_params,
        pooling_params=None,
        priority=200,  # batch
        tenant_priority_str="batch",
    )

    # Test priority ordering (lower priority value = higher priority)
    assert high_req < normal_req
    assert normal_req < batch_req
    assert high_req < batch_req


def test_request_length_bucket_ordering():
    """Test that requests with same priority are ordered by length_bucket."""
    sampling_params = SamplingParams(max_tokens=10)

    # Create requests with same priority but different length buckets
    req_bucket_0 = Request(
        request_id="bucket-0",
        prompt_token_ids=[1] * 10,
        sampling_params=sampling_params,
        pooling_params=None,
        priority=100,
    )
    req_bucket_0.length_bucket = 0

    req_bucket_1 = Request(
        request_id="bucket-1",
        prompt_token_ids=[1] * 10,
        sampling_params=sampling_params,
        pooling_params=None,
        priority=100,
    )
    req_bucket_1.length_bucket = 1

    req_bucket_2 = Request(
        request_id="bucket-2",
        prompt_token_ids=[1] * 10,
        sampling_params=sampling_params,
        pooling_params=None,
        priority=100,
    )
    req_bucket_2.length_bucket = 2

    # Test length bucket ordering
    assert req_bucket_0 < req_bucket_1
    assert req_bucket_1 < req_bucket_2
    assert req_bucket_0 < req_bucket_2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
