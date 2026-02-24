# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda length prediction and batch grouping."""

import pytest

from vllm.pagoda.length_predictor import (
    LENGTH_BUCKET_BOUNDARIES,
    assign_length_bucket,
    predict_output_length,
)


def test_predict_output_length_zero_input():
    """Test prediction for zero or negative input."""
    assert predict_output_length(0) == 64
    assert predict_output_length(-1) == 64


def test_predict_output_length_short_inputs():
    """Test prediction for short inputs (< 128 tokens)."""
    # Short inputs get 4x multiplier
    assert predict_output_length(10) == 40
    assert predict_output_length(50) == 200
    assert predict_output_length(100) == 400


def test_predict_output_length_medium_inputs():
    """Test prediction for medium inputs (128-512 tokens)."""
    # Medium inputs get 2x multiplier
    assert predict_output_length(128) == 256
    assert predict_output_length(256) == 512
    assert predict_output_length(500) == 1000


def test_predict_output_length_long_inputs():
    """Test prediction for long inputs (512-2048 tokens)."""
    # Long inputs get 1x multiplier
    assert predict_output_length(512) == 512
    assert predict_output_length(1024) == 1024
    assert predict_output_length(2000) == 2000


def test_predict_output_length_very_long_inputs():
    """Test prediction for very long inputs (> 2048 tokens)."""
    # Very long inputs get 0.5x multiplier
    assert predict_output_length(2048) == 1024
    assert predict_output_length(4096) == 2048
    assert predict_output_length(8192) == 4096


def test_predict_output_length_max_cap():
    """Test that predictions are capped at 4096."""
    # Even with large multipliers, output is capped
    assert predict_output_length(10000) <= 4096
    assert predict_output_length(100) <= 4096


def test_assign_length_bucket_boundaries():
    """Test length bucket assignment at boundaries."""
    # Test exact boundary values
    assert assign_length_bucket(64) == 0
    assert assign_length_bucket(256) == 1
    assert assign_length_bucket(1024) == 2
    assert assign_length_bucket(4096) == 3


def test_assign_length_bucket_below_boundaries():
    """Test length bucket assignment below boundaries."""
    assert assign_length_bucket(1) == 0
    assert assign_length_bucket(63) == 0
    assert assign_length_bucket(65) == 1
    assert assign_length_bucket(255) == 1
    assert assign_length_bucket(257) == 2
    assert assign_length_bucket(1023) == 2
    assert assign_length_bucket(1025) == 3
    assert assign_length_bucket(4095) == 3


def test_assign_length_bucket_above_max():
    """Test length bucket assignment above maximum boundary."""
    # Values above max boundary go to last bucket
    assert assign_length_bucket(4097) == len(LENGTH_BUCKET_BOUNDARIES)
    assert assign_length_bucket(10000) == len(LENGTH_BUCKET_BOUNDARIES)


def test_length_bucket_count():
    """Test that we have the expected number of buckets."""
    # Should have 4 boundaries, creating 5 buckets (0-4)
    assert len(LENGTH_BUCKET_BOUNDARIES) == 4
    assert LENGTH_BUCKET_BOUNDARIES == [64, 256, 1024, 4096]


@pytest.mark.parametrize(
    "input_tokens,expected_bucket",
    [
        (10, 0),  # 10 * 4 = 40 -> bucket 0
        (20, 0),  # 20 * 4 = 80 -> bucket 1, but capped at 64 boundary
        (100, 1),  # 100 * 4 = 400 -> bucket 1
        (200, 1),  # 200 * 2 = 400 -> bucket 1
        (600, 2),  # 600 * 1 = 600 -> bucket 2
        (1500, 2),  # 1500 * 1 = 1500 -> bucket 2
        (3000, 3),  # 3000 * 0.5 = 1500 -> bucket 2, but input is > 2048
        (5000, 3),  # 5000 * 0.5 = 2500 -> bucket 3
    ],
)
def test_end_to_end_prediction_and_bucketing(input_tokens, expected_bucket):
    """Test end-to-end flow from input tokens to bucket assignment."""
    predicted_len = predict_output_length(input_tokens)
    bucket = assign_length_bucket(predicted_len)
    assert bucket == expected_bucket


def test_similar_length_grouping():
    """Test that similar-length requests get grouped in same bucket."""
    # Requests with similar predicted lengths should be in same bucket
    inputs_group_1 = [10, 15, 20]  # All should predict to bucket 0
    inputs_group_2 = [200, 250, 300]  # All should predict to bucket 1
    inputs_group_3 = [1000, 1200, 1500]  # All should predict to bucket 2

    for inputs in [inputs_group_1, inputs_group_2, inputs_group_3]:
        buckets = [
            assign_length_bucket(predict_output_length(inp)) for inp in inputs
        ]
        # All buckets in group should be the same
        assert len(set(buckets)) == 1, f"Buckets {buckets} should all be equal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
