# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output length predictor for Pagoda scheduling.

Provides a simple heuristic-based predictor that estimates output token
counts from input length. Used by the scheduler to assign length buckets
for more efficient batch scheduling.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

# Length bucket boundaries (in predicted output tokens)
LENGTH_BUCKET_BOUNDARIES = [64, 256, 1024, 4096]


def predict_output_length(input_token_count: int) -> int:
    """Predict output token count from input length using a simple heuristic.

    Uses a basic ratio-based estimate: shorter inputs tend to produce
    proportionally longer outputs, while longer inputs produce shorter
    relative outputs.

    Args:
        input_token_count: Number of tokens in the input prompt

    Returns:
        Estimated number of output tokens
    """
    if input_token_count <= 0:
        return 64

    if input_token_count < 128:
        return min(input_token_count * 4, 4096)
    elif input_token_count < 512:
        return min(input_token_count * 2, 4096)
    elif input_token_count < 2048:
        return min(input_token_count, 4096)
    else:
        return min(input_token_count // 2, 4096)


def assign_length_bucket(predicted_output_len: int) -> int:
    """Assign a length bucket index based on predicted output length.

    Bucket indices correspond to LENGTH_BUCKET_BOUNDARIES:
    - Bucket 0: predicted_output_len <= 64
    - Bucket 1: predicted_output_len <= 256
    - Bucket 2: predicted_output_len <= 1024
    - Bucket 3: predicted_output_len <= 4096
    - Bucket 4: predicted_output_len > 4096

    Args:
        predicted_output_len: Estimated number of output tokens

    Returns:
        Bucket index (0 to len(LENGTH_BUCKET_BOUNDARIES))
    """
    for i, boundary in enumerate(LENGTH_BUCKET_BOUNDARIES):
        if predicted_output_len <= boundary:
            return i
    return len(LENGTH_BUCKET_BOUNDARIES)
