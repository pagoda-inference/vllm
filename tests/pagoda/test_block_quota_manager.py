# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda KV cache block quota enforcement."""

import pytest

from vllm.pagoda.block_quota_manager import BlockQuotaManager


class TestBlockQuotaManager:
    """Tests for BlockQuotaManager."""

    def setup_method(self):
        """Create a fresh manager for each test."""
        self.manager = BlockQuotaManager()

    def test_initial_usage_is_zero(self):
        """Test that initial usage for any tenant is zero."""
        assert self.manager.get_usage("tenant-1") == 0
        assert self.manager.get_usage("tenant-2") == 0

    def test_can_allocate_within_quota(self):
        """Test allocation within quota is allowed."""
        assert self.manager.can_allocate("tenant-1", num_blocks=10, quota=100)
        assert self.manager.can_allocate("tenant-1", num_blocks=100, quota=100)

    def test_can_allocate_exceeds_quota(self):
        """Test allocation exceeding quota is rejected."""
        assert not self.manager.can_allocate(
            "tenant-1", num_blocks=101, quota=100
        )

    def test_can_allocate_with_existing_usage(self):
        """Test allocation check accounts for existing usage."""
        self.manager.allocate("tenant-1", num_blocks=80)

        # 80 + 20 = 100 <= 100, should be allowed
        assert self.manager.can_allocate("tenant-1", num_blocks=20, quota=100)

        # 80 + 21 = 101 > 100, should be rejected
        assert not self.manager.can_allocate(
            "tenant-1", num_blocks=21, quota=100
        )

    def test_allocate_tracks_usage(self):
        """Test that allocate correctly tracks block usage."""
        self.manager.allocate("tenant-1", num_blocks=10)
        assert self.manager.get_usage("tenant-1") == 10

        self.manager.allocate("tenant-1", num_blocks=20)
        assert self.manager.get_usage("tenant-1") == 30

    def test_release_reduces_usage(self):
        """Test that release correctly reduces block usage."""
        self.manager.allocate("tenant-1", num_blocks=50)
        self.manager.release("tenant-1", num_blocks=20)
        assert self.manager.get_usage("tenant-1") == 30

    def test_release_below_zero_resets_to_zero(self):
        """Test that releasing more than allocated resets to zero."""
        self.manager.allocate("tenant-1", num_blocks=10)
        self.manager.release("tenant-1", num_blocks=20)
        assert self.manager.get_usage("tenant-1") == 0

    def test_tenant_isolation(self):
        """Test that tenants have independent quotas."""
        self.manager.allocate("tenant-1", num_blocks=50)
        self.manager.allocate("tenant-2", num_blocks=30)

        assert self.manager.get_usage("tenant-1") == 50
        assert self.manager.get_usage("tenant-2") == 30

        # Tenant-1 at 50, can't allocate 60 more with quota 100
        assert not self.manager.can_allocate(
            "tenant-1", num_blocks=60, quota=100
        )
        # Tenant-2 at 30, can allocate 60 more with quota 100
        assert self.manager.can_allocate("tenant-2", num_blocks=60, quota=100)

    def test_get_all_usage(self):
        """Test get_all_usage returns all tenant usage."""
        self.manager.allocate("tenant-1", num_blocks=10)
        self.manager.allocate("tenant-2", num_blocks=20)
        self.manager.allocate("tenant-3", num_blocks=30)

        usage = self.manager.get_all_usage()
        assert usage == {
            "tenant-1": 10,
            "tenant-2": 20,
            "tenant-3": 30,
        }

    def test_get_all_usage_empty(self):
        """Test get_all_usage with no tenants."""
        usage = self.manager.get_all_usage()
        assert usage == {}

    def test_allocate_release_cycle(self):
        """Test full allocate-release cycle."""
        # Allocate blocks
        self.manager.allocate("tenant-1", num_blocks=50)
        assert not self.manager.can_allocate(
            "tenant-1", num_blocks=60, quota=100
        )

        # Release blocks
        self.manager.release("tenant-1", num_blocks=50)
        assert self.manager.get_usage("tenant-1") == 0

        # Should be able to allocate again
        assert self.manager.can_allocate("tenant-1", num_blocks=60, quota=100)

    def test_zero_quota(self):
        """Test behavior with zero quota."""
        assert not self.manager.can_allocate("tenant-1", num_blocks=1, quota=0)
        # Zero blocks should still be allowed
        assert self.manager.can_allocate("tenant-1", num_blocks=0, quota=0)

    def test_multiple_small_allocations(self):
        """Test many small allocations accumulate correctly."""
        for _ in range(100):
            self.manager.allocate("tenant-1", num_blocks=1)
        assert self.manager.get_usage("tenant-1") == 100

        assert not self.manager.can_allocate(
            "tenant-1", num_blocks=1, quota=100
        )

    def test_try_allocate_success(self):
        """Test try_allocate succeeds within quota and updates usage."""
        assert self.manager.try_allocate("tenant-1", num_blocks=10, quota=100)
        assert self.manager.get_usage("tenant-1") == 10

    def test_try_allocate_failure(self):
        """Test try_allocate fails when exceeding quota without side effects."""
        self.manager.allocate("tenant-1", num_blocks=90)
        assert not self.manager.try_allocate(
            "tenant-1", num_blocks=20, quota=100
        )
        # Usage unchanged
        assert self.manager.get_usage("tenant-1") == 90

    def test_try_allocate_exact_quota(self):
        """Test try_allocate at exact quota boundary."""
        assert self.manager.try_allocate("tenant-1", num_blocks=100, quota=100)
        assert self.manager.get_usage("tenant-1") == 100
        # One more should fail
        assert not self.manager.try_allocate(
            "tenant-1", num_blocks=1, quota=100
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
