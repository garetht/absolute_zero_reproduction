import pytest
import torch
from model.args import AZRArgs
from model.compute.advantages import compute_advantages


class TestComputeAdvantages:
    
    @pytest.fixture
    def args(self):
        """Create test AZRArgs with minimal required fields."""
        args = AZRArgs()
        args.eps = 1e-8  # Add eps attribute for normalization
        return args
    
    def test_compute_advantages_basic_normalization(self, args):
        """Test that advantages are properly normalized (mean=0, std=1)."""
        # Create test rewards tensor: (role=2, task=3, minibatch_size=4)
        rewards = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0]],
            [[2.0, 4.0, 6.0, 8.0],
             [1.0, 3.0, 5.0, 7.0],
             [10.0, 20.0, 30.0, 40.0]]
        ])
        
        advantages = compute_advantages(args, rewards)
        
        # Check that mean is approximately 0 for each role-task combination
        for role in range(2):
            for task in range(3):
                mean = advantages[role, task].mean().item()
                assert abs(mean) < 1e-6, f"Mean should be ~0, got {mean} for role={role}, task={task}"
                
                # Check that std is approximately 1
                std = advantages[role, task].std().item()
                assert abs(std - 1.0) < 1e-6, f"Std should be ~1, got {std} for role={role}, task={task}"
    
    def test_compute_advantages_single_value(self, args):
        """Test edge case with single value per role-task (std=0)."""
        # Create rewards with single minibatch_size
        rewards = torch.tensor([
            [[5.0],
             [10.0],
             [15.0]],
            [[20.0],
             [25.0],
             [30.0]]
        ])
        
        advantages = compute_advantages(args, rewards)
        
        # When std=0, the normalization should handle it gracefully with eps
        assert torch.all(advantages == 0.0), "Single values should result in 0 advantages"
    
    def test_compute_advantages_identical_rewards(self, args):
        """Test when all rewards are identical (std=0)."""
        # All rewards are the same value
        rewards = torch.ones(2, 3, 5) * 7.0
        
        advantages = compute_advantages(args, rewards)
        
        # All advantages should be 0 when rewards are identical
        assert torch.all(advantages == 0.0), "Identical rewards should result in 0 advantages"
    
    def test_compute_advantages_preserves_shape(self, args):
        """Test that output shape matches input shape."""
        rewards = torch.randn(2, 3, 8)  # role=2, task=3, minibatch_size=8
        
        advantages = compute_advantages(args, rewards)
        
        assert advantages.shape == rewards.shape, f"Shape mismatch: {advantages.shape} != {rewards.shape}"
    
    def test_compute_advantages_different_scales(self, args):
        """Test normalization works correctly for different scales of rewards."""
        # Test with very small rewards
        small_rewards = torch.tensor([
            [[0.001, 0.002, 0.003, 0.004]],
            [[0.01, 0.02, 0.03, 0.04]]
        ])
        
        small_advantages = compute_advantages(args, small_rewards)
        
        # Test with very large rewards
        large_rewards = torch.tensor([
            [[1000, 2000, 3000, 4000]],
            [[10000, 20000, 30000, 40000]]
        ])
        
        large_advantages = compute_advantages(args, large_rewards)
        
        # Both should be normalized to same scale
        assert torch.allclose(small_advantages[0, 0].std(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(large_advantages[0, 0].std(), torch.tensor(1.0), atol=1e-6)
    
    @pytest.mark.parametrize("role_dim,task_dim,batch_dim", [
        (1, 1, 10),
        (2, 3, 5),
        (5, 2, 20),
    ])
    def test_compute_advantages_various_dimensions(self, args, role_dim, task_dim, batch_dim):
        """Test with various tensor dimensions."""
        rewards = torch.randn(role_dim, task_dim, batch_dim)
        
        advantages = compute_advantages(args, rewards)
        
        assert advantages.shape == (role_dim, task_dim, batch_dim)
        
        # Verify normalization for each role-task
        for role in range(role_dim):
            for task in range(task_dim):
                mean = advantages[role, task].mean().item()
                std = advantages[role, task].std().item()
                assert abs(mean) < 1e-5, f"Mean not ~0: {mean}"
                assert abs(std - 1.0) < 1e-5, f"Std not ~1: {std}"
