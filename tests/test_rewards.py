import pytest
import torch
from unittest.mock import Mock, patch
from custom_types import Role, TaskType, Answer
from model.compute.reward import compute_r_propose, compute_r_total


class TestComputeRPropose:
    
    def test_compute_r_propose_basic(self):
        """Test basic r_propose computation."""
        # Test with average r_solve = 0.5
        r_solve = torch.tensor([0.4, 0.5, 0.6])
        r_propose = compute_r_propose(r_solve)
        
        # Should be 1 - mean(r_solve) = 1 - 0.5 = 0.5
        assert torch.allclose(r_propose, torch.tensor([0.5]))
    
    def test_compute_r_propose_all_zeros(self):
        """Test when all r_solve values are 0 (too hard)."""
        r_solve = torch.zeros(5)
        r_propose = compute_r_propose(r_solve)
        
        # Should be 0 when avg is 0
        assert torch.allclose(r_propose, torch.tensor([0.0]))
    
    def test_compute_r_propose_all_ones(self):
        """Test when all r_solve values are 1 (too easy)."""
        r_solve = torch.ones(5)
        r_propose = compute_r_propose(r_solve)
        
        # Should be 0 when avg is 1
        assert torch.allclose(r_propose, torch.tensor([0.0]))
    
    def test_compute_r_propose_mixed_values(self):
        """Test with various r_solve values."""
        r_solve = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        r_propose = compute_r_propose(r_solve)
        
        # Mean is 0.5, so r_propose should be 0.5
        expected = torch.tensor([0.5])
        assert torch.allclose(r_propose, expected)
    
    @pytest.mark.parametrize("r_solve_values,expected", [
        ([0.2, 0.3, 0.4], 0.7),  # mean=0.3, r_propose=0.7
        ([0.6, 0.7, 0.8], 0.3),  # mean=0.7, r_propose=0.3
        ([0.1, 0.9], 0.5),       # mean=0.5, r_propose=0.5
    ])
    def test_compute_r_propose_parametrized(self, r_solve_values, expected):
        """Test r_propose computation with various inputs."""
        r_solve = torch.tensor(r_solve_values)
        r_propose = compute_r_propose(r_solve)
        
        assert torch.allclose(r_propose, torch.tensor([expected]), atol=1e-6)


class TestComputeRTotal:
    
    @pytest.fixture
    def mock_answer_correct(self):
        """Create a mock Answer with correct formatting."""
        answer = Mock(spec=Answer)
        answer.reward = 1.0
        return answer
    
    @pytest.fixture
    def mock_answer_incorrect(self):
        """Create a mock Answer with incorrect formatting."""
        answer = Mock(spec=Answer)
        answer.reward = -1.0
        return answer
    
    @patch('model.compute.reward.validate_formatting_and_correctness')
    @patch('model.compute.reward.compute_r_propose')
    def test_compute_r_total_proposer_valid_format(self, mock_r_propose, mock_validate):
        """Test r_total for proposer with valid formatting."""
        # Setup mocks
        mock_answer = Mock(reward=1.0)
        mock_validate.return_value = mock_answer
        mock_r_propose.return_value = torch.tensor([0.7])
        
        solver_responses = ["response1", "response2"]
        r_proposer_format = torch.tensor([0.5, 0.8])
        
        result = compute_r_total(
            solver_responses, 
            Role.PROPOSER, 
            TaskType.ABDUCTION,
            r_proposer_format
        )
        
        # Should return r_propose value since r_proposer_format >= 0
        assert torch.allclose(result, torch.tensor([0.7, 0.7]))
    
    @patch('model.compute.reward.validate_formatting_and_correctness')
    def test_compute_r_total_proposer_invalid_format(self, mock_validate):
        """Test r_total for proposer with invalid formatting."""
        # Setup mock
        mock_answer = Mock(reward=0.5)
        mock_validate.return_value = mock_answer
        
        solver_responses = ["response1", "response2"]
        r_proposer_format = torch.tensor([-1.0, -2.0])
        
        result = compute_r_total(
            solver_responses,
            Role.PROPOSER,
            TaskType.DEDUCTION,
            r_proposer_format
        )
        
        # Should return r_proposer_format values since they're < 0
        assert torch.allclose(result, r_proposer_format)
    
    @patch('model.compute.reward.validate_formatting_and_correctness')
    def test_compute_r_total_solver(self, mock_validate):
        """Test r_total for solver role."""
        # Setup mock answers with different rewards
        mock_answers = [Mock(reward=0.8), Mock(reward=0.6), Mock(reward=1.0)]
        mock_validate.side_effect = mock_answers
        
        solver_responses = ["resp1", "resp2", "resp3"]
        r_proposer_format = torch.tensor([0.0, 0.0, 0.0])  # Not used for solver
        
        result = compute_r_total(
            solver_responses,
            Role.SOLVER,
            TaskType.INDUCTION,
            r_proposer_format
        )
        
        # Should return r_solve values directly
        expected = torch.tensor([0.8, 0.6, 1.0])
        assert torch.allclose(result, expected)
    
    @patch('model.compute.reward.validate_formatting_and_correctness')
    @patch('model.compute.reward.compute_r_propose')
    def test_compute_r_total_mixed_proposer_format(self, mock_r_propose, mock_validate):
        """Test with mixed positive and negative r_proposer_format values."""
        # Setup mocks
        mock_answer = Mock(reward=0.5)
        mock_validate.return_value = mock_answer
        mock_r_propose.return_value = torch.tensor([0.4])
        
        solver_responses = ["resp1", "resp2", "resp3", "resp4"]
        r_proposer_format = torch.tensor([0.5, -1.0, 0.0, -2.0])
        
        result = compute_r_total(
            solver_responses,
            Role.PROPOSER,
            TaskType.ABDUCTION,
            r_proposer_format
        )
        
        # Expected: [0.4, -1.0, 0.4, -2.0]
        # (r_propose for non-negative, original for negative)
        expected = torch.tensor([0.4, -1.0, 0.4, -2.0])
        assert torch.allclose(result, expected)
    
    @pytest.mark.parametrize("task_type", list(TaskType))
    @patch('model.compute.reward.validate_formatting_and_correctness')
    def test_compute_r_total_all_task_types(self, mock_validate, task_type):
        """Test that all task types are handled correctly."""
        mock_answer = Mock(reward=0.75)
        mock_validate.return_value = mock_answer
        
        solver_responses = ["response"]
        r_proposer_format = torch.tensor([0.5])
        
        result = compute_r_total(
            solver_responses,
            Role.SOLVER,
            task_type,
            r_proposer_format
        )
        
        # Verify validate was called with correct task_type
        mock_validate.assert_called_with("response", task_type)
        assert torch.allclose(result, torch.tensor([0.75]))
