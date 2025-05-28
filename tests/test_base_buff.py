import torch
from buffer.base_buff import MegaBuffer
from custom_types import Problem, TaskType
from model.args import AZRArgs


class TestMegaBufferReset:
    def test_reset_clears_buffer_list(self):
        args = AZRArgs(d_vocab=100)
        logprobs = torch.ones(2, 3, 4, 5, 100)
        sample_ids = torch.ones(2, 3, 4, 5, dtype=torch.long)
        
        mega_buffer = MegaBuffer(args, logprobs, sample_ids)
        # Add some problems to buffer
        mega_buffer.buffer = [Problem(prime=7, x_list=[3], y_list=[5], task_type=TaskType.ABDUCTION),
                             Problem(prime=11, x_list=[2], y_list=[6], task_type=TaskType.DEDUCTION),
                             Problem(prime=13, x_list=[4], y_list=[8], task_type=TaskType.INDUCTION)]
        mega_buffer.reset()

        assert mega_buffer.buffer == []

    def test_reset_zeros_logprobs_tensor(self):
        args = AZRArgs(d_vocab=100)
        logprobs = torch.ones(2, 3, 4, 5, 100)
        sample_ids = torch.ones(2, 3, 4, 5, dtype=torch.long)
        
        mega_buffer = MegaBuffer(args, logprobs, sample_ids)
        mega_buffer.reset()

        expected_logprobs = torch.zeros(logprobs.shape)
        assert torch.equal(mega_buffer.logprobs, expected_logprobs)

    def test_reset_zeros_sample_ids_tensor(self):
        args = AZRArgs(d_vocab=100)
        logprobs = torch.ones(2, 3, 4, 5, 100)
        sample_ids = torch.ones(2, 3, 4, 5, dtype=torch.long)

        mega_buffer = MegaBuffer(args, logprobs, sample_ids)
        mega_buffer.reset()

        expected_sample_ids = torch.zeros(sample_ids.shape)
        assert torch.equal(mega_buffer.sample_ids, expected_sample_ids)

    def test_reset_preserves_original_tensozr_shapes(self):
        args = AZRArgs(d_vocab=100)
        original_logprobs_shape = (2, 3, 4, 5, 100)
        original_sample_ids_shape = (2, 3, 4, 5)
        logprobs = torch.ones(original_logprobs_shape)
        sample_ids = torch.ones(original_sample_ids_shape, dtype=torch.long)
        
        mega_buffer = MegaBuffer(args, logprobs, sample_ids)
        mega_buffer.buffer = [Problem(prime=7, x_list=[3], y_list=[5], task_type=TaskType.ABDUCTION)]
        mega_buffer.reset()

        assert mega_buffer.logprobs.shape == original_logprobs_shape
        assert mega_buffer.sample_ids.shape == original_sample_ids_shape

    def test_reset_multiple_calls(self):
        args = AZRArgs(d_vocab=10)
        logprobs = torch.ones(1, 1, 1, 1, 10)
        sample_ids = torch.ones(1, 1, 1, 1, dtype=torch.long)
        
        mega_buffer = MegaBuffer(args, logprobs, sample_ids)
        mega_buffer.buffer = [Problem(prime=7, x_list=[3], y_list=[5], task_type=TaskType.ABDUCTION),
                             Problem(prime=11, x_list=[2], y_list=[6], task_type=TaskType.DEDUCTION)]
        mega_buffer.reset()
        mega_buffer.reset()

        assert mega_buffer.buffer == []
        assert torch.equal(mega_buffer.logprobs, torch.zeros(logprobs.shape))
        assert torch.equal(mega_buffer.sample_ids, torch.zeros(sample_ids.shape))

    def test_reset_with_empty_buffer(self):
        args = AZRArgs(d_vocab=5)
        logprobs = torch.ones(1, 1, 1, 1, 5)
        sample_ids = torch.ones(1, 1, 1, 1, dtype=torch.long)
        
        mega_buffer = MegaBuffer(args, logprobs, sample_ids)
        mega_buffer.reset()

        assert mega_buffer.buffer == []
        assert torch.equal(mega_buffer.logprobs, torch.zeros(logprobs.shape))
        assert torch.equal(mega_buffer.sample_ids, torch.zeros(sample_ids.shape))