import torch
from buffer.base_buff import MegaBuffer
from custom_types import BaseSample


class TestMegaBufferReset:
    def test_reset_clears_buffer_list(self):
        seed_buffer = []
        logprobs = torch.ones(2, 3, 4, 5, 100)
        sample_ids = torch.ones(2, 3, 4, 5, dtype=torch.long)
        buffer = [BaseSample(snippet="", message="", prompt_tokens=torch.Tensor([])),
                  BaseSample(snippet="", message="", prompt_tokens=torch.Tensor([])),
                  BaseSample(snippet="", message="", prompt_tokens=torch.Tensor([]))]

        mega_buffer = MegaBuffer(seed_buffer, logprobs, sample_ids, buffer)
        mega_buffer.reset()

        assert mega_buffer.buffer == []

    def test_reset_zeros_logprobs_tensor(self):
        seed_buffer = []
        logprobs = torch.ones(2, 3, 4, 5, 100)
        sample_ids = torch.ones(2, 3, 4, 5, dtype=torch.long)
        buffer = []

        mega_buffer = MegaBuffer(seed_buffer, logprobs, sample_ids, buffer)
        mega_buffer.reset()

        expected_logprobs = torch.zeros(logprobs.shape)
        assert torch.equal(mega_buffer.logprobs, expected_logprobs)

    def test_reset_zeros_sample_ids_tensor(self):
        seed_buffer = []
        logprobs = torch.ones(2, 3, 4, 5, 100)
        sample_ids = torch.ones(2, 3, 4, 5, dtype=torch.long)
        buffer = []

        mega_buffer = MegaBuffer(seed_buffer, logprobs, sample_ids, buffer)
        mega_buffer.reset()

        expected_sample_ids = torch.zeros(sample_ids.shape)
        assert torch.equal(mega_buffer.sample_ids, expected_sample_ids)

    def test_reset_preserves_original_tensozr_shapes(self):
        seed_buffer = []
        original_logprobs_shape = (2, 3, 4, 5, 100)
        original_sample_ids_shape = (2, 3, 4, 5)
        logprobs = torch.ones(original_logprobs_shape)
        sample_ids = torch.ones(original_sample_ids_shape, dtype=torch.long)
        buffer = [BaseSample(snippet="", message="", prompt_tokens=torch.Tensor([]))]

        mega_buffer = MegaBuffer(seed_buffer, logprobs, sample_ids, buffer)
        mega_buffer.reset()

        assert mega_buffer.logprobs.shape == original_logprobs_shape
        assert mega_buffer.sample_ids.shape == original_sample_ids_shape

    def test_reset_multiple_calls(self):
        seed_buffer = []
        logprobs = torch.ones(1, 1, 1, 1, 10)
        sample_ids = torch.ones(1, 1, 1, 1, dtype=torch.long)
        buffer = [BaseSample(snippet="", message="", prompt_tokens=torch.Tensor([])),
                  BaseSample(snippet="", message="", prompt_tokens=torch.Tensor([]))]

        mega_buffer = MegaBuffer(seed_buffer, logprobs, sample_ids, buffer)
        mega_buffer.reset()
        mega_buffer.reset()

        assert mega_buffer.buffer == []
        assert torch.equal(mega_buffer.logprobs, torch.zeros(logprobs.shape))
        assert torch.equal(mega_buffer.sample_ids, torch.zeros(sample_ids.shape))

    def test_reset_with_empty_buffer(self):
        seed_buffer = []
        logprobs = torch.ones(1, 1, 1, 1, 5)
        sample_ids = torch.ones(1, 1, 1, 1, dtype=torch.long)
        buffer = []

        mega_buffer = MegaBuffer(seed_buffer, logprobs, sample_ids, buffer)
        mega_buffer.reset()

        assert mega_buffer.buffer == []
        assert torch.equal(mega_buffer.logprobs, torch.zeros(logprobs.shape))
        assert torch.equal(mega_buffer.sample_ids, torch.zeros(sample_ids.shape))
