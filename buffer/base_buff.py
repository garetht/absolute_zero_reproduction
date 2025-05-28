
from jaxtyping import Int
import numpy
from torch import Tensor
import torch
from transformers import AutoModelForCausalLM

from model.args import AZRArgs
from custom_types import MiniBatch, Role, TaskType, BaseSample


class BaseBuffer:
    """
    Base class for the buffer objects (also class for the seed buffer).
    """

    def __init__(
            self,
            args: AZRArgs,
            samples: list[BaseSample],
    ):
        self.args = args
        self.samples = samples

    def sample_ids(self):
        return torch.cat([s.sample_ids for s in self.samples])

    def sample(self) -> BaseSample:
        pass

    def extend(self, sample: BaseSample):
        pass


class MegaBuffer:
    def __init__(
        self,
        seed_buffer: list[BaseSample],
        logprobs: Int[Tensor, "role task batch_size max_response_len vocab_size"],
        sample_ids: Int[Tensor, "role task batch_size max_response_len"],
        buffer: list[BaseSample]
    ):
        self.seed_buffer = seed_buffer
        self.logprobs = logprobs
        self.sample_ids = sample_ids
        # batch_size is the index of the sample in the buffer, same for any role task combo
        self.buffer = buffer

    def get_minibatches(self, args:AZRArgs) -> list[MiniBatch]:
        # looks at the buffer from the current rollout, returns samples indexed using their position in the batch
        out = []
        for indices in torch.randperm(args.batch_size).reshape(args.n_minibatches, -1):
            minibatch_samples = [self.buffer[i] for i in indices]
            minibatch_sample_ids = self.sample_ids[:,:,indices]
            assert minibatch_sample_ids.shape == (len(Role), len(TaskType), args.minibatch_size, args.max_response_length), "Sample ids should have shape  (len(Role), len(TaskType), args.minibatch_size, args.max_response_length)"
            minibatch_logprobs = self.logprobs[:,:,indices]
            assert minibatch_logprobs.shape == (len(Role), len(TaskType), args.minibatch_size, args.max_response_length, args.d_vocab), "Logprobs should have shape  (len(Role), len(TaskType), args.minibatch_size, args.max_response_length, args.vocab_size)"
            out.append(MiniBatch(
                samples=minibatch_samples,
                sample_ids=minibatch_sample_ids,
                logprobs=minibatch_logprobs
            ))
        return out

    def reset(self) -> None:
        self.seed_buffer.extend(self.buffer)
        self.buffer = []
        self.logprobs = torch.zeros_like(self.logprobs)
        self.sample_ids = torch.zeros_like(self.sample_ids)

    @property
    def combined_buffer(self):
        return self.seed_buffer + self.buffer

    def sample_from_buffer(self, num_to_sample: int) -> list[BaseSample]:
        indices = numpy.random.choice(len(self.combined_buffer), num_to_sample, replace=True)
        return [self.combined_buffer[i] for i in indices]
