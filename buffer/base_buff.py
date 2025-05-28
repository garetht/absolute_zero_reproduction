from jaxtyping import Int
import numpy
from torch import Tensor
import torch

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from model.args import AZRArgs
from model.eval.prime_inversion import get_problems, PRIMES
from custom_types import MiniBatch, Role, TaskType, BaseSample, PrimeSample


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


class MegaBuffer:
    def __init__(
        self,
        args: AZRArgs,
        logprobs: Int[Tensor, "role task batch_size max_response_len vocab_size"],
        sample_ids: Int[Tensor, "role task batch_size max_response_len"],
        attention_masks: Int[Tensor, "role task batch_size max_response_len"] = None,
    ):
        self.args = args
        self.seed_buffer: list[BaseSample] = []
        self.logprobs = logprobs
        self.sample_ids = sample_ids
        # Initialize attention masks if not provided
        if attention_masks is None:
            self.attention_masks = torch.ones_like(sample_ids)
        else:
            self.attention_masks = attention_masks
        # batch_size is the index of the sample in the buffer, same for any role task combo
        self.buffer: list[BaseSample] = []

    def get_minibatches(self) -> list[MiniBatch]:
        # looks at the buffer from the current rollout, returns samples indexed using their position in the batch
        out = []
        for indices in torch.randperm(self.args.batch_size).reshape(
            self.args.n_minibatches, -1
        ):
            minibatch_samples = [self.buffer[i] for i in indices]
            minibatch_sample_ids = self.sample_ids[:, :, indices]
            assert minibatch_sample_ids.shape == (
                len(Role),
                len(TaskType),
                self.args.minibatch_size,
                self.args.max_response_length,
            ), (
                "Sample ids should have shape  (len(Role), len(TaskType), args.minibatch_size, args.max_response_length)"
            )
            minibatch_logprobs = self.logprobs[:, :, indices]
            assert minibatch_logprobs.shape == (
                len(Role),
                len(TaskType),
                self.args.minibatch_size,
                self.args.max_response_length,
                self.args.d_vocab,
            ), (
                "Logprobs should have shape  (len(Role), len(TaskType), args.minibatch_size, args.max_response_length, args.vocab_size)"
            )
            minibatch_attention_masks = self.attention_masks[:, :, indices]
            out.append(
                MiniBatch(
                    samples=minibatch_samples,
                    sample_ids=minibatch_sample_ids,
                    logprobs=minibatch_logprobs,
                    attention_masks=minibatch_attention_masks,
                )
            )
        return out

    def reset(self) -> None:
        self.seed_buffer.extend(self.buffer)
        self.buffer = []
        self.logprobs = torch.zeros_like(self.logprobs, device=self.logprobs.device)
        self.sample_ids = torch.zeros_like(self.sample_ids, device=self.sample_ids.device)
        self.attention_masks = torch.zeros_like(self.attention_masks, device=self.attention_masks.device)

    @property
    def combined_buffer(self):
        return self.seed_buffer + self.buffer

    def sample_from_buffer(self, num_to_sample: int) -> list[BaseSample]:
        indices = numpy.random.choice(
            len(self.combined_buffer), num_to_sample, replace=True
        )
        return [self.combined_buffer[i] for i in indices]

    def initialize_seed_buffer(
        self, tokenizer: PreTrainedTokenizerFast, num_samples: int = 1
    ) -> None:
        """
        Initialize seed buffer with k (default 1) samples
        """
        problems = get_problems(n=num_samples, primes=PRIMES, seed=self.args.seed)
        self.seed_buffer.extend(
            [
                PrimeSample.from_problem(p, tokenizer, self.args.max_prompt_length)
                for p in problems
            ]
        )
