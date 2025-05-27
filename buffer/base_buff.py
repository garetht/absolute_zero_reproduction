from dataclasses import dataclass

from jaxtyping import Int
import numpy
from torch import Tensor
import torch
from transformers import AutoModelForCausalLM

from buffer.abduction import AbductionBuffer
from buffer.deduction import DeductionBuffer
from buffer.induction import InductionBuffer
from model.args import AZRArgs
from custom_types import TaskType, BaseSample





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


@dataclass
class MegaBuffer:
    seed_buffer: list[BaseSample]
    logprobs: Int[Tensor, "role task batch_size vocab_size"]
    sample_ids: Int[Tensor, "role task batch_size seq_len"]
    # batch_size is the index of the sample in the buffer, same for any role task combo
    buffer: list[BaseSample]
    def get_minibatch(): 
        # looks at the buffer from the current rollout, returns samples indexed using their position in the batch
        pass

    def solver_sample_from_buffer(self, num_to_sample: int) -> list[BaseSample]:
        indices = numpy.random.choice(len(self.seed_buffer), num_to_sample, replace=True)
        return [self.seed_buffer[i] for i in indices]


    def sample_abduction_deduction(self) -> BaseSample:
        pass

    def sample_abduction(self) -> BaseSample:
        pass

    def sample_deduction(self) -> BaseSample:
        pass


def get_samples(
        model: AutoModelForCausalLM,
        prompt: str,
        batch_size: int,
        gen_len: int,
        temperature: float,
        top_k: int,
        prepend_bos: bool,
) -> tuple[Int[Tensor, "batch seq_len"], list[BaseSample]]:
    pass
