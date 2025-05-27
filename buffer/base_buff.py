from dataclasses import dataclass

from jaxtyping import Int
from torch import Tensor
import torch
from transformers import AutoModelForCausalLM

from buffer.abduction import AbductionBuffer
from buffer.deduction import DeductionBuffer
from buffer.induction import InductionBuffer
from model.args import AZRArgs
from custom_types import TaskType, Sample





class BaseBuffer:
    """
    Base class for the buffer objects (also class for the seed buffer).
    """

    def __init__(
            self,
            args: AZRArgs,
            samples: list[Sample],
    ):
        self.args = args
        self.samples = samples

    def sample_ids(self):
        return torch.cat([s.sample_ids for s in self.samples])

    def sample(self) -> Sample:
        pass

    def extend(self, sample: Sample):
        pass


@dataclass
class MegaBuffer:
    seed_buffer: BaseBuffer
    induction_buffer: InductionBuffer
    abduction_buffer: AbductionBuffer
    deduction_buffer: DeductionBuffer
    logprobs: Int[Tensor, "role task batch_size vocab_size"]
    sample_ids: Int[Tensor, "role task batch_size seq_len"]
    # TODO
    def get_minibatch(): 

    def sample_from_buffer(self, buffer: TaskType, num_to_sample: int) -> list[Sample]:
        samples = []
        for i in range(num_to_sample):
            if buffer == TaskType.INDUCTION:
                sample = self.induction_buffer.sample()
            elif buffer == TaskType.ABDUCTION:
                sample = self.abduction_buffer.sample()
            elif buffer == TaskType.DEDUCTION:
                sample = self.deduction_buffer.sample()
            else:
                sample = self.seed_buffer.sample()

            if sample is not None:
                samples.append(sample)

        return samples

    def sample_abduction_deduction(self) -> Sample:
        pass

    def sample_abduction(self) -> Sample:
        pass

    def sample_deduction(self) -> Sample:
        pass


def get_samples(
        model: AutoModelForCausalLM,
        prompt: str,
        batch_size: int,
        gen_len: int,
        temperature: float,
        top_k: int,
        prepend_bos: bool,
) -> tuple[Int[Tensor, "batch seq_len"], list[Sample]]:
    pass
