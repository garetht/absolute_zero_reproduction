from dataclasses import dataclass
from typing_extensions import Literal

from jaxtyping import Float, Int
from torch import Tensor
import torch
from transformers import AutoModelForCausalLM

from buffer.abduction import AbductionBuffer
from buffer.deduction import DeductionBuffer
from buffer.induction import InductionBuffer
from model.args import AZRArgs


@dataclass
class IOPair:
    input_str: str
    output_str: str


@dataclass
class Snippet:
    snippet: str

@dataclass
class Sample:
    snippet: Snippet
    function_io: list[IOPair]
    input_types: Literal["str", "int", "list", "tuple"]
    output_types: Literal["str", "int", "list", "tuple"]
    message: str
    imports: str  # executable string
    prompt_tokens: list[str]
    sample_ids: Int[Tensor, "seq_len"]


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

    def sample(self):
        pass


@dataclass
class MegaBuffer:
    seed_buffer: BaseBuffer
    induction_buffer: InductionBuffer
    abduction_buffer: AbductionBuffer
    deduction_buffer: DeductionBuffer

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
