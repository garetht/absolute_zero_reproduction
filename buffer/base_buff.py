from dataclasses import dataclass
from typing import Literal
from jaxtyping import Float, Int

from transformers import AutoModelForCausalLM

from torch import Tensor


@dataclass
class IOPair:
    input_str: str
    output_str: str


@dataclass
class Sample:
    snippet: str
    function_io: list[IOPair]
    input_types: Literal["str", "int", "list", "tuple"]
    output_types: Literal["str", "int", "list", "tuple"]
    message: str
    imports: str  # executable string
    prompt_tokens: list[str]


class BaseBuffer:
    """
    Base class for the buffer objects (also class for the seed buffer).
    """

    def __init__(
        self,
        args: AZRArgs,
        sample_ids: Float[Tensor, "batch_size seq_len"],
        logprobs: Float[Tensor, "batch_size seq_len"],
        values: Float[Tensor, "batch_size seq_len"],
        ref_logits: Float[Tensor, "batch_size seq_len"],
    ):
        self.args = args
        self.sample_ids = sample_ids
        self.logprobs = logprobs
        self.values = values
        self.ref_logits = ref_logits


def get_samples(
    model: AutoModelForCausalLM,
    prompt: str,
    batch_size: int,
    gen_len: int,
    temperature: float,
    top_k: int,
    prepend_bos: bool,
) -> tuple[Int[Tensor, "batch seq_len"], list[Sample]]:
    return
