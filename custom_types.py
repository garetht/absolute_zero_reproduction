from dataclasses import dataclass
from enum import Enum
from typing import Generic, Literal, Optional, TypeVar
from jaxtyping import Int, Float
from torch import Tensor


"""
Task type to list of tuples where tuple[0] is the model answer and tuple[1] is the ground truth answer
"""


@dataclass
class Answer:
    input: Optional[str]
    program: str
    output: Optional[str]
    reward: float


class TaskType(Enum):
    DEDUCTION = "DEDUCTION"
    ABDUCTION = "ABDUCTION"
    INDUCTION = "INDUCTION"


class Role(Enum):
    SOLVER = "SOLVER"
    PROPOSER = "PROPOSER"

T = TypeVar("T")

@dataclass
class IOPair(Generic[T]):
    input_str: T
    output_str: T


@dataclass
class BaseSample:
    snippet: str
    message: str
    prompt_tokens: Int[Tensor, "max_prompt_length"]

@dataclass
class FunctionSample(BaseSample):
    snippet: str
    function_io: list[IOPair[str]]
    input_types: Literal["str", "int", "list", "tuple"]
    output_types: Literal["str", "int", "list", "tuple"]
    message: str
    imports: str  # executable string

@dataclass
class PrimeSample(BaseSample):
    @property
    def prime(self) -> int : 
        return int(self.snippet)
    function_io: list[IOPair[int]]

@dataclass
class MiniBatch:
    samples: list[BaseSample]
    sample_ids: Int[Tensor, "role task minibatch_size seq_len"]
    logprobs: Float[Tensor, "role task minibatch_size seq_len vocab_size"]

