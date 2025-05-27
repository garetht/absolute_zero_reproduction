from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional
from jaxtyping import Int
from torch import Tensor


"""
Task type to list of tuples where tuple[0] is the model answer and tuple[1] is the ground truth answer
"""


@dataclass
class Answer:
    input: Optional[str]
    program: str
    output: Optional[str]


@dataclass
class Reward:
    formatting: float
    correctness: float


class TaskType(Enum):
    DEDUCTION = "DEDUCTION"
    ABDUCTION = "ABDUCTION"
    INDUCTION = "INDUCTION"


class Role(Enum):
    SOLVER = "SOLVER"
    PROPOSER = "PROPOSER"

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
    sample_ids: Int[Tensor, "seq_len"]