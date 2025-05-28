from dataclasses import dataclass, field
from enum import Enum
from typing_extensions import TypedDict, Literal

from typing import Generic, Literal, Optional, TypeVar
from jaxtyping import Int, Float
from torch import Tensor

"""
Task type to list of tuples where tuple[0] is the model answer and tuple[1] is the ground truth answer
"""


@dataclass
class Answer:
    input: Optional[int]
    program: Optional[int]
    output: Optional[int]
    reward: float


class TaskType(Enum):
    DEDUCTION = 0
    ABDUCTION = 1
    INDUCTION = 2


class Role(Enum):
    SOLVER = 0
    PROPOSER = 1


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
    def prime(self) -> int:
        return int(self.snippet)

    function_io: list[IOPair[int]]


class ProblemResult(TypedDict):
    problem: str
    extracted_answer: Optional[int]
    correct_answer: Optional[int]
    is_correct: bool
    time_seconds: float


@dataclass
class Problem:
    prime: int
    x: int
    y: int
    task_type: TaskType

    @property
    def blank(self) -> Literal['x', 'y', 'p']:
        match self.task_type:
            case TaskType.ABDUCTION:
                blank = 'x'
            case TaskType.DEDUCTION:
                blank = 'y'
            case TaskType.INDUCTION:
                blank = 'p'

        return blank

    def __repr__(self) -> str:
        """Return a nicely formatted string representation of the problem."""
        if self.blank == 'x':
            return f"Find x such that x * {self.y} ≡ 1 (mod {self.prime}) [x = {self.x}]"
        elif self.blank == 'y':
            return f"Find y such that {self.x} * y ≡ 1 (mod {self.prime}) [y = {self.y}]"
        else:
            return f"Find a p such that {self.x} * {self.y} ≡ 1 (mod p) [p = {self.prime}]"

    @staticmethod
    def from_prime_sample(prime_sample: PrimeSample, task_type: TaskType) -> 'Problem':
        return Problem(
            prime=prime_sample.prime,
            x=prime_sample.function_io[0].input_str,
            y=prime_sample.function_io[0].output_str,
            task_type=task_type
        )


class EvaluationResults(TypedDict):
    model: str
    correct: int
    no_response: int
    total: int
    timestamp: str
    problem_results: list[ProblemResult]
    total_eval_time_seconds: float
    accuracy: float


@dataclass
class MiniBatch:
    samples: list[BaseSample]
    sample_ids: Int[Tensor, "role task minibatch_size seq_len"]
    logprobs: Float[Tensor, "role task minibatch_size max_response_length vocab_size"]

