from dataclasses import dataclass, field
from enum import Enum
from typing_extensions import TypedDict, Literal

from typing import Generic, Literal, Optional, TypeVar
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

T = TypeVar("T")

@dataclass
class IOPair(Generic[T]):
    input_str: T
    output_str: T

@dataclass
class BaseSample:
    snippet: str
    message: str
    prompt_tokens: list[str]

@dataclass
class FunctionSample(BaseSample):
    snippet: str
    function_io: list[IOPair[str]]
    input_types: Literal["str", "int", "list", "tuple"]
    output_types: Literal["str", "int", "list", "tuple"]
    message: str
    imports: str  # executable string
    prompt_tokens: list[str]

@dataclass
class PrimeSample(BaseSample):
    @property
    def prime(self) -> int : 
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
    x: Optional[int]
    y: Optional[int]
    blank: Literal['x', 'y', 'p']
    task_type: TaskType
    # For reproducible display
    desc: str = field(default='')

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
        match task_type:
            case TaskType.ABDUCTION:
                blank = 'x'
            case TaskType.DEDUCTION:
                blank = 'y'
            case TaskType.INDUCTION:
                blank = 'p'

        return Problem(
            prime=prime_sample.prime,
            x=prime_sample.function_io[0].input_str,
            y=prime_sample.function_io[0].output_str,
            blank=blank,
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
