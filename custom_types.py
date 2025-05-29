from dataclasses import dataclass
from enum import Enum
from typing_extensions import TypedDict, Literal
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from typing import Generic, Optional, TypeVar
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

    @property
    def is_valid(self) -> bool:
        return self.reward >= 0


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

    def __post_init__(self):
        if len(self.function_io) == 0:
            raise ValueError("a created PrimeSample must have at least one IOPair")

    @staticmethod
    def from_problem(
        problem: "Problem", tokenizer: PreTrainedTokenizerFast, max_prompt_length: int
    ) -> "PrimeSample":
        prompt = f"Find the inverse of {problem.x} modulo {problem.prime}."
        prompt_tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_prompt_length,
        )[0]
        return PrimeSample(
            snippet=str(problem.prime),
            function_io=[IOPair(input_str=problem.x, output_str=problem.y)],
            message=str(prompt),
            prompt_tokens=prompt_tokens,
        )


class ProblemResult(TypedDict):
    problem: str
    extracted_answer: Optional[int]
    correct_answer: Optional[int]
    is_correct: bool
    time_seconds: float


@dataclass
class Problem:
    prime: int
    x_list: list[int]
    y_list: list[int]
    task_type: TaskType
    # Store pre-computed prompt strings for both roles to avoid recomputation
    # These are used during learning phase to compute importance ratios with new policy
    prompt_cache: dict[Role, str] = None

    def __post_init__(self):
        if self.prompt_cache is None:
            self.prompt_cache = {}
        # Ensure lists have at least one element
        if not self.x_list or not self.y_list:
            raise ValueError("x_list and y_list must have at least one element")

    @property
    def x(self) -> int:
        """Get first x value for backward compatibility with abduction/deduction"""
        return self.x_list[0]
    
    @property  
    def y(self) -> int:
        """Get first y value for backward compatibility with abduction/deduction"""
        return self.y_list[0]

    @property
    def blank(self) -> Literal["x", "y", "p"]:
        match self.task_type:
            case TaskType.ABDUCTION:
                blank = "x"
            case TaskType.DEDUCTION:
                blank = "y"
            case TaskType.INDUCTION:
                blank = "p"

        return blank

    def get_prompt(self, role: Role) -> str:
        """Get cached prompt for role, or generate and cache if not exists"""
        if role not in self.prompt_cache:
            from utils.string_formatting import create_proposer_prompt, create_solver_prompt
            if role == Role.PROPOSER:
                self.prompt_cache[role] = create_proposer_prompt(self)
            else:  # Role.SOLVER
                self.prompt_cache[role] = create_solver_prompt(self)
        return self.prompt_cache[role]

    def __repr__(self) -> str:
        """Return a nicely formatted string representation of the problem."""
        if self.blank == "x":
            return (
                f"Find x such that x * {self.y} ≡ 1 (mod {self.prime}) [x = {self.x}]"
            )
        elif self.blank == "y":
            return (
                f"Find y such that {self.x} * y ≡ 1 (mod {self.prime}) [y = {self.y}]"
            )
        else:
            return (
                f"Find a p such that {self.x} * {self.y} ≡ 1 (mod p) [p = {self.prime}]"
            )

    @staticmethod
    def from_prime_sample(prime_sample: PrimeSample, task_type: TaskType) -> "Problem":
        return Problem(
            prime=prime_sample.prime,
            x_list=[io.input_str for io in prime_sample.function_io],
            y_list=[io.output_str for io in prime_sample.function_io],
            task_type=task_type,
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
    logprobs: Float[Tensor, "role task minibatch_size max_response_length"]
    attention_masks: Int[Tensor, "role task minibatch_size seq_len"]
    rewards: Float[Tensor, "role task minibatch_size"]
