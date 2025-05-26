
from enum import Enum
"""
Task type to list of tuples where tuple[0] is the model answer and tuple[1] is the ground truth answer
"""
@dataclass
class Answers():
    abduction: list[tuple[str]]
    deduction: list[tuple[str]]
    induction: list[tuple[str]]

@dataclass
class Reward():
    formatting: float
    correctness: float


class Task(Enum):
    DEDUCTION = "DEDUCTION"
    ABDUCTION = "ABDUCTION"
    INDUCTION = "INDUCTION"


class Role(Enum):
    SOLVER = "SOLVER"
    PROPOSER = "PROPOSER"