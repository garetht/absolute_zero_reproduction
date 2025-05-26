from dataclasses import dataclass
from enum import Enum

from buffer.base_buff import IOPair

"""
Task type to list of tuples where tuple[0] is the model answer and tuple[1] is the ground truth answer
"""


@dataclass
class Answers:
    abduction: tuple[str, str]
    deduction: tuple[str, str]
    induction: list[IOPair]


@dataclass
class Reward:
    formatting: float
    correctness: float


class Task(Enum):
    DEDUCTION = "DEDUCTION"
    ABDUCTION = "ABDUCTION"
    INDUCTION = "INDUCTION"


class Role(Enum):
    SOLVER = "SOLVER"
    PROPOSER = "PROPOSER"
