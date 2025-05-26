from dataclasses import dataclass
from enum import Enum
from typing import Optional

from buffer.base_buff import IOPair

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
