
from torch import Tensor
from jaxtyping import Float

from buffer.base_buff import Sample
from custom_types import Answer
"""
Computes the correctness of the model's answer baed on value equality in python
"""
def compute_r_solve(answers: list[Answer], samples: list[Sample]) -> Float[Tensor, "batch_size"]:
    pass

"""
Computes the "learnability" reward based on the average r_solve score (avg is computed in this function). A high reward means that the model's proposed question is answerable but challenging.
"""
def compute_r_propose(r_solve: Float[Tensor, "task batch_size"]) -> Float[Tensor, ""]:
    pass

"""
Computes the total reward for the model's answer, validating the format and correctness of the answer.
"""
def compute_r_total(answers: list[Answer], samples: list[Sample]) -> Float[Tensor, ""]:
    pass
