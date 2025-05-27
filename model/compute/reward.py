
from torch import Tensor
from jaxtyping import Float

from buffer.base_buff import BaseSample
from custom_types import Answer, Reward, Role
"""
Computes the correctness of the model's answer baed on value equality in python
"""
def compute_r_solve(answers: list[Answer], samples: list[BaseSample]) -> Float[Tensor, "batch_size"]:
    pass

"""
Computes the "learnability" reward based on the average r_solve score (avg is computed in this function). A high reward means that the model's proposed question is answerable but challenging.
"""
def compute_r_propose(r_solve: Float[Tensor, "task batch_size"]) -> Float[Tensor, ""]:
    pass

"""
Computes the total reward for the model's answer, validating the format and correctness of the answer. answers_and_rewards is a list of answers and solver format rewards for that answer.
"""
def compute_r_total(answers_and_rewards: list[tuple[Answer, Reward]], role: Role, r_proposer_format: Reward) -> Float[Tensor, "batch_size"]:
    # select the appropriate formatting reward based on the role, if there is no formatting reward, use r_solve or r_propose 
    pass
