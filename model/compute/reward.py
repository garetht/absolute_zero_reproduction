
from torch import Tensor
from jaxtyping import Float
import torch
from custom_types import Role, TaskType
from model.trainer import validate_formatting_and_correctness

"""
Computes the "learnability" reward based on the average r_solve score (avg is computed in this function). A high reward means that the model's proposed question is answerable but challenging.
"""
def compute_r_propose(r_solve: Float[Tensor, "batch_size"]) -> Float[Tensor, ""]:
    """
    Computes the r_propose reward based on the average r_solve score.
    """
    r_solve = r_solve.mean(dim=-1, keepdim=True)  # average over the batch size
    # r_propose is 0 if avg rsolve is 0 or 1 (ie problem is too easy or too hard)
    r_propose = 1.0 - r_solve
    r_propose[r_solve == 0.0] = 0.0
    r_propose[r_solve == 1.0] = 0.0
    return r_propose


def compute_r_total(solver_responses: list[str], role: Role, task_type: TaskType, r_proposer_format: Float[Tensor, "batch_size"]) -> Float[Tensor, "batch_size"]:
    """
    Computes the total reward for the model's responses. First it computes the formatting and correctness reward of the solver's response, then it computes the r_total reward based on r_solve and r_propose. If the role is proposer, it returns the r_propose reward if r_proposer_format is greater than or equal to 0, otherwise it returns the value in r_proposer_format. If the role is solver, it returns the r_solve reward. 
    """

    answers = [validate_formatting_and_correctness(response, task_type) for response in solver_responses] # this is len batch_size
    r_solve = torch.tensor([answer.reward for answer in answers], dtype=torch.float32)
    if role == Role.PROPOSER:
        # create a tensor to return and populate it with r_propose if r_proposer_format is  >0, else populate with value in r_proposer_format
        r_propose = compute_r_propose(r_solve)
        reward = torch.where(r_proposer_format >= 0, r_propose, r_proposer_format)
        return reward
    elif role == Role.SOLVER:
        # if the role is solver, we just return the r_solve
        return r_solve


