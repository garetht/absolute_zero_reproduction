from model.args import AZRArgs
from jaxtyping import Float
from torch import Tensor
"""
Computes the normalized advantage of the model's answer
"""
def compute_advantages(args: AZRArgs, rewards: Float[Tensor, "role task minibatch_size"]) -> Float[Tensor, "role task minibatch_size"]:
    # normalize each reward for each role and task by subtracting the mean and dividing by the standard deviation
    rewards -= rewards.mean(dim=-1, keepdim=True)
    rewards /= (rewards.std(dim=-1, keepdim=True) + args.eps)  # add a small constant to avoid division by zero
    return rewards