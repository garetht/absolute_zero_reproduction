import torch
from model.args import AZRArgs
from jaxtyping import Float
from torch import Tensor
"""
Computes the normalized advantage of the model's answer
"""
def compute_advantages(args: AZRArgs, rewards: Float[Tensor, "role task minibatch_size"]) -> Float[Tensor, "role task minibatch_size"]:
    # If we only have one item in the last dimension, just return the rewards directly
    if rewards.size(-1) == 1:
        return rewards
    print(f"{rewards.shape=}")
    # normalize each reward for each role and task by subtracting the mean and dividing by the standard deviation
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    
    # Only add eps if std is actually zero to avoid division by zero
    std = torch.where(std == 0, args.eps, std)
    
    normalized = (rewards - mean) / std
    return normalized
