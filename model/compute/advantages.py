import torch
from model.args import AZRArgs
from jaxtyping import Float
from torch import Tensor


def compute_advantages(args: AZRArgs, rewards: Float[Tensor, "role task minibatch_size"]) -> Float[Tensor, "role task minibatch_size"]:
    """
    Compute normalized advantages from rewards for policy gradient training.
    
    Args:
        args: AZRArgs object containing eps parameter for numerical stability
        rewards: Shape (role, task, minibatch_size) - Raw rewards for each role/task/sample
        
    Returns:
        Shape (role, task, minibatch_size) - Normalized advantages (mean-centered, std-normalized)
    """
    # If we only have one item in the last dimension, just return the rewards directly
    if rewards.size(-1) == 1:
        return rewards
    
    # normalize each reward for each role and task by subtracting the mean and dividing by the standard deviation
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    
    # Only add eps if std is actually zero to avoid division by zero
    std = torch.where(std == 0, args.eps, std)
    
    normalized = (rewards - mean) / std
    return normalized