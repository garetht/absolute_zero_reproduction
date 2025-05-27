
from model.args import AZRArgs
from jaxtyping import Float
from torch import Tensor
"""
Computes the normalized advantage of the model's answer
"""
def compute_advantages(args: AZRArgs, rewards: Float[Tensor, "role task batch_size"]) -> Float[Tensor, "task batch_size"]:
    pass