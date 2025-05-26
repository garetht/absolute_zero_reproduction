"""
Computes the normalized advantage of the model's answer
"""
def compute_advantages(args: AZRArgs, rewards: Float[Tensor, "task batch_size"]) -> Float[Tensor, "task"]:
    pass