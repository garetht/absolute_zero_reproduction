import torch
from transformers import AutoModelForCausalLM
from jaxtyping import Float

Buffer = object
AZRArgs = object

# create adamw optimizer and scheduler
def create_optimizer_and_scheduler() -> torch.optim.Optimizer: ...

class AZRTrainer:
    training_model: AutoModelForCausalLM
    reference_model: AutoModelForCausalLM
    buffer: object
    step: int

    def __init__(self, args: AZRArgs, training_model: AutoModelForCausalLM, reference_model: AutoModelForCausalLM):
        self.args = args
        self.training_model = training_model
        self.reference_model = reference_model
        self.optimizer = create_optimizer_and_scheduler()
        self.step = 0

    def compute_azr_objective(self) -> Float[torch.Tensor, ""]: ...

    def rollout_phase(self) -> Buffer:
        pass

    def learning_phase(self, buffer: Buffer) -> None: ...
