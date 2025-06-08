from dataclasses import dataclass

import constants
import torch

@dataclass
class AZRArgs:
    d_vocab: int  # Default for GPT-2 and similar models
    dtype: torch.dtype = torch.bfloat16

    # Basic / global
    seed: int = constants.RANDOM_SEED
    run_name: str = "AZR-Run"

    # Wandb / logging
    use_wandb: bool = True
    wandb_project_name: str = "AZR"
    wandb_entity: str | None = None
    
    # HuggingFace Hub
    push_to_hub: bool = True
    hub_repo_prefix: str = "azr"  # Will create repos like "azr-run-20241201-123456"

    # Duration of different phases
    total_phases: int = 10

    # Optimization hyperparameters
    max_grad_norm: float = 1.0

    # Base model & sampling arguments
    base_model: str = constants.MODEL_NAME

    # Model Configuration
    max_prompt_length: int = 128
    max_response_length: int = 256

    # Training Settings
    lr: float = 1e-6 
    grad_clip: float = 1.0

    # RL Settings
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    # our params
    n_minibatches: int = 2  # Reduced from 4
    minibatch_size: int = 4  # Reduced from 8
    batch_size = minibatch_size * n_minibatches
    clip_ratio: float = 0.2
    eps: float = 1e-5
