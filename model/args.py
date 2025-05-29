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
    use_wandb: bool = False
    wandb_project_name: str = "AZR"
    wandb_entity: str | None = None

    # Duration of different phases
    total_phases: int = 100

    # Optimization hyperparameters
    max_grad_norm: float = 1.0
    warmup_steps: int = 20
    final_scale: float = 0.1

    # FROM THE PAPER!!

    # Base model & sampling arguments
    base_model: str = constants.MODEL_NAME
    gen_len: int = 30
    prepend_bos: bool = True

    # Model Configuration
    max_prompt_length: int = 128
    max_response_length: int = 256
    seed_batch_factor: int = 4
    max_programs: int = 16384

    # Training Settings
    lr: float = 1e-5 # dave made me change this
    grad_clip: float = 1.0
    total_steps: int = 500

    # RL Settings
    entropy_coefficient: float = 0.001
    ppo_epochs: int = 1
    n_rollouts: int = 1
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    k_references: int = 6
    n_samples_to_estimate_task_accuracy: int = 8
    # our params
    n_minibatches: int = 8
    minibatch_size: int = 4
    batch_size = minibatch_size * n_minibatches
    clip_ratio: float = 0.2
    eps: float = 1e-5
