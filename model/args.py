from dataclasses import dataclass
from typing import Callable

import constants



@dataclass
class AZRArgs:
    # Basic / global
    seed: int = constants.RANDOM_SEED

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "AZR"
    wandb_entity: str | None = None

    # Duration of different phases
    total_phases: int = 100
    batch_size: int = 128

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
    max_prompt_length: int = 6144
    max_response_length: int = 8096
    seed_batch_factor: int = 4
    max_programs: int = 16384

    # Training Settings
    lr: float = 1e-6
    train_batch_size: int = 64 * 6
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
    n_minibatches: int = 16
    minibatch_size: int = 8
    batch_size = minibatch_size*n_minibatches
    d_vocab: int = 50257  # Default for GPT-2 and similar models
    clip_ratio: float = 0.2