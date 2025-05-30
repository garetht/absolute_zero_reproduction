from jaxtyping import Int, Float
import numpy
from torch import Tensor
import torch

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from model.args import AZRArgs
from model.eval.prime_inversion import generate_problems, PRIMES
from custom_types import MiniBatch, Role, TaskType, BaseSample, Problem


class BaseBuffer:
    """
    Base class for the buffer objects (also class for the seed buffer).
    """

    def __init__(
        self,
        args: AZRArgs,
        samples: list[BaseSample],
    ):
        self.args = args
        self.samples = samples

    def sample_ids(self):
        return torch.cat([s.sample_ids for s in self.samples])


class MegaBuffer:
    def __init__(
        self,
        args: AZRArgs,
        logprobs: Int[Tensor, "role task batch_size max_response_len"],
        sample_ids: Int[Tensor, "role task batch_size max_response_len"],
        attention_masks: Int[Tensor, "role task batch_size max_response_len"] = None,
        rewards: Float[Tensor, "role task batch_size"] = None,
    ):
        self.args = args
        self.seed_buffer: list[Problem] = []
        self.logprobs = logprobs
        self.sample_ids = sample_ids
        # Initialize attention masks if not provided
        if attention_masks is None:
            self.attention_masks = torch.ones_like(sample_ids)
        else:
            self.attention_masks = attention_masks
        # Initialize rewards if not provided
        if rewards is None:
            self.rewards = torch.zeros((len(Role), len(TaskType), args.batch_size), 
                                     dtype=logprobs.dtype, device=logprobs.device)
        else:
            self.rewards = rewards
        # batch_size is the index of the sample in the buffer, same for any role task combo
        self.buffer: list[Problem] = []

    def get_minibatches(self, model=None, tokenizer=None) -> list[MiniBatch]:
        # Combine buffer and seed_buffer if needed to reach batch_size
        combined_problems, buffer_mask = self._get_combined_problems_with_mask()
        
        out = []
        for indices in torch.randperm(self.args.batch_size).reshape(
            self.args.n_minibatches, -1
        ):
            minibatch_problems = [combined_problems[i] for i in indices]
            minibatch_buffer_mask = [buffer_mask[i] for i in indices]
            
            # Initialize tensors for this minibatch
            minibatch_sample_ids = torch.zeros(
                (len(Role), len(TaskType), self.args.minibatch_size, self.args.max_response_length),
                dtype=torch.int64, device=self.sample_ids.device
            )
            minibatch_logprobs = torch.zeros(
                (len(Role), len(TaskType), self.args.minibatch_size, self.args.max_response_length),
                dtype=self.logprobs.dtype, device=self.logprobs.device
            )
            minibatch_attention_masks = torch.zeros(
                (len(Role), len(TaskType), self.args.minibatch_size, self.args.max_response_length),
                dtype=torch.int, device=self.attention_masks.device
            )
            minibatch_rewards = torch.zeros(
                (len(Role), len(TaskType), self.args.minibatch_size),
                dtype=self.rewards.dtype, device=self.rewards.device
            )
            
            # Fill data for each problem in minibatch
            seed_entries: list[tuple[Problem, int]] = []      # (problem, mb_idx)
            for mb_idx, (problem, is_from_buffer) in enumerate(
                zip(minibatch_problems, minibatch_buffer_mask)
            ):
                if is_from_buffer:
                    # Use pre-calculated data from rollout
                    buffer_idx = indices[mb_idx]  # Original index in buffer
                    minibatch_sample_ids[:, :, mb_idx] = self.sample_ids[:, :, buffer_idx]
                    minibatch_logprobs[:, :, mb_idx] = self.logprobs[:, :, buffer_idx]
                    minibatch_attention_masks[:, :, mb_idx] = self.attention_masks[:, :, buffer_idx]
                    minibatch_rewards[:, :, mb_idx] = self.rewards[:, :, buffer_idx]
                else:
                    seed_entries.append((problem, mb_idx))

            # ---- NEW: single batched forward pass for all seed problems ----
            if seed_entries and model is not None and tokenizer is not None:
                self._calculate_batch_seed_logprobs(
                    seed_entries,
                    minibatch_sample_ids,
                    minibatch_logprobs,
                    minibatch_attention_masks,
                    model,
                    tokenizer,
                )
                # rewards for seed problems stay zero
            
            out.append(
                MiniBatch(
                    samples=minibatch_problems,
                    sample_ids=minibatch_sample_ids,
                    logprobs=minibatch_logprobs,
                    attention_masks=minibatch_attention_masks,
                    rewards=minibatch_rewards,
                )
            )
        return out

    def reset(self) -> None:
        self.seed_buffer.extend(self.buffer)
        self.buffer = []
        self.logprobs = torch.zeros_like(self.logprobs, device=self.logprobs.device, requires_grad=True)
        self.sample_ids = torch.zeros_like(self.sample_ids, device=self.sample_ids.device, dtype=torch.int64)
        self.attention_masks = torch.zeros_like(self.attention_masks, device=self.attention_masks.device)
        self.rewards = torch.zeros_like(self.rewards, device=self.rewards.device)

    def _get_combined_problems_with_mask(self) -> tuple[list[Problem], list[bool]]:
        """Combine buffer and seed_buffer problems to reach batch_size. Returns (problems, is_from_buffer_mask)"""
        if len(self.buffer) >= self.args.batch_size:
            return self.buffer[:self.args.batch_size], [True] * self.args.batch_size
        
        # Need to supplement with seed buffer
        needed = self.args.batch_size - len(self.buffer)
        seed_problems = self.seed_buffer[:needed]
        
        combined_problems = self.buffer + seed_problems
        buffer_mask = [True] * len(self.buffer) + [False] * len(seed_problems)
        
        return combined_problems, buffer_mask
    
    def _calculate_seed_logprobs(
        self,
        problem,
        mb_idx,
        minibatch_sample_ids,
        minibatch_logprobs,
        minibatch_attention_masks,
        model,
        tokenizer,
    ):
        """Calculate logprobs on-the-fly for a seed problem (now batched over roles)"""
        from model.inference import generate_response_bulk  # bulk version

        # Collect one prompt per role, then run the model once
        prompts = [problem.get_prompt(role) for role in Role]
        _, logprobs_bulk, sample_ids_bulk, _, attention_masks_bulk = generate_response_bulk(
            self.args, model, tokenizer, prompts
        )

        # Distribute the outputs to their respective role / task indices
        task_idx = problem.task_type.value
        for role_idx, role in enumerate(Role):
            minibatch_sample_ids[role_idx, task_idx, mb_idx] = sample_ids_bulk[role_idx]
            minibatch_logprobs[role_idx, task_idx, mb_idx] = logprobs_bulk[role_idx]
            minibatch_attention_masks[role_idx, task_idx, mb_idx] = attention_masks_bulk[role_idx]

    # ---------------------------------------------------------------------
    # NEW helper: batched seed generation (across problems *and* roles)
    # ---------------------------------------------------------------------
    def _calculate_batch_seed_logprobs(
        self,
        seed_entries: list[tuple[Problem, int]],  # (problem, mb_idx)
        minibatch_sample_ids,
        minibatch_logprobs,
        minibatch_attention_masks,
        model,
        tokenizer,
    ):
        """Run one bulk generation over every (problem, role) pair in seed_entries."""
        from model.inference import generate_response_bulk

        prompts, mapping = [], []   # mapping -> (role_idx, task_idx, mb_idx)
        for problem, mb_idx in seed_entries:
            task_idx = problem.task_type.value
            for role_idx, role in enumerate(Role):
                prompts.append(problem.get_prompt(role))
                mapping.append((role_idx, task_idx, mb_idx))

        # Bulk generation
        _, logprobs_bulk, sample_ids_bulk, _, attention_masks_bulk = generate_response_bulk(
            self.args, model, tokenizer, prompts
        )

        # Scatter back into minibatch tensors
        for i, (role_idx, task_idx, mb_idx) in enumerate(mapping):
            minibatch_sample_ids[role_idx, task_idx, mb_idx] = sample_ids_bulk[i]
            minibatch_logprobs[role_idx, task_idx, mb_idx] = logprobs_bulk[i]
            minibatch_attention_masks[role_idx, task_idx, mb_idx] = attention_masks_bulk[i]

    @property
    def combined_buffer(self):
        return self.seed_buffer + self.buffer

    def sample_from_buffer(self, num_to_sample: int) -> list[Problem]:
        indices = numpy.random.choice(
            len(self.combined_buffer), num_to_sample, replace=True
        )
        return [self.combined_buffer[i] for i in indices]

    def initialize_seed_buffer(
        self, tokenizer: PreTrainedTokenizerFast, num_samples: int = None
    ) -> None:
        """
        Initialize seed buffer with batch_size problems
        """
        if num_samples is None:
            num_samples = self.args.batch_size
        problems = generate_problems(n=num_samples, primes=PRIMES, seed=self.args.seed)
        self.seed_buffer.extend(problems)
