"""
AZR (Adaptive Zero-shot Reasoning) Trainer Module.

This module contains the AZRTrainer class and related utilities for training
language models using the AZR methodology. It provides functionality for
rollout phases, learning phases, and objective computation.
"""
import torch
import wandb
from jaxtyping import Float, Int
from transformers import AutoModelForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from buffer.base_buff import BaseBuffer, MegaBuffer
from constants import DEVICE
from custom_types import MiniBatch, TaskType, Role, IOPair, Answer, Problem
from david.sampler import generate_with_logprobs
from model.args import AZRArgs
from model.compute.advantages import compute_advantages
from model.compute.reward import compute_r_total
from model.inference import generate_response, generate_response_bulk, \
    generate_response_bulk_with_grads
from model.eval.baselines import run_baseline_evaluation_prime_samples
from utils.debug_grads import debug_tensor_grads
from utils.string_formatting import validate_proposer_formatting_and_correctness, \
    create_problem_from_answer, validate_single_response, CHECK_MAP


def create_problem_from_io_pairs(prime: int, io_pairs: list[IOPair[int]]) -> Problem:
    """Create a Problem from prime and IO pairs for induction"""
    return Problem(
        prime=prime,
        x_list=[io.input_str for io in io_pairs],
        y_list=[io.output_str for io in io_pairs],
        task_type=TaskType.INDUCTION
    )


class AZRTrainer:
    """
    AZR (Adaptive Zero-shot Reasoning) Trainer for zero data training.

    Attributes:
        training_model (AutoModelForCausalLM): The model being trained.
        mega_buffer (object): Buffer for storing rollout data.
        step (int): Current training step counter.
        args (AZRArgs): Training configuration arguments.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
    """
    training_model: AutoModelForCausalLM
    mega_buffer: MegaBuffer
    step: int
    tokenizer: PreTrainedTokenizerFast

    def __init__(self, args: AZRArgs, mega_buffer: MegaBuffer,
                 tokenizer: PreTrainedTokenizerFast,
                 optimizer: torch.optim.Optimizer,
                 training_model: AutoModelForCausalLM, run_name: str):
        """
        Initializes a new training instance with the provided configuration and components.

        Sets up the training environment by storing references to the model, tokenizer, optimizer,
        and data buffer. Initializes the training step counter to zero and stores the run name
        for tracking purposes.

        :param args: Configuration arguments containing training parameters and settings
        :param mega_buffer: Data buffer containing training samples and batching functionality
        :param tokenizer: Pre-trained tokenizer for text processing and encoding
        :param optimizer: PyTorch optimizer for model parameter updates
        :param training_model: Causal language model to be trained
        :param run_name: Identifier string for the current training run
        """
        self.args = args
        self.training_model = training_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.mega_buffer = mega_buffer
        self.step = 0
        self.run_name = run_name

    def compute_azr_objective(self, advantages: Float[torch.Tensor, "role task minibatch_size"],
                              new_logprobs: Float[torch.Tensor, "role task minibatch_size seq_len d_vocab"],
                              completion_ids: Float[torch.Tensor, "role task minibatch_size seq_len"],
                              minibatch: MiniBatch) -> Float[torch.Tensor, ""]:
        """
        Compute the AZR training objective.
        
        Returns:
            Float[torch.Tensor, ""]: The computed objective value as a scalar tensor.
        """
        # compute the importance ratio using logprobs and sample_ids
        # using the new sample_ids and the old logprobs, get the logprobs from the old policy for the new sample_ids

        new_logprobs = new_logprobs.gather(-1, minibatch.sample_ids.unsqueeze(-1).to(torch.int64)).squeeze(
            -1)  # shape: (role, task, minibatch_size, seq_len)

        old_logprobs = minibatch.logprobs

        importance_ratio = (new_logprobs - old_logprobs).exp()  # shape: (role, task, minibatch_size, seq_len)

        # Apply attention masks to zero out padded positions
        attention_masks = minibatch.attention_masks.float()  # shape: (role, task, minibatch_size, seq_len)
        masked_importance_ratio = importance_ratio * attention_masks

        unsqueezed_advantages = advantages.unsqueeze(-1)
        non_clipped = unsqueezed_advantages * masked_importance_ratio  # shape: (role, task, minibatch_size, seq_len, )
        # compute the clipped objective
        clipped = unsqueezed_advantages.clamp(-self.args.clip_ratio,
                                   self.args.clip_ratio) * masked_importance_ratio  # shape: (role, task, minibatch_size, seq_len,


        debug_tensor_grads(non_clipped, "non_clipped")
        debug_tensor_grads(clipped, "clipped")
        debug_tensor_grads(masked_importance_ratio, "masked_importance_ratio")
        debug_tensor_grads(importance_ratio, "importance_ratio")
        debug_tensor_grads(attention_masks, "attention_masks")

        # Use attention masks for proper averaging - only count valid positions
        objective_per_position = torch.minimum(non_clipped, clipped)
        valid_positions = attention_masks.sum()  # Total number of valid positions
        if valid_positions > 0:
            return (objective_per_position * attention_masks).sum() / valid_positions  # Masked average
        else:
            return torch.tensor(0.0, device=objective_per_position.device)  # Fallback if no valid positions

    # Ensure gradients are not computed
    @torch.no_grad()
    def rollout_phase(self) ->  None:
        """
        Execute the rollout phase of AZR training.
        
        Generates data using the current training model and stores it in a buffer
        for subsequent learning phase processing.
        
        Returns:
            BaseBuffer: BaseBuffer containing rollout data for the learning phase.
        """
        proposer_format_correctness_rewards = torch.zeros((len(TaskType), self.args.batch_size), device=DEVICE)
        all_rewards = torch.zeros((len(Role), len(TaskType), self.args.batch_size), device=DEVICE)

        # Sample problems for each task type (batch_size problems each)
        induction_problems = self.mega_buffer.sample_from_buffer(num_to_sample=self.args.batch_size)
        abduction_problems = self.mega_buffer.sample_from_buffer(num_to_sample=self.args.batch_size)
        deduction_problems = self.mega_buffer.sample_from_buffer(num_to_sample=self.args.batch_size)

        # BATCH ALL PROPOSER CALLS FOR MASSIVE PARALLELIZATION
        
        # Collect prompts for each task type
        induction_prompts = [problem.get_prompt(Role.PROPOSER) for problem in induction_problems]
        abduction_prompts = [problem.get_prompt(Role.PROPOSER) for problem in abduction_problems]
        deduction_prompts = [problem.get_prompt(Role.PROPOSER) for problem in deduction_problems]

        # Generate responses in batches (3 bulk calls instead of 3*batch_size individual calls)
        induction_responses, induction_logprobs_bulk, induction_sample_ids_bulk, _, induction_attention_masks_bulk = generate_response_bulk(
            self.args, self.training_model, self.tokenizer, induction_prompts)
        abduction_responses, abduction_logprobs_bulk, abduction_sample_ids_bulk, _, abduction_attention_masks_bulk = generate_response_bulk(
            self.args, self.training_model, self.tokenizer, abduction_prompts)
        deduction_responses, deduction_logprobs_bulk, deduction_sample_ids_bulk, _, deduction_attention_masks_bulk = generate_response_bulk(
            self.args, self.training_model, self.tokenizer, deduction_prompts)

        # Process responses and validation
        for batch_idx in range(self.args.batch_size):
            # INDUCTION - process individual response from batch
            induction_response = induction_responses[batch_idx]
            induction_answers = validate_single_response(induction_response, CHECK_MAP[TaskType.INDUCTION])
            valid_answers = [a for a in induction_answers if a.is_valid]
            valid_pairs = [IOPair(input_str=e.input, output_str=e.output) for e in valid_answers]
            
            if len(valid_pairs) > 0:
                problem = create_problem_from_io_pairs(induction_problems[batch_idx].prime, valid_pairs)
                self.mega_buffer.buffer.append(problem)

            # ABDUCTION and DEDUCTION - validate responses
            abduction_answer = validate_proposer_formatting_and_correctness(abduction_responses[batch_idx], TaskType.ABDUCTION)
            deduction_answer = validate_proposer_formatting_and_correctness(deduction_responses[batch_idx], TaskType.DEDUCTION)

            # Add valid problems to buffer
            if abduction_answer.reward >= 0:
                problem = create_problem_from_answer(abduction_answer, TaskType.ABDUCTION)
                self.mega_buffer.buffer.append(problem)
            if deduction_answer.reward >= 0:
                problem = create_problem_from_answer(deduction_answer, TaskType.DEDUCTION)
                self.mega_buffer.buffer.append(problem)

            # Store logprobs and rewards in buffer
            # ABDUCTION
            self.mega_buffer.logprobs[Role.PROPOSER.value, TaskType.ABDUCTION.value, batch_idx, ...] = abduction_logprobs_bulk[batch_idx]
            proposer_format_correctness_rewards[TaskType.ABDUCTION.value, batch_idx] = abduction_answer.reward
            self.mega_buffer.sample_ids[Role.PROPOSER.value, TaskType.ABDUCTION.value, batch_idx, ...] = abduction_sample_ids_bulk[batch_idx]
            self.mega_buffer.attention_masks[Role.PROPOSER.value, TaskType.ABDUCTION.value, batch_idx, ...] = abduction_attention_masks_bulk[batch_idx]
            
            # DEDUCTION
            self.mega_buffer.logprobs[Role.PROPOSER.value, TaskType.DEDUCTION.value, batch_idx, ...] = deduction_logprobs_bulk[batch_idx]
            proposer_format_correctness_rewards[TaskType.DEDUCTION.value, batch_idx] = deduction_answer.reward
            self.mega_buffer.sample_ids[Role.PROPOSER.value, TaskType.DEDUCTION.value, batch_idx, ...] = deduction_sample_ids_bulk[batch_idx]
            self.mega_buffer.attention_masks[Role.PROPOSER.value, TaskType.DEDUCTION.value, batch_idx, ...] = deduction_attention_masks_bulk[batch_idx]
            
            # INDUCTION
            self.mega_buffer.logprobs[Role.PROPOSER.value, TaskType.INDUCTION.value, batch_idx, ...] = induction_logprobs_bulk[batch_idx]
            proposer_format_correctness_rewards[TaskType.INDUCTION.value, batch_idx] = torch.tensor(
                [a.reward for a in induction_answers], device=DEVICE).mean()
            self.mega_buffer.sample_ids[Role.PROPOSER.value, TaskType.INDUCTION.value, batch_idx, ...] = induction_sample_ids_bulk[batch_idx]
            self.mega_buffer.attention_masks[Role.PROPOSER.value, TaskType.INDUCTION.value, batch_idx, ...] = induction_attention_masks_bulk[batch_idx]

        # SOLVE PROBLEMS
        for task_type in TaskType:
            problems: list[Problem] = self.mega_buffer.sample_from_buffer(num_to_sample=self.args.batch_size)
            task_prompts = [problem.get_prompt(Role.SOLVER) for problem in problems]
            print(f"{task_prompts=}")
            print(f"{len(problems)=}")
            responses, logprobs, sample_ids, prompt_ids, attention_masks = generate_response_bulk(self.args,
                                                                                                  self.training_model,
                                                                                                  self.tokenizer,
                                                                                                  task_prompts)
            # write logprobs to the mega buffer
            # megabuffer.logprobs shape : (role task batch_size seq_len vocab_size)
            # logprobs obj shape (batchsize seq_len vocab_size)
            self.mega_buffer.logprobs[Role.SOLVER.value, task_type.value, ...] = logprobs
            self.mega_buffer.sample_ids[Role.SOLVER.value, task_type.value, ...] = sample_ids
            self.mega_buffer.attention_masks[Role.SOLVER.value, task_type.value, ...] = attention_masks
            r_format_proposer = proposer_format_correctness_rewards[task_type.value, ...]  # shape: (batch_size)
            # compute rewards
            # one reward per task type per batch item
            for role in Role:
                # we want to pass proposer_format rewards and samples, from which we can compute r_solve
                # TODO - compute formatting rewards for the solver response before rtotal?
                # ie convert response to answer objects (which store the formatting reward)
                all_rewards[role.value][task_type.value] = compute_r_total(self.args, problems, responses, role,
                                                                           task_type,
                                                                           r_format_proposer)

        # Store rewards in buffer for sampling
        self.mega_buffer.rewards = all_rewards

    # DEPRECATED: These functions are replaced by batched approach in rollout_phase
    # Keeping them commented out for now in case we need to revert
    # def propose_task(self, task_type: TaskType) -> tuple[
    #     str, Float[torch.Tensor, "seq_len vocab_size"], Int[torch.Tensor, "seq_len"], Int[torch.Tensor, "seq_len"]]:
    #     problem = self.mega_buffer.sample_from_buffer(num_to_sample=1)[0]
    #     prompt = problem.get_prompt(Role.PROPOSER)
    #     response, logprobs, sample_ids, prompt_tokens, attention_mask = generate_response(self.args,
    #                                                                                       self.training_model,
    #                                                                                       self.tokenizer, prompt)
    #
    #     print(f"{response=}")
    #     # Note: prompt_tokens no longer needed since Problem caches prompts
    #     return response, logprobs, sample_ids, attention_mask
    #
    # def generate_and_validate_io_pairs(self, program: Problem, num_io_pairs: int) -> tuple[
    #     list[IOPair], Float[torch.Tensor, "seq_len vocab_size"], Int[torch.Tensor, "seq_len"], Int[
    #         torch.Tensor, "max_prompt_len"], list[Answer], Int[torch.Tensor, "seq_len"]]:
    #     induction_prompt = program.get_prompt(Role.PROPOSER)
    #     response, logprobs, sample_ids, prompt_tokens, attention_mask = generate_response(self.args,
    #                                                                                       self.training_model,
    #                                                                                       self.tokenizer,
    #                                                                                       induction_prompt)
    #
    #     answers = validate_single_response(response, CHECK_MAP[TaskType.INDUCTION])
    #     valid_answers = [a for a in answers if a.is_valid]
    #
    #     io_pairs = [IOPair(input_str=e.input, output_str=e.output) for e in valid_answers]
    #     # Note: prompt_tokens no longer needed since Problem caches prompts
    #     return io_pairs, logprobs, sample_ids, prompt_tokens, answers, attention_mask

    def learning_phase(self) -> None:
        """
        Execute the learning phase of AZR training.
        
        Updates the training model parameters based on the data collected
        during the rollout phase.
        
        Args:
            buffer (Buffer): Buffer containing rollout data to learn from.
        """

        self.mega_buffer.reset()

        self.rollout_phase()

        # now do minibatch policy updates
        for mini_batch in self.mega_buffer.get_minibatches(self.training_model, self.tokenizer):
            self.step += 1
            # first do a forward pass on current policy to get the logprobs used in importance ratio
            # generate for both roles since loss uses both proposer and solver logprobs
            new_logprobs = torch.zeros(
                (len(Role), len(TaskType), self.args.minibatch_size, self.args.max_response_length, self.args.d_vocab),
                device=DEVICE, dtype=self.args.dtype
            )

            new_attention_masks = torch.zeros(
                (len(Role), len(TaskType), self.args.minibatch_size, self.args.max_response_length),
                device=DEVICE, dtype=torch.int
            )

            for role in Role:
                prompts = [problem.get_prompt(role) for problem in mini_batch.samples]
                all_logprobs, logprobs, completion_ids, attention_masks = generate_response_bulk_with_grads(
                    self.args, self.training_model, self.tokenizer, prompts
                )

                # Fill tensor for this role across all task types (but only the problem's specific task type matters)
                for mb_idx, problem in enumerate(mini_batch.samples):
                    print(f"{all_logprobs.shape=}")
                    print(f"{new_logprobs.shape=}")
                    new_logprobs[role.value, problem.task_type.value, mb_idx] = all_logprobs[mb_idx]
                    new_attention_masks[role.value, problem.task_type.value, mb_idx] = attention_masks[mb_idx]

            advantages = compute_advantages(self.args, mini_batch.rewards)  # shape role task minibatch_size
            objective = self.compute_azr_objective(advantages, new_logprobs, completion_ids, mini_batch)

            self.optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(self.training_model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # Evaluate after gradient update
            eval_results = run_baseline_evaluation_prime_samples(
                self.args, self.training_model, self.tokenizer, mini_batch.samples
            )
            print(f"Minibatch accuracy: {eval_results['accuracy']:.2%}")

            # Log metrics to wandb if enabled
            if self.args.use_wandb:
                # Log accuracy
                wandb.log({"minibatch_accuracy": eval_results['accuracy']}, step=self.step)

                # Log rewards by role and task type
                reward_logs = {}
                for role_idx, role in enumerate(['proposer', 'solver']):
                    for task_idx, task in enumerate(['abduction', 'deduction', 'induction']):
                        mean_reward = self.mega_buffer.rewards[role_idx, task_idx].mean().item()
                        reward_logs[f"reward/{role}_{task}"] = mean_reward

                wandb.log(reward_logs, step=self.step)
