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
from model.args import AZRArgs
from model.compute.advantages import compute_advantages
from model.compute.reward import compute_r_total
from model.inference import generate_response, generate_response_bulk, remove_dvocab_from_logprobs, compute_logprobs_for_tokens
from model.eval.baselines import run_baseline_evaluation_prime_samples
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
        
        # Ensure model is in training mode and has gradients enabled
        self.training_model.train()
        for param in self.training_model.parameters():
            param.requires_grad_(True)

    def compute_azr_objective(self, advantages: Float[torch.Tensor, "role task minibatch_size"],
                              new_logprobs: Float[torch.Tensor, "role task minibatch_size seq_len"],
                              new_sample_ids: Int[torch.Tensor, "role task minibatch_size seq_len"],
                              minibatch: MiniBatch) -> Float[torch.Tensor, ""]:
        """
        Compute the AZR training objective.
        
        Returns:
            Float[torch.Tensor, ""]: The computed objective value as a scalar tensor.
        """
        # compute the importance ratio using logprobs and sample_ids
        # using the new sample_ids and the old logprobs, get the logprobs from the old policy for the new sample_ids
        old_logprobs = minibatch.logprobs.gather(-1, new_sample_ids.unsqueeze(-1).to(torch.int64)).squeeze(
            -1)  # shape: (role, task, minibatch_size, seq_len)
        importance_ratio = (new_logprobs - old_logprobs).exp()  # shape: (role, task, minibatch_size, seq_len)

        # Apply attention masks to zero out padded positions
        attention_masks = minibatch.attention_masks.float()  # shape: (role, task, minibatch_size, seq_len)
        masked_importance_ratio = importance_ratio * attention_masks

        unsqueezed_advantages = advantages.unsqueeze(-1)
        non_clipped = unsqueezed_advantages * masked_importance_ratio  # shape: (role, task, minibatch_size, seq_len, )
        # compute the clipped objective
        clipped = unsqueezed_advantages.clamp(-self.args.clip_ratio,
                                   self.args.clip_ratio) * masked_importance_ratio  # shape: (role, task, minibatch_size, seq_len,

        # Use attention masks for proper averaging - only count valid positions
        objective_per_position = torch.minimum(non_clipped, clipped)
        valid_positions = attention_masks.sum()  # Total number of valid positions
        if valid_positions > 0:
            return (objective_per_position * attention_masks).sum() / valid_positions  # Masked average
        else:
            return torch.tensor(0.0, device=objective_per_position.device)  # Fallback if no valid positions

    def rollout_phase(self) -> Float[torch.Tensor, "role task batch_size"]:
        """
        Execute the rollout phase of AZR training.
        
        Generates data using the current training model and stores it in a buffer
        for subsequent learning phase processing.
        
        Returns:
            BaseBuffer: BaseBuffer containing rollout data for the learning phase.
        """
        proposer_format_correctness_rewards = torch.zeros((len(TaskType), self.args.batch_size), device=DEVICE)
        all_rewards = torch.zeros((len(Role), len(TaskType), self.args.batch_size), device=DEVICE)

        for batch_idx in range(self.args.batch_size):
            induction_problem: Problem = self.mega_buffer.sample_from_buffer(num_to_sample=1)[0]
            print(induction_problem)

            # INDUCTION
            valid_pairs, induction_logprobs, induction_sample_ids, induction_prompt_tokens, induction_answers, induction_attention_mask = self.generate_and_validate_io_pairs(
                induction_problem,
                self.args.n_samples_to_estimate_task_accuracy)

            if len(valid_pairs) > 0:
                problem = create_problem_from_io_pairs(induction_problem.prime, valid_pairs)
                self.mega_buffer.buffer.append(problem)

            # ABDUCTION and DEDUCTION
            abduction_response, abduction_logprobs, abduction_sample_ids, abduction_attention_mask = self.propose_task(
                TaskType.ABDUCTION)
            deduction_response, deduction_logprobs, deduction_sample_ids, deduction_attention_mask = self.propose_task(
                TaskType.DEDUCTION)

            # validate the responses have correct formatting, and run,  create answer objects during this proccess
            abduction_answer = validate_proposer_formatting_and_correctness(abduction_response, TaskType.ABDUCTION)
            deduction_answer = validate_proposer_formatting_and_correctness(deduction_response, TaskType.DEDUCTION)

            # if the abduction answer has valid formatting
            if abduction_answer.reward >= 0:
                problem = create_problem_from_answer(abduction_answer, TaskType.ABDUCTION)
                self.mega_buffer.buffer.append(problem)

            # if the deduction answer has valid formatting
            if deduction_answer.reward >= 0:
                problem = create_problem_from_answer(deduction_answer, TaskType.DEDUCTION)
                self.mega_buffer.buffer.append(problem)

            # BEFORE SOLVING, WRITE LOGPROBS TO THE MEGA BUFFER AND PARTIAL REWARDS TO OUR TENSOR
            # write logprobs to the mega buffer
            # megabuffer.logprobs shape : (role task batch_size seq_len vocab_size)
            for idx, task_type in enumerate(TaskType):
                match task_type:
                    case TaskType.ABDUCTION:
                        self.mega_buffer.logprobs[
                            Role.PROPOSER.value, task_type.value, batch_idx, ...] = abduction_logprobs
                        # write the rewards to proposer_format_correctness_rewards
                        proposer_format_correctness_rewards[task_type.value, batch_idx] = abduction_answer.reward
                        self.mega_buffer.sample_ids[
                            Role.PROPOSER.value, task_type.value, batch_idx, ...] = abduction_sample_ids
                        self.mega_buffer.attention_masks[
                            Role.PROPOSER.value, task_type.value, batch_idx, ...] = abduction_attention_mask
                    case TaskType.DEDUCTION:
                        self.mega_buffer.logprobs[
                            Role.PROPOSER.value, task_type.value, batch_idx, ...] = deduction_logprobs

                        proposer_format_correctness_rewards[task_type.value, batch_idx] = deduction_answer.reward
                        self.mega_buffer.sample_ids[
                            Role.PROPOSER.value, task_type.value, batch_idx, ...] = deduction_sample_ids
                        self.mega_buffer.attention_masks[
                            Role.PROPOSER.value, task_type.value, batch_idx, ...] = deduction_attention_mask
                    case TaskType.INDUCTION:
                        self.mega_buffer.logprobs[
                            Role.PROPOSER.value, task_type.value, batch_idx, ...] = induction_logprobs
                        # TODO: maybe not mean??
                        proposer_format_correctness_rewards[task_type.value, batch_idx] = torch.tensor(
                            [a.reward for a in induction_answers], device=DEVICE).mean()
                        self.mega_buffer.sample_ids[
                            Role.PROPOSER.value, task_type.value, batch_idx, ...] = induction_sample_ids
                        self.mega_buffer.attention_masks[
                            Role.PROPOSER.value, task_type.value, batch_idx, ...] = induction_attention_mask

        # SOLVE PROBLEMS
        for task_type in TaskType:
            problems: list[Problem] = self.mega_buffer.sample_from_buffer(num_to_sample=self.args.batch_size)
            task_prompts = [problem.get_prompt(Role.SOLVER) for problem in problems]
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

        return all_rewards  # shape: (role, task, batch_size)

    def propose_task(self, task_type: TaskType) -> tuple[
        str, Float[torch.Tensor, "seq_len vocab_size"], Int[torch.Tensor, "seq_len"], Int[torch.Tensor, "seq_len"]]:
        problem = self.mega_buffer.sample_from_buffer(num_to_sample=1)[0]
        prompt = problem.get_prompt(Role.PROPOSER)
        response, logprobs, sample_ids, prompt_tokens, attention_mask = generate_response(self.args,
                                                                                          self.training_model,
                                                                                          self.tokenizer, prompt)
        # Note: prompt_tokens no longer needed since Problem caches prompts
        return response, logprobs, sample_ids, attention_mask

    def generate_and_validate_io_pairs(self, program: Problem, num_io_pairs: int) -> tuple[
        list[IOPair], Float[torch.Tensor, "seq_len vocab_size"], Int[torch.Tensor, "seq_len"], Int[
            torch.Tensor, "max_prompt_len"], list[Answer], Int[torch.Tensor, "seq_len"]]:
        induction_prompt = program.get_prompt(Role.PROPOSER)
        response, logprobs, sample_ids, prompt_tokens, attention_mask = generate_response(self.args,
                                                                                          self.training_model,
                                                                                          self.tokenizer,
                                                                                          induction_prompt)

        answers = validate_single_response(response, CHECK_MAP[TaskType.INDUCTION])
        valid_answers = [a for a in answers if a.is_valid]

        io_pairs = [IOPair(input_str=e.input, output_str=e.output) for e in valid_answers]
        # Note: prompt_tokens no longer needed since Problem caches prompts
        return io_pairs, logprobs, sample_ids, prompt_tokens, answers, attention_mask

    def learning_phase(self) -> None:
        """
        Execute the learning phase of AZR training.
        
        Updates the training model parameters based on the data collected
        during the rollout phase.
        
        Args:
            buffer (Buffer): Buffer containing rollout data to learn from.
        """

        self.mega_buffer.reset()
        all_rewards = self.rollout_phase()

        # now do minibatch policy updates
        for index, mini_batch in enumerate(self.mega_buffer.get_minibatches(self.training_model, self.tokenizer)):
            minibatch_all_rewards = all_rewards[:, :, index * self.args.minibatch_size:(index + 1) * self.args.minibatch_size]

            self.step += 1
            # Simplified approach: compute logprobs for actual problems and extract what we need
            
            # For each role, compute new logprobs with gradients  
            role_task_logprobs = {}
            for role in Role:
                prompts = [problem.get_prompt(role) for problem in mini_batch.samples]
                
                # Get actual tokens for each problem (using their specific task type)
                actual_tokens = []
                actual_masks = []
                for mb_idx, problem in enumerate(mini_batch.samples):
                    actual_tokens.append(mini_batch.sample_ids[role.value, problem.task_type.value, mb_idx])
                    actual_masks.append(mini_batch.attention_masks[role.value, problem.task_type.value, mb_idx])
                
                actual_tokens = torch.stack(actual_tokens)  # (minibatch_size, seq_len)
                actual_masks = torch.stack(actual_masks)    # (minibatch_size, seq_len)
                
                # Compute logprobs with gradients for this role
                role_logprobs = compute_logprobs_for_tokens(
                    self.training_model, self.tokenizer, prompts, actual_tokens, actual_masks
                )  # (minibatch_size, seq_len, vocab_size)
                
                role_task_logprobs[role] = role_logprobs
            
            # Extract logprobs for each token and create the final tensor
            new_logprobs_list = []
            for mb_idx, problem in enumerate(mini_batch.samples):
                problem_logprobs = []
                for role in Role:
                    # Get the logprobs for this problem and role
                    role_logprobs = role_task_logprobs[role][mb_idx]  # (seq_len, vocab_size)
                    # Extract logprobs for the actual tokens generated
                    tokens = mini_batch.sample_ids[role.value, problem.task_type.value, mb_idx]  # (seq_len,)
                    token_logprobs = torch.gather(role_logprobs, dim=-1, index=tokens.unsqueeze(-1).to(torch.int64)).squeeze(-1)  # (seq_len,)
                    problem_logprobs.append(token_logprobs)
                new_logprobs_list.append(torch.stack(problem_logprobs))  # (role, seq_len)
            
            new_logprobs = torch.stack(new_logprobs_list).transpose(0, 1)  # (role, minibatch_size, seq_len)
            
            new_logprobs = remove_dvocab_from_logprobs(new_logprobs, mini_batch.sample_ids)
            print(f"new_logprobs.requires_grad: {new_logprobs.requires_grad}")
            print(f"new_logprobs.grad_fn: {new_logprobs.grad_fn}")
            
            advantages = compute_advantages(self.args, minibatch_all_rewards)  # shape role task minibatch_size
            print(f"advantages.requires_grad: {advantages.requires_grad}")
            
            objective = self.compute_azr_objective(advantages, new_logprobs, mini_batch.sample_ids,
                                                   mini_batch)
            print(f"objective.requires_grad: {objective.requires_grad}")
            print(f"objective.grad_fn: {objective.grad_fn}")
            
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
                        mean_reward = all_rewards[role_idx, task_idx].mean().item()
                        reward_logs[f"reward/{role}_{task}"] = mean_reward

                wandb.log(reward_logs, step=self.step)
