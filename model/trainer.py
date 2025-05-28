"""
AZR (Adaptive Zero-shot Reasoning) Trainer Module.

This module contains the AZRTrainer class and related utilities for training
language models using the AZR methodology. It provides functionality for
rollout phases, learning phases, and objective computation.
"""

import torch
from transformers import AutoModelForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from jaxtyping import Float, Int

from buffer.base_buff import BaseBuffer, MegaBuffer, 
from model.args import AZRArgs
from model.compute.advantages import compute_advantages
from model.compute.reward import compute_r_total
from model.inference import generate_response, generate_response_bulk, remove_dvocab_from_logprobs
from custom_types import Answer, MiniBatch, TaskType, Role, BaseSample, IOPair
from utils.validate_by_executing import validate_by_executing_induction, validate_by_executing_deduction_abduction
from utils.string_formatting import format_task_prompts, format_for_abduction, format_for_induction


def create_optimizer_and_scheduler() -> torch.optim.Optimizer:
    """
    Create AdamW optimizer and learning rate scheduler.
    
    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer with scheduler.
    """
    ...


def format_sample_from_io_pairs(valid_pairs_and_rewards: list[IOPair]) -> BaseSample:
    pass

def extract_io_pairs_from_string(response: str, num_io_pairs: int) -> list[IOPair]:
    pass

def validate_formatting_and_correctness(response: str, task_type: TaskType) -> Answer:
    return validate_formatting_and_correctness_bulk([response], task_type)[0]

def validate_formatting_and_correctness_bulk(responses: list[str], task_type: TaskType) -> list[Answer]:
    pass # call validate by executing?

def create_sample_from_answer(answer: Answer, task_type: TaskType) -> BaseSample:
    pass





class AZRTrainer:
    """
    AZR (Adaptive Zero-shot Reasoning) Trainer for zero data training.

    Attributes:
        training_model (AutoModelForCausalLM): The model being trained.
        megabuffer (object): Buffer for storing rollout data.
        step (int): Current training step counter.
        args (AZRArgs): Training configuration arguments.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
    """
    training_model: AutoModelForCausalLM
    mega_buffer: MegaBuffer
    step: int
    tokenizer: PreTrainedTokenizerFast

    def __init__(self, args: AZRArgs, mega_buffer: MegaBuffer, tokenizer: PreTrainedTokenizerFast,
                 training_model: AutoModelForCausalLM):
        """
        Initialize the AZR trainer with models and configuration.
        
        Args:
            args (AZRArgs): Configuration arguments for training.
            training_model (AutoModelForCausalLM): The model to be trained.
        """
        self.args = args
        self.training_model = training_model
        self.tokenizer = tokenizer
        self.optimizer = create_optimizer_and_scheduler()
        self.mega_buffer = mega_buffer
        self.step = 0

    def compute_azr_objective(self, advantages: Float[torch.Tensor, "role task minibatch_size"], new_logprobs: Float[torch.Tensor, "role task minibatch_size seq_len"], new_sample_ids:  Int[torch.Tensor, "role task minibatch_size seq_len"], minibatch: MiniBatch ) -> Float[torch.Tensor, ""]:
        """
        Compute the AZR training objective.
        
        Returns:
            Float[torch.Tensor, ""]: The computed objective value as a scalar tensor.
        """
        # compute the importance ratio using logprobs and sample_ids
        # using the new sample_ids and the old logprobs, get the logprobs from the old policy for the new sample_ids
        old_logprobs = minibatch.logprobs.gather(-1, new_sample_ids.unsqueeze(-1)).squeeze(-1)  # shape: (role, task, minibatch_size, seq_len)
        importance_ratio = (new_logprobs - old_logprobs).exp()  # shape: (role, task, minibatch_size, seq_len)
        non_clipped = advantages * importance_ratio  # shape: (role, task, minibatch_size, seq_len, )
        # compute the clipped objective
        clipped = advantages.clamp(-self.args.clip_ratio, self.args.clip_ratio) * importance_ratio # shape: (role, task, minibatch_size, seq_len, 
        return torch.min(non_clipped, clipped).mean()  # shape: ()

    def rollout_phase(self) -> Float[torch.Tensor, "role task batch_size"]:
        """
        Execute the rollout phase of AZR training.
        
        Generates data using the current training model and stores it in a buffer
        for subsequent learning phase processing.
        
        Returns:
            BaseBuffer: BaseBuffer containing rollout data for the learning phase.
        """
        # 
        proposer_format_correctness_rewards = torch.tensor((len(TaskType), self.args.train_batch_size))
        all_rewards = torch.tensor((len(TaskType), self.args.train_batch_size))
        self.step += 1
        for i in range(self.args.train_batch_size):
            program = self.mega_buffer.sample_abduction_deduction()
            
            # INDUCTION
            io_pairs, induction_logprobs, induction_sample_ids = self.generate_io_pairs(program, self.args.n_samples_to_estimate_task_accuracy)
            valid_pairs, induct_proposer_format_reward = validate_by_executing_induction(io_pairs)
            

            sample = format_sample_from_io_pairs([io for io in valid_pairs])
            self.mega_buffer.seed_buffer.append(sample)

            # ABDUCTION and DEDUCTION
            abduction_response, abduction_logprobs, abduction_sample_ids = self.propose_task(TaskType.ABDUCTION)
            deduction_response, deduction_logprobs, deduction_sample_ids = self.propose_task(TaskType.DEDUCTION)
            
            # validate the responses have correct formatting, and run,  create answer objects during this proccess 
            abduction_answer = validate_formatting_and_correctness(abduction_response, TaskType.ABDUCTION)
            deduction_answer = validate_formatting_and_correctness(deduction_response, TaskType.DEDUCTION)
            
            

            # if the abduction answer has valid formatting
            if abduction_answer.reward >= 0:
                sample = create_sample_from_answer(abduction_answer, TaskType.ABDUCTION)
                self.mega_buffer.seed_buffer.append(sample)

            # if the deduction answer has valid formatting
            if deduction_answer.reward >= 0:
                sample = create_sample_from_answer(deduction_answer, TaskType.DEDUCTION)
                self.mega_buffer.seed_buffer.append(sample)

            # BEFORE SOLVING, WRITE LOGPROBS TO THE MEGA BUFFER AND PARTIAL REWARDS TO OUR TENSOR
            # write logprobs to the mega buffer
            # megabuffer.logprobs shape : (role task batch_size seq_len vocab_size)
            for idx, task_type in enumerate(TaskType):
                match task_type:
                    case TaskType.ABDUCTION:
                        self.mega_buffer.logprobs[0, idx, i, ...] = abduction_logprobs
                        # write the rewards to proposer_format_correctness_rewards
                        proposer_format_correctness_rewards[idx, i] = abduction_answer.reward
                        self.mega_buffer.sample_ids[0, idx, i, ...] = abduction_sample_ids
                    case TaskType.DEDUCTION:
                        self.mega_buffer.logprobs[0, idx, i, ...] = deduction_logprobs
                        proposer_format_correctness_rewards[idx, i] = deduction_answer.reward
                        self.mega_buffer.sample_ids[0, idx, i, ...] = deduction_sample_ids
                    case TaskType.INDUCTION:
                        self.mega_buffer.logprobs[0, idx, i, ...] = induction_logprobs
                        proposer_format_correctness_rewards[idx, i] = induct_proposer_format_reward
                        self.mega_buffer.sample_ids[0, idx, i, ...] = induction_sample_ids

        # SOLVE PROBLEMS
        for i, task_type in enumerate(TaskType):
            samples: list[BaseSample] = self.mega_buffer.solver_sample_from_buffer(self.args.train_batch_size)
            task_prompts = format_task_prompts(samples,task_type)
            responses, logprobs, sample_ids, prompt_ids = generate_response_bulk(self.args, self.training_model, self.tokenizer, task_prompts)
            # write logprobs to the mega buffer
            # megabuffer.logprobs shape : (role task batch_size seq_len vocab_size)
            # logprobs obj shape (batchsize seq_len vocab_size)
            self.mega_buffer.logprobs[1, i, ...] = logprobs
            self.mega_buffer.sample_ids[1, i, ...] = sample_ids
            r_format_proposer = proposer_format_correctness_rewards[i, ...]  # shape: (batch_size)
            # compute rewards
            # one reward per task type per batch item
            for j, role in enumerate(Role):
                # we want to pass proposer_format rewards and samples, from which we can compute r_solve
                # TODO - compute formatting rewards for the solver response before rtotal?
                # ie convert response to answer objects (which store the formatting reward)
                all_rewards[j][i] = compute_r_total(samples, responses, role, r_format_proposer)
         
        return all_rewards  # shape: (role, task, batch_size)


    def propose_task(self, task_type: TaskType) -> tuple[str, Float[torch.Tensor, "seq_len vocab_size"], Int[torch.Tensor, "seq_len"]]:
        sample = self.mega_buffer.sample_abduction_deduction()
        match task_type:
            case TaskType.ABDUCTION:
                prompt = format_for_abduction(sample)
            case TaskType.DEDUCTION:
                prompt = format_for_induction(sample)
            case _:
                raise ValueError(f"Unsupported task type in propose_task: {task_type}")
        response, logprobs, sample_ids, prompt_tokens = generate_response(self.args, self.training_model, self.tokenizer, prompt)
        sample.prompt_tokens = prompt_tokens
        return response, logprobs, sample_ids


    def generate_io_pairs(self, program: BaseSample, num_io_pairs: int) -> tuple[list[IOPair], Float[torch.Tensor, "seq_len vocab_size"], Int[torch.Tensor, "seq_len"]]:
        induction_prompt = format_for_induction(program, num_io_pairs)
        response, logprobs, sample_ids, prompt_tokens = generate_response(self.args, self.training_model, self.tokenizer, induction_prompt)
        io_pairs = extract_io_pairs_from_string(response, num_io_pairs)
        program.prompt_tokens = prompt_tokens
        return io_pairs, logprobs, sample_ids

   
    def learning_phase(self) -> None:
        """
        Execute the learning phase of AZR training.
        
        Updates the training model parameters based on the data collected
        during the rollout phase.
        
        Args:
            buffer (Buffer): Buffer containing rollout data to learn from.
        """        
        
        all_rewards = self.rollout_phase()
        # now do minibatch policy updates
        for mini_batch in self.mega_buffer.get_minibatches(self.args):
            # first do a forward pass on current policy to get the logprobs used in importance ratio
            # get the prompts that we need to use for the forawrd
            prompts = [self.tokenizer.decode(s.prompt_tokens, skip_special_tokens=True) for s in mini_batch.samples]
            _, logprobs, sample_ids, _ = generate_response_bulk(
                self.args, self.training_model, self.tokenizer, prompts
            )
            logprobs = remove_dvocab_from_logprobs(logprobs, sample_ids)
            advantages = compute_advantages(self.args, all_rewards) # shape role task minibatch_size
            objective = self.compute_azr_objective(advantages, logprobs, sample_ids, 
                                                   mini_batch)
            self.optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(self.training_model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()