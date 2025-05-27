"""
AZR (Adaptive Zero-shot Reasoning) Trainer Module.

This module contains the AZRTrainer class and related utilities for training
language models using the AZR methodology. It provides functionality for
rollout phases, learning phases, and objective computation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float

from buffer.base_buff import BaseBuffer, MegaBuffer, Sample, IOPair
from model.args import AZRArgs
from model.compute.reward import compute_r_total
from model.inference import generate_response, generate_response_bulk
from custom_types import Answer, TaskType, Reward
from utils.validate_by_executing import validate_by_executing_induction, validate_by_executing_deduction_abduction
from utils.string_formatting import format_task_prompts, format_for_abduction, format_for_induction


def create_optimizer_and_scheduler() -> torch.optim.Optimizer:
    """
    Create AdamW optimizer and learning rate scheduler.
    
    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer with scheduler.
    """
    ...


def format_sample_from_io_pairs(valid_pairs: list[IOPair]) -> Sample:
    pass


def create_sample_from_answer(answer: Answer) -> Sample:
    pass


def extract_program_input_output(response: str, task_type: TaskType) -> Answer:
    return extract_program_input_output_bulk([response], task_type)[0]

def extract_program_input_output_bulk(responses: list[str], task_type: TaskType) -> list[Answer]:
    pass

def create_sample_from_answer(answer: Answer, task_type: TaskType) -> Sample:
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
    tokenizer: AutoTokenizer

    def __init__(self, args: AZRArgs, mega_buffer: MegaBuffer, tokenizer: AutoTokenizer,
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

    def compute_azr_objective(self) -> Float[torch.Tensor, ""]:
        """
        Compute the AZR training objective.
        
        Returns:
            Float[torch.Tensor, ""]: The computed objective value as a scalar tensor.
        """
        pass

    def rollout_phase(self) -> None:
        """
        Execute the rollout phase of AZR training.
        
        Generates data using the current training model and stores it in a buffer
        for subsequent learning phase processing.
        
        Returns:
            BaseBuffer: BaseBuffer containing rollout data for the learning phase.
        """
        for i in range(self.args.train_batch_size):
            program = self.mega_buffer.sample_abduction_deduction()

            # format somehow?
            io_pairs: list[IOPair] = self.generate_io_pairs(program, self.args.n_samples_to_estimate_task_accuracy)
            valid_pairs = validate_by_executing_induction(io_pairs)

            sample = format_sample_from_io_pairs([io for (io, _) in valid_pairs])
            self.mega_buffer.induction_buffer.extend(sample)

            abduction_reward, abduction_answer = self.propose_task(TaskType.ABDUCTION)
            deduction_reward, deduction_answer = self.propose_task(TaskType.DEDUCTION)

            if abduction_answer is not None:
                sample = create_sample_from_answer(abduction_answer, TaskType.ABDUCTION)
                self.mega_buffer.abduction_buffer.extend(sample)

            if deduction_answer is not None:
                sample = create_sample_from_answer(deduction_answer, TaskType.DEDUCTION)
                self.mega_buffer.deduction_buffer.extend(sample)

        all_rewards = torch.tensor((len(TaskType),))
        for i, task_type in enumerate(TaskType):
            samples: list[Sample] = self.mega_buffer.sample_from_buffer(task_type, self.args.train_batch_size)
            task_prompts = format_task_prompts(samples)
            responses = generate_response_bulk(self.args, self.training_model, self.tokenizer, task_prompts)

            answers = extract_program_input_output_bulk(responses, task_type)
            reward = compute_r_total(answers, samples)
            all_rewards[i] = reward

        # policy update


    def propose_task(self, task_type: TaskType) -> tuple[Reward, Answer | None]:
        sample = self.mega_buffer.sample_abduction_deduction()
        prompt = format_for_abduction(sample)
        response = generate_response(self.args, self.training_model, self.tokenizer, prompt)

        answer = extract_program_input_output(response, task_type)
        is_valid, reward = validate_by_executing_deduction_abduction(
            answer
        )

        return reward, answer if is_valid else None

    def generate_io_pairs(self, program: Sample, num_io_pairs: int) -> list[IOPair]:
        induction_prompt = format_for_induction(program)

        responses = []
        for i in range(num_io_pairs):
            responses.append(generate_response(self.args, self.training_model, self.tokenizer, induction_prompt))

        return responses

   
    def learning_phase(self, buffer: BaseBuffer) -> None:
        """
        Execute the learning phase of AZR training.
        
        Updates the training model parameters based on the data collected
        during the rollout phase.
        
        Args:
            buffer (Buffer): Buffer containing rollout data to learn from.
        """
        ...
