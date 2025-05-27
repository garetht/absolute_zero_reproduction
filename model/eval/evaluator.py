import re
import time
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional

from custom_types import ProblemResult
from model.eval.problem import Problem
from model.eval.prompts import create_prompt


class Evaluator:
    """Evaluator class for running model evaluations on the prime inversion problem"""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_new_tokens: int = 100,
                 batch_size: int = 1):
        """
        Initialize the evaluator with model configuration.

        Args:
            model: The AutoModelForCausalLM under evaluation
            tokenizer: The AutoTokenizer for the model
            max_new_tokens: Maximum number of new tokens to generate
            batch_size: Number of problems to process in each batch
        """
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model.config.name_or_path

    def _process_batch(self, batch_problems: List[Problem], batch_idx: int, total_batches: int) -> tuple[
        torch.Tensor, List[str], float]:

        """Process a single batch of problems."""
        batch_prompts = [create_prompt(prob) for prob in batch_problems]

        print(
            f"\n📦 Processing batch {batch_idx + 1}/{total_batches} "
            f"(problems {batch_idx * self.batch_size + 1}-{min((batch_idx + 1) * self.batch_size, batch_idx * self.batch_size + len(batch_problems))})"
        )

        # Tokenize all prompts in the batch
        batch_inputs = self.tokenizer(
            batch_prompts, return_tensors="pt", padding=True, padding_side="left"
        ).to(self.model.device)

        start_time = time.time()
        with torch.no_grad():
            batch_outputs = self.model.generate(
                **batch_inputs,
                max_new_tokens=self.max_new_tokens
            )
        end_time = time.time()

        batch_time = end_time - start_time
        print(f"⏱️  Batch generation time: {batch_time:.2f}s ({batch_time / len(batch_problems):.2f}s per problem)")

        return batch_outputs, batch_prompts, batch_time

    def _process_problem_result(self, problem: Problem, outputs: torch.Tensor,
                                prompt: str, batch_time: float, batch_size: int) -> ProblemResult:
        """Process the result for a single problem."""
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)

        # Get the part of the response that comes after the prompt
        model_response = response[len(prompt):]

        # Extract answer
        extracted_answer = self.extract_boxed_number(model_response)
        correct_answer = problem.x if problem.blank == 'x' else problem.y

        # Compare answers (considering modulo)
        is_correct = False
        if extracted_answer is not None:
            # Check if answers are equivalent modulo prime
            if problem.blank == 'p':
                is_correct = (problem.x * problem.y) % extracted_answer == 1
            else:
                is_correct = (extracted_answer % problem.prime) == (correct_answer % problem.prime)

        print(
            f"""{"✅" if is_correct else "⛔"} | Problem {problem} | Model Response {extracted_answer}""")

        return {
            "problem": problem.desc,
            "extracted_answer": extracted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "time_seconds": batch_time / batch_size  # Approximate per-problem time
        }

    def _print_batch_summary(self, batch_idx: int, batch_correct: int, batch_size: int,
                             no_responses: int,
                             overall_correct: int, total_processed: int):
        """Print summary for the current batch."""
        batch_accuracy = batch_correct / batch_size
        print(
            f"📈 Batch {batch_idx + 1} results: {batch_correct}/{batch_size} correct ({batch_accuracy:.1%}) ")

        current_overall_accuracy = overall_correct / total_processed
        current_responded_accuracy = overall_correct / (total_processed - no_responses)

        print(f"📊 Overall progress: {overall_correct}/{total_processed} correct ({current_overall_accuracy:.1%}) ({current_responded_accuracy:.1%} of responded)")

    def _print_final_results(self, results: Dict[str, Any], total_eval_time: float, num_problems: int):
        """Print final evaluation results."""
        print(f"\n🎯 Evaluation completed!")
        print(f"📊 Final Results:")
        print(f"   • Model: {self.model_name}")
        print(f"   • Total problems: {results['total']}")
        print(f"   • Correct answers: {results['correct']}")
        print(f"   • Overall accuracy: {results['accuracy']:.2%}")
        print(f"   • Total evaluation time: {total_eval_time:.1f}s")
        print(f"   • Average time per problem: {total_eval_time / num_problems:.2f}s")

    @staticmethod
    def extract_boxed_number(text: str) -> Optional[int]:
        """
        Extract a number from LaTeX \\boxed{} notation in the given text.

        Args:
            text: The text to search for boxed numbers

        Returns:
            The integer found within \\boxed{} notation, or None if no match is found
        """
        # Regex pattern to match \boxed{<number>} and extract the number
        regexp_match = re.search(r"\\boxed\{([+-]?\d+)\}", text)
        if regexp_match:
            return int(regexp_match.group(1))
        else:
            return None

    def evaluate(self, problems: List[Problem], eval_start_time: float = None) -> Dict[str, Any]:
        """Run evaluation on the given problems."""
        if eval_start_time is None:
            eval_start_time = time.time()

        results = {
            "model": self.model_name,
            "correct": 0,
            "no_response": 0,
            "total": len(problems),
            "timestamp": datetime.now().isoformat(),
            "problems": [],
            "total_eval_time_seconds": 0.0
        }

        # Calculate total number of batches for progress tracking
        total_batches = (len(problems) + self.batch_size - 1) // self.batch_size

        # Process problems in batches with progress bar
        with tqdm(total=len(problems), desc="Evaluating problems", unit="problem") as pbar:
            for batch_idx, i in enumerate(range(0, len(problems), self.batch_size)):
                batch_problems = problems[i:i + self.batch_size]

                batch_outputs, batch_prompts, batch_time = self._process_batch(
                    batch_problems, batch_idx, total_batches
                )

                # Process each result in the batch
                batch_correct = 0
                for j, (problem, outputs) in enumerate(zip(batch_problems, batch_outputs)):
                    problem_result = self._process_problem_result(
                        problem, outputs, batch_prompts[j], batch_time, len(batch_problems)
                    )

                    if problem_result["is_correct"]:
                        results["correct"] += 1
                        batch_correct += 1
                    if problem_result["extracted_answer"] is None:
                        results["no_response"] += 1

                    results["problems"].append(problem_result)

                    # Update progress bar
                    pbar.update(1)

                    # Update progress bar description with current accuracy
                    current_accuracy = results["correct"] / len(results["problems"])
                    pbar.set_postfix({
                        'accuracy': f'{current_accuracy:.1%}',
                        'correct': f'{results["correct"]}/{len(results["problems"])}'
                    })

                # Print batch summary
                self._print_batch_summary(
                    batch_idx, batch_correct, len(batch_problems), results["no_response"],
                    results["correct"], len(results["problems"])
                )

        eval_end_time = time.time()
        total_eval_time = eval_end_time - eval_start_time

        # Calculate accuracy
        accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
        results["accuracy"] = accuracy
        results["total_eval_time_seconds"] = total_eval_time

        self._print_final_results(results, total_eval_time, len(problems))

        return results


def evaluate_model_from_name(model_name: str, problems: List[Problem], max_new_tokens: int = 100,
                             batch_size: int = 1) -> Dict[str, Any]:
    """
    Evaluates a language model on a set of programming problems by loading the model
    from its name.

    :param model_name: The name or path of the pre-trained model to load from Hugging Face
                      model hub
    :param problems: List of prime inversion problems to evaluate the model against, each
                    containing test cases and expected outputs
    :param max_new_tokens: Maximum number of new tokens the model can generate for each
                          problem solution
    :param batch_size: Number of problems to process simultaneously in each evaluation batch
    :return: Dictionary containing evaluation results including accuracy
             metrics, timing information, and detailed per-problem analysis
    :raises ValueError: When model_name is empty or invalid
    :raises RuntimeError: When model loading fails
    :raises ConnectionError: When unable to download model from Hugging Face hub
    """

    print(f"🔄 Loading model: {model_name}")
    print(f"📊 Evaluation setup: {len(problems)} problems, batch_size={batch_size}, max_new_tokens={max_new_tokens}")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"✅ Model loaded successfully on device: {model.device}")
    print(f"🚀 Starting evaluation...")

    return evaluate_model(model, tokenizer, problems, max_new_tokens, batch_size)


def evaluate_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, problems: List[Problem],
                   max_new_tokens: int = 100,
                   batch_size: int = 1) -> Dict[str, Any]:
    """
    Evaluates a language model on a set of programming problems using an actual model and tokenizer

    :param model: The model under evaluation
    :param tokenizer: The tokenizer for the model under evaluation
    :param problems: List of prime inversion problems to evaluate the model against, each
                    containing test cases and expected outputs
    :param max_new_tokens: Maximum number of new tokens the model can generate for each
                          problem solution
    :param batch_size: Number of problems to process simultaneously in each evaluation batch
    :return: Dictionary containing evaluation results including accuracy
             metrics, timing information, and detailed per-problem analysis
    :raises ValueError: When model_name is empty or invalid
    :raises RuntimeError: When model loading fails
    :raises ConnectionError: When unable to download model from Hugging Face hub
    """
    eval_start_time = time.time()

    evaluator = Evaluator(model, tokenizer, max_new_tokens=max_new_tokens, batch_size=batch_size)
    return evaluator.evaluate(problems, eval_start_time)
