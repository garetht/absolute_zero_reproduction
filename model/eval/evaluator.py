import re
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, AutoTokenizer
from typing import List

from custom_types import ProblemResult, Problem, EvaluationResults
from model.args import AZRArgs
from model.eval.prime_inversion import is_prime
from model.inference import generate_response_bulk
from utils.string_formatting import extract_boxed_number, create_solver_prompt


class Evaluator:
    """Evaluator class for running model evaluations on the prime inversion problem"""

    def __init__(
            self,
            args: AZRArgs,
            model: AutoModelForCausalLM,
            tokenizer: PreTrainedTokenizerFast,
    ):
        """
        Initialize the evaluator with model configuration.

        Args:
            model: The AutoModelForCausalLM under evaluation
            tokenizer: The AutoTokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model.config.name_or_path
        self.batch_size = args.batch_size
        self.max_new_tokens = args.max_response_length
        self.args = args

    def _process_batch(self, batch_problems: List[Problem], batch_idx: int, total_batches: int) -> tuple[
        torch.Tensor, List[str], float]:

        """Process a single batch of problems."""
        batch_prompts = [create_solver_prompt(prob) for prob in batch_problems]

        print(
            f"\n📦 Processing batch {batch_idx + 1}/{total_batches} "
            f"(problems {batch_idx * self.batch_size + 1}-{min((batch_idx + 1) * self.batch_size, batch_idx * self.batch_size + len(batch_problems))})"
        )

        start_time = time.time()
        # Use generate_response_bulk instead of model.generate
        responses, logprobs, gen_ids, prompt_ids, attention_masks = generate_response_bulk(
            self.args,
            self.model,
            self.tokenizer,
            batch_prompts,
        )
        end_time = time.time()

        batch_time = end_time - start_time
        print(
            f"⏱️  Batch generation time: {batch_time:.2f}s ({batch_time / len(batch_problems):.2f}s per problem)"
        )

        # responses is a list of strings, already stripped of the prompt
        return responses, batch_prompts, batch_time

    def _process_problem_result(
            self,
            problem: Problem,
            model_response: str,
            batch_time: float,
            batch_size: int,
    ) -> ProblemResult:
        """Process the result for a single problem using the already stripped model_response."""
        # model_response is already the response (prompt stripped)
        extracted_answer = extract_boxed_number(model_response)
        correct_answer = problem.x if problem.blank == "x" else problem.y

        # Compare answers (considering modulo)
        is_correct = False
        if extracted_answer is not None:
            # Check if answers are equivalent modulo prime
            if problem.blank == "p":
                is_correct = (
                                     problem.x * problem.y
                             ) % extracted_answer == 1 and is_prime(extracted_answer)
            else:
                is_correct = (extracted_answer % problem.prime) == (
                        correct_answer % problem.prime
                )

        print(
            f"""{"✅" if is_correct else "⛔"} | Problem {problem} | Model Response {extracted_answer}"""
        )

        return {
            "problem": str(problem),
            "extracted_answer": extracted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "time_seconds": batch_time / batch_size,  # Approximate per-problem time
        }

    def _print_batch_summary(
            self,
            batch_idx: int,
            batch_correct: int,
            batch_size: int,
            no_responses: int,
            overall_correct: int,
            total_processed: int,
    ):
        """Print summary for the current batch."""
        batch_accuracy = batch_correct / batch_size
        print(
            f"📈 Batch {batch_idx + 1} results: {batch_correct}/{batch_size} correct ({batch_accuracy:.1%}) "
        )

        current_overall_accuracy = overall_correct / total_processed
        current_responded_accuracy = overall_correct / (total_processed - no_responses)

        print(
            f"📊 Overall progress: {overall_correct}/{total_processed} correct ({current_overall_accuracy:.1%}) ({current_responded_accuracy:.1%} of responded)"
        )

    def _print_final_results(
            self, results: EvaluationResults, total_eval_time: float, num_problems: int
    ):
        """Print final evaluation results."""
        print("\n🎯 Evaluation completed!")
        print("📊 Final Results:")
        print(f"   • Model: {self.model_name}")
        print(f"   • Total problems: {results['total']}")
        print(f"   • Correct answers: {results['correct']}")
        print(f"   • Overall accuracy: {results['accuracy']:.2%}")
        print(f"   • Total evaluation time: {total_eval_time:.1f}s")
        print(f"   • Average time per problem: {total_eval_time / num_problems:.2f}s")

    def evaluate(
            self, problems: List[Problem], eval_start_time: float = None
    ) -> EvaluationResults:
        """Run evaluation on the given problems."""
        if eval_start_time is None:
            eval_start_time = time.time()

        results: EvaluationResults = {
            "model": self.model_name,
            "correct": 0,
            "no_response": 0,
            "total": len(problems),
            "timestamp": datetime.now().isoformat(),
            "problem_results": [],
            "total_eval_time_seconds": 0.0,
            "accuracy": 0.0,
        }

        # Calculate total number of batches for progress tracking
        total_batches = (len(problems) + self.batch_size - 1) // self.batch_size

        # Process problems in batches with progress bar
        for batch_idx, i in enumerate(range(0, len(problems), self.batch_size)):
            batch_problems = problems[i: i + self.batch_size]

            # Use new _process_batch which returns responses
            responses, batch_prompts, batch_time = self._process_batch(
                batch_problems, batch_idx, total_batches
            )

            # Process each result in the batch
            batch_correct = 0
            for j, (problem, model_response) in enumerate(
                    zip(batch_problems, responses)
            ):
                problem_result = self._process_problem_result(
                    problem,
                    model_response,
                    batch_time,
                    len(batch_problems),
                )

                if problem_result["is_correct"]:
                    results["correct"] += 1
                    batch_correct += 1

                if problem_result["extracted_answer"] is None:
                    results["no_response"] += 1

                results["problem_results"].append(problem_result)

            # Print batch summary
            self._print_batch_summary(
                batch_idx,
                batch_correct,
                len(batch_problems),
                results["no_response"],
                results["correct"],
                len(results["problem_results"]),
            )

        eval_end_time = time.time()
        total_eval_time = eval_end_time - eval_start_time

        # Calculate accuracy
        accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
        results["accuracy"] = accuracy
        results["total_eval_time_seconds"] = total_eval_time

        self._print_final_results(results, total_eval_time, len(problems))

        return results


def evaluate_model_from_name(
        args: AZRArgs,
        model_name: str,
        problems: List[Problem],
) -> EvaluationResults:
    """
    Evaluates a language model on a set of programming problems by loading the model
    from its name.

    :param args:
    :param model_name: The name or path of the pre-trained model to load from Hugging Face
                      model hub
    :param problems: List of prime inversion problems to evaluate the model against, each
                    containing test cases and expected outputs
    :return: Dictionary containing evaluation results including accuracy
             metrics, timing information, and detailed per-problem analysis
    :raises ValueError: When model_name is empty or invalid
    :raises RuntimeError: When model loading fails
    :raises ConnectionError: When unable to download model from Hugging Face hub
    """

    print(f"🔄 Loading model: {model_name}")
    print(
        f"📊 Evaluation setup: {len(problems)} problems, batch_size={args.batch_size}, max_new_tokens={args.max_response_length}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    print(f"✅ Model loaded successfully on device: {model.device}")
    print("🚀 Starting evaluation...")

    return evaluate_model(args, model, tokenizer, problems)


def evaluate_model(
        args: AZRArgs,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerFast,
        problems: List[Problem],
) -> EvaluationResults:
    """
    Evaluates a language model on a set of programming problems using an actual model and tokenizer

    :param args: Parameters for generation
    :param model: The model under evaluation
    :param tokenizer: The tokenizer for the model under evaluation
    :param problems: List of prime inversion problems to evaluate the model against, each
                    containing test cases and expected outputs
    :return: Dictionary containing evaluation results including accuracy
             metrics, timing information, and detailed per-problem analysis
    :raises ValueError: When model_name is empty or invalid
    :raises RuntimeError: When model loading fails
    :raises ConnectionError: When unable to download model from Hugging Face hub
    """
    eval_start_time = time.time()

    evaluator = Evaluator(
        args, model, tokenizer
    )
    return evaluator.evaluate(problems, eval_start_time)
