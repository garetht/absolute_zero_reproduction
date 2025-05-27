import re
import time
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional

from model.eval.prime_inversion import Problem
from model.eval.prompts import create_prompt


class Evaluator:
    """Evaluator class for running model evaluations on mathematical problems."""

    def __init__(self, model_name: str, max_new_tokens: int = 100, batch_size: int = 1):
        """
        Initialize the evaluator with model configuration.

        Args:
            model_name: Name/path of the model to evaluate
            max_new_tokens: Maximum number of new tokens to generate
            batch_size: Number of problems to process in each batch
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        print(f"📥 Loading model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _process_batch(self, batch_problems: List[Problem], batch_idx: int, total_batches: int) -> tuple:
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
                                prompt: str, batch_time: float, batch_size: int) -> Dict[str, Any]:
        """Process the result for a single problem."""
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        print("the model's response is: ", response)

        # Get the part of the response that comes after the prompt
        model_response = response[len(prompt):]

        # Extract answer
        extracted_answer = self.extract_boxed_number(model_response)
        correct_answer = problem.x if problem.blank == 'x' else problem.y

        # Compare answers (considering modulo)
        is_correct = False
        if extracted_answer is not None:
            # Check if answers are equivalent modulo prime
            is_correct = (extracted_answer % problem.prime) == (correct_answer % problem.prime)

        return {
            "problem": problem.desc,
            "extracted_answer": extracted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "time_seconds": batch_time / batch_size  # Approximate per-problem time
        }

    def _print_batch_summary(self, batch_idx: int, batch_correct: int, batch_size: int,
                             overall_correct: int, total_processed: int):
        """Print summary for the current batch."""
        batch_accuracy = batch_correct / batch_size
        print(f"📈 Batch {batch_idx + 1} results: {batch_correct}/{batch_size} correct ({batch_accuracy:.1%})")

        current_overall_accuracy = overall_correct / total_processed
        print(f"📊 Overall progress: {overall_correct}/{total_processed} correct ({current_overall_accuracy:.1%})")

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

        if self.model is None or self.tokenizer is None:
            self.load_model()

        results = {
            "model": self.model_name,
            "correct": 0,
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
                    batch_idx, batch_correct, len(batch_problems),
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
    """Evaluate a model on the prime inversion problems."""
    eval_start_time = time.time()

    print(f"🔄 Loading model: {model_name}")
    print(f"📊 Evaluation setup: {len(problems)} problems, batch_size={batch_size}, max_new_tokens={max_new_tokens}")

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    print(f"✅ Model loaded successfully on device: {model.device}")
    print(f"🚀 Starting evaluation...")

    evaluator = Evaluator(model_name, max_new_tokens=max_new_tokens, batch_size=batch_size)
    return evaluator.evaluate(problems, eval_start_time)
