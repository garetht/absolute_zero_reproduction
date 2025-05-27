import argparse
import random
import re
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional

from custom_types import PrimeSample
from model.eval.prime_inversion import generate_problems, Problem, solve_modular_inverse


def create_prompt(problem: Problem) -> str:
    """Create a prompt for the given problem."""
    if problem.blank == 'x':
        prompt = f"""I'm working with modular arithmetic. Given a prime number p and an integer y, I need to find x such that:

x * {problem.y} ≡ 1 (mod {problem.prime})

In other words, x is the modular inverse of {problem.y} modulo {problem.prime}.
What is the value of x?

Please provide your answer within <answer></answer> tags.
"""
    else:  # problem.blank == 'y'
        prompt = f"""I'm working with modular arithmetic. Given a prime number p and an integer x, I need to find y such that:

{problem.x} * y ≡ 1 (mod {problem.prime})

In other words, y is the modular inverse of {problem.x} modulo {problem.prime}.
What is the value of y?

Please provide your answer within <answer></answer> tags.
"""
    return prompt


def extract_answer(response: str) -> Optional[int]:
    """Extract the answer from the model's response."""
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        try:
            # Try to extract and convert to int
            answer_text = match.group(1).strip()
            # Handle potential expressions or calculations
            if "=" in answer_text:
                # If there's an equality, take the right side
                answer_text = answer_text.split("=")[-1].strip()
            return int(answer_text)
        except ValueError:
            return None
    return None


def evaluate_model(model_name: str, problems: List[Problem], max_new_tokens: int = 100,
                   batch_size: int = 1, save_responses: bool = True) -> Dict[str, Any]:
    """Evaluate a model on the prime inversion problems."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    results = {
        "model": model_name,
        "correct": 0,
        "total": len(problems),
        "timestamp": datetime.now().isoformat(),
        "problems": []
    }

    # Process problems in batches
    for i in range(0, len(problems), batch_size):
        batch_problems = problems[i:i + batch_size]
        batch_prompts = [create_prompt(prob) for prob in batch_problems]

        # Tokenize all prompts in the batch
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

        start_time = time.time()
        with torch.no_grad():
            batch_outputs = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        end_time = time.time()

        # Process each result in the batch
        for j, (problem, outputs) in enumerate(zip(batch_problems, batch_outputs)):
            response = tokenizer.decode(outputs, skip_special_tokens=True)

            # Get the part of the response that comes after the prompt
            model_response = response[len(batch_prompts[j]):]

            # Extract answer
            extracted_answer = extract_answer(model_response)

            # Calculate correct answer
            if problem.blank == 'x':
                correct_answer = solve_modular_inverse(p=problem.prime, x=None, y=problem.y, verbose=False)
            else:  # problem.blank == 'y'
                correct_answer = solve_modular_inverse(p=problem.prime, x=problem.x, y=None, verbose=False)

            # Compare answers (considering modulo)
            is_correct = False
            if extracted_answer is not None:
                # Check if answers are equivalent modulo prime
                is_correct = (extracted_answer % problem.prime) == (correct_answer % problem.prime)

            if is_correct:
                results["correct"] += 1

            # Save problem results
            problem_result = {
                "problem": problem.desc,
                "extracted_answer": extracted_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "time_seconds": (end_time - start_time) / batch_size  # Approximate per-problem time
            }

            if save_responses:
                problem_result["prompt"] = batch_prompts[j]
                problem_result["response"] = model_response

            results["problems"].append(problem_result)

            # Print result
            prob_idx = i + j
            print(f"Problem {prob_idx + 1}/{len(problems)}: {problem.desc}")
            print(f"  Extracted: {extracted_answer}")
            print(f"  Correct: {correct_answer}")
            print(f"  Result: {'✓' if is_correct else '✗'}")
            print(f"  Time: {(end_time - start_time) / batch_size:.2f}s per problem")
            print()

    # Calculate accuracy
    accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
    results["accuracy"] = accuracy

    print(f"Overall accuracy: {accuracy:.2%}")
    return results


def run_baseline_evaluation_prime_samples(model_name: str, problems: list[PrimeSample],
                                          max_new_tokens: int = 100, batch_size: int = 1, seed: int = 42):
    r = random.Random(seed)
    return run_baseline_evaluation_problems(
        model_name, [Problem.from_prime_sample(ps, r.choice(['x', 'y'])) for ps in problems], max_new_tokens,
        batch_size
    )


def run_baseline_evaluation_problems(model_name: str, problems: list[Problem],
                                     max_new_tokens: int = 100, batch_size: int = 1):
    pass


def run_baseline_evaluation(model_name: str, problems: list[Problem],
                            max_new_tokens: int = 100, batch_size: int = 1) -> Dict[str, Any]:
    """Run the baseline evaluation for a specified model."""
    if primes is None:
        primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    # Generate problems
    problems = generate_problems(num_problems, primes, seed)

    # Evaluate model
    return evaluate_model(
        model_name=model_name,
        problems=problems,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language models on modular inverse problems")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--num_problems", type=int, default=20, help="Number of problems to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens for generation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--no_save_responses", action="store_true", help="Do not save full model responses")

    args = parser.parse_args()

    results = run_baseline_evaluation(
        model_name=args.model,
        num_problems=args.num_problems,
        seed=args.seed,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        save_responses=not args.no_save_responses
    )

    print(f"Evaluation complete for {args.model}")
    print(f"Accuracy: {results['accuracy']:.2%}")
