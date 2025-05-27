import argparse
import random
import re
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from custom_types import PrimeSample
from model.eval.prime_inversion import generate_problems, Problem, solve_modular_inverse
from model.eval.test_prime_inversion import PRIMES


def create_prompt(problem: Problem) -> str:
    """Create a prompt for the given problem."""
    if problem.blank == 'x':
        prompt = f"""
        Given a prime number p and an integer y, I need to find x such that:

x * {problem.y} ≡ 1 (mod {problem.prime})

For example, if p = 11 and y = 2, then 6 * 2 ≡ 1 (mod 11), so <answer>6</answer> is the answer.
For example, if p = 13 and y = 5, then 10 * 5 ≡ 1 (mod 13), so <answer>10</answer> is the answer.

Provide your answer as a single unformatted number within <answer></answer> tags, e.g. <answer>x</answer>
"""
    else:  # problem.blank == 'y'
        prompt = f"""Given a prime number p and an integer x, I need to find y such that:

{problem.x} * y ≡ 1 (mod {problem.prime})

For example, if p = 11 and x = 6, then 6 * 2 ≡ 1 (mod 11), so <answer>2</answer> is the answer.
For example, if p = 13 and x = 10, then 10 * 5 ≡ 1 (mod 13), so <answer>5</answer> is the answer.

Provide your answer as a single unformatted number within <answer></answer> tags, e.g. <answer>x</answer>
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
    eval_start_time = time.time()
    
    print(f"🔄 Loading model: {model_name}")
    print(f"📊 Evaluation setup: {len(problems)} problems, batch_size={batch_size}, max_new_tokens={max_new_tokens}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    print(f"✅ Model loaded successfully on device: {model.device}")
    print(f"🚀 Starting evaluation...")

    results = {
        "model": model_name,
        "correct": 0,
        "total": len(problems),
        "timestamp": datetime.now().isoformat(),
        "problems": []
    }

    # Calculate total number of batches for progress tracking
    total_batches = (len(problems) + batch_size - 1) // batch_size
    
    # Process problems in batches with progress bar
    with tqdm(total=len(problems), desc="Evaluating problems", unit="problem") as pbar:
        for batch_idx, i in enumerate(range(0, len(problems), batch_size)):
            batch_problems = problems[i:i + batch_size]
            batch_prompts = [create_prompt(prob) for prob in batch_problems]
            
            print(f"\n📦 Processing batch {batch_idx + 1}/{total_batches} (problems {i + 1}-{min(i + batch_size, len(problems))})")

            # Tokenize all prompts in the batch
            batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

            start_time = time.time()
            with torch.no_grad():
                batch_outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens
                )
            end_time = time.time()
            
            batch_time = end_time - start_time
            print(f"⏱️  Batch generation time: {batch_time:.2f}s ({batch_time/len(batch_problems):.2f}s per problem)")

            # Process each result in the batch
            batch_correct = 0
            for j, (problem, outputs) in enumerate(zip(batch_problems, batch_outputs)):
                response = tokenizer.decode(outputs, skip_special_tokens=True)
                print("the model's response is: ", response)

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
                    batch_correct += 1

                # Save problem results
                problem_result = {
                    "problem": problem.desc,
                    "extracted_answer": extracted_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "time_seconds": batch_time / len(batch_problems)  # Approximate per-problem time
                }

                if save_responses:
                    problem_result["prompt"] = batch_prompts[j]
                    problem_result["response"] = model_response

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
            batch_accuracy = batch_correct / len(batch_problems)
            print(f"📈 Batch {batch_idx + 1} results: {batch_correct}/{len(batch_problems)} correct ({batch_accuracy:.1%})")
            
            # Print current overall accuracy
            current_overall_accuracy = results["correct"] / len(results["problems"])
            print(f"📊 Overall progress: {results['correct']}/{len(results['problems'])} correct ({current_overall_accuracy:.1%})")

    eval_end_time = time.time()
    total_eval_time = eval_end_time - eval_start_time

    # Calculate accuracy
    accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
    results["accuracy"] = accuracy
    results["total_eval_time_seconds"] = total_eval_time

    print(f"\n🎯 Evaluation completed!")
    print(f"📊 Final Results:")
    print(f"   • Model: {model_name}")
    print(f"   • Total problems: {results['total']}")
    print(f"   • Correct answers: {results['correct']}")
    print(f"   • Overall accuracy: {accuracy:.2%}")
    print(f"   • Total evaluation time: {total_eval_time:.1f}s")
    print(f"   • Average time per problem: {total_eval_time/len(problems):.2f}s")
    
    return results


def run_baseline_evaluation_prime_samples(model_name: str, problems: list[PrimeSample],
                                          max_new_tokens: int = 100, batch_size: int = 1, seed: int = 42):
    r = random.Random(seed)
    return run_baseline_evaluation(
        model_name, [Problem.from_prime_sample(ps, r.choice(['x', 'y'])) for ps in problems], max_new_tokens,
        batch_size
    )


def run_baseline_evaluation_random_problems(model_name: str, num_problems: int = 100,
                                            max_new_tokens: int = 100, batch_size: int = 1, seed: int = 42):
    return run_baseline_evaluation(
        model_name, generate_problems(n=num_problems, primes=PRIMES[7:20], seed=seed), max_new_tokens,
        batch_size
    )


def run_baseline_evaluation(model_name: str, problems: list[Problem],
                            max_new_tokens: int = 100, batch_size: int = 1):
    return evaluate_model(
        model_name=model_name,
        problems=problems,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language models on modular inverse problems")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="HuggingFace model name")
    parser.add_argument("--num_problems", type=int, default=20, help="Number of problems to generate")
    parser.add_argument("--max-new-tokens", type=int, default=500, help="Number of problems to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")

    args = parser.parse_args()

    results = run_baseline_evaluation_random_problems(
        model_name=args.model,
        num_problems=args.num_problems,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Evaluation complete for {args.model}")
    print(f"Accuracy: {results['accuracy']:.2%}")
