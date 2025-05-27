import argparse
import random


from custom_types import PrimeSample
from model.eval.evaluator import Evaluator, evaluate_model_from_name
from model.eval.prime_inversion import generate_problems, Problem
from model.eval.test_prime_inversion import PRIMES


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
    return evaluate_model_from_name(
        model_name=model_name,
        problems=problems,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language models on modular inverse problems")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model name")
    parser.add_argument("--num_problems", type=int, default=20, help="Number of problems to generate")
    parser.add_argument("--max-new-tokens", type=int, default=500, help="Number of problems to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")

    args = parser.parse_args()

    baseline_evaluation_results = run_baseline_evaluation_random_problems(
        model_name=args.model,
        num_problems=args.num_problems,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Evaluation complete for {args.model}")
    print(f"Accuracy: {baseline_evaluation_results['accuracy']:.2%}")
