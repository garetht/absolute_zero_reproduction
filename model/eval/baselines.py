import argparse
import random
from typing import Any

from custom_types import PrimeSample
from model.eval.evaluator import evaluate_model_from_name, evaluate_model
from model.eval.prime_inversion import generate_problems, PRIMES
from model.eval.problem import Problem
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_baseline_evaluation_prime_samples(model: AutoModelForCausalLM,
                                          tokenizer: AutoTokenizer,
                                          prime_samples: list[PrimeSample],
                                          max_new_tokens: int = 100,
                                          batch_size: int = 1,
                                          seed: int = 42) -> dict[str, Any]:
    """
    This function takes prime samples and then performs baseline evaluation using the specified model
     and parameters. The conversion process involves selecting either 'x' or 'y' as the variable
     name for each prime sample using the provided random seed for reproducibility.

    :param model: Model being evaluated
    :param tokenizer: Tokenizer for the model being evaluated
    :param prime_samples: Collection of prime samples to be evaluated
    :param max_new_tokens: Maximum number of new tokens to generate during evaluation
    :param batch_size: Number of problems to process in each batch
    :param seed: Random seed for reproducible variable name selection
    :return: Dictionary containing evaluation results and metrics
    """
    r = random.Random(seed)
    return evaluate_model(
        model, tokenizer, [Problem.from_prime_sample(ps, r.choice(['x', 'y'])) for ps in prime_samples], max_new_tokens,
        batch_size
    )


def run_baseline_evaluation_random_problems(model_name: str,
                                            num_problems: int = 100,
                                            first_prime_index: int = 7,
                                            last_prime_index: int = 20,
                                            max_new_tokens: int = 100,
                                            batch_size: int = 1,
                                            seed: int = 42) -> dict[str, Any]:
    """
    Executes baseline evaluation on a language model using randomly generated
    mathematical problems within a specified range of prime numbers. This function
    validates input parameters, generates a set of problems using primes from the
    specified index range, and runs the baseline evaluation process to assess model
    performance on mathematical reasoning tasks.

    :param model_name: Name or identifier of the language model to evaluate
    :param num_problems: Number of mathematical problems to generate for evaluation
    :param first_prime_index: The Nth prime to start considering from when generating
    :param last_prime_index: The Nth prime inclusive to stop considering from when generating. The maximum
    supported is 75
    :param max_new_tokens: Maximum number of tokens the model should generate per response
    :param batch_size: Number of problems to process simultaneously in each batch
    :param seed: Random seed value for reproducible problem generation
    :return: Dictionary containing evaluation results and performance metrics
    :raises ValueError: When first_prime_index is negative or exceeds PRIMES list bounds
    :raises ValueError: When last_prime_index is not greater than first_prime_index or
                       exceeds PRIMES list length
    """
    if first_prime_index < 0 or first_prime_index >= len(PRIMES):
        raise ValueError(f"first_prime_index must be between 0 and {len(PRIMES) - 1}, got {first_prime_index}")

    if last_prime_index <= first_prime_index or last_prime_index > len(PRIMES):
        raise ValueError(
            f"last_prime_index must be between {first_prime_index + 1} and {len(PRIMES)}, got {last_prime_index}")

    return evaluate_model_from_name(
        model_name, generate_problems(n=num_problems, primes=PRIMES[first_prime_index:last_prime_index], seed=seed),
        max_new_tokens,
        batch_size
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate language models on modular inverse problems",
        epilog="Usage example: PYTHONPATH=. python model/eval/baselines.py --model=Qwen/Qwen2.5-3B-Instruct --batch_size=10"
    )
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
