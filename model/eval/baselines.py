import argparse
import random

from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_types import Problem, EvaluationResults, TaskType
from model.args import AZRArgs
from model.eval.evaluator import evaluate_model_from_name, evaluate_model
from model.eval.prime_inversion import generate_problems, PRIMES


def run_baseline_evaluation_prime_samples(args: AZRArgs, model: AutoModelForCausalLM,
                                          tokenizer: AutoTokenizer,
                                          problems: list[Problem]) -> EvaluationResults:
    """
    This function takes problems and then performs baseline evaluation using the specified model
     and parameters.

    :param args:
    :param model: Model being evaluated
    :param tokenizer: Tokenizer for the model being evaluated
    :param problems: Collection of problems to be evaluated
    :return: Dictionary containing evaluation results and metrics
    """

    print("running baseline eval!")
    for problem in problems:
        print(str(problem))

    return evaluate_model(
        args,
        model, tokenizer, problems
    )


def run_baseline_evaluation_random_problems(args: AZRArgs, model_name: str,
                                            num_problems: int = 100,
                                            first_prime_index: int = 7,
                                            last_prime_index: int = 20,
                                            seed: int = 42) -> EvaluationResults:
    """
    Executes baseline evaluation on a language model using randomly generated
    mathematical problems within a specified range of prime numbers. This function
    validates input parameters, generates a set of problems using primes from the
    specified index range, and runs the baseline evaluation process to assess model
    performance on mathematical reasoning tasks.

    :param args:
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
        args,
        model_name, generate_problems(n=num_problems, primes=PRIMES[first_prime_index:last_prime_index], seed=seed),
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

    azr_args = AZRArgs(batch_size=args.batch_size, max_response_length=args.max_new_tokens)

    baseline_evaluation_results = run_baseline_evaluation_random_problems(
        args=azr_args,
        model_name=args.model,
        num_problems=args.num_problems,
        seed=args.seed,
    )

    print(f"Evaluation complete for {args.model}")
    print(f"Accuracy: {baseline_evaluation_results['accuracy']:.2%}")
