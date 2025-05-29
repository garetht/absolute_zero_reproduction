import random
from typing import Optional

from custom_types import Problem, TaskType


def is_prime(n: int) -> bool:
    """
    Phenomenally efficient prime checking function.
    Uses multiple optimization techniques for maximum performance.
    """
    # Handle small cases first (most common)
    if n < 2:
        return False
    if n < 4:
        return True  # 2 and 3 are prime
    if n % 2 == 0 or n % 3 == 0:
        return False

    # Use 6k±1 optimization - all primes > 3 are of form 6k±1
    # This reduces iterations by ~66% compared to checking all odd numbers
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6

    return True


def solve_modular_inverse(p: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None,
                          verbose: bool = False) -> set[int]:
    """
    Solves for the unknown in xy ≡ 1 mod p, where p is prime, and one of x or y is given.

    Args:
        p (int): A prime modulus.
        x (int, optional): The value of x. If None, y must be given.
        y (int, optional): The value of y. If None, x must be given.
        verbose (bool, optional): Whether to print solving steps.

    Returns:
        set[int]: All answers that will satisfy the problem. Only missing primes
        will have multiple possible solutions.
    """
    values = [p, x, y]

    none_count = sum(1 for val in values if val is None)
    if none_count > 1:
        raise ValueError("At most one of p, x, y can be None")

    if x is not None and p is not None:
        if not is_prime(p):
            return set()

        # Compute modular inverse of x mod p
        result = pow(x, -1, p)
        if verbose:
            print(f"Given x = {x}, solving for y such that x*y ≡ 1 mod {p}: y = {result}")
        return {result}
    elif y is not None and p is not None:
        # Compute modular inverse of y mod p
        result = pow(y, -1, p)
        if verbose:
            print(f"Given y = {y}, solving for x such that x*y ≡ 1 mod {p}: x = {result}")
        return {result}
    elif x is not None and y is not None:
        satisfying_primes = set()
        for prime in PRIMES:
            if (x * y) % prime == 1:
                satisfying_primes.add(prime)

        return satisfying_primes

    raise ValueError("No values were provided")

def modular_inverse(a: int, p: int) -> int:
    # Using Fermat's little theorem to compute modular inverse: a^(p-2) mod p
    return pow(a, p - 2, p)


def validate_modular_inverse(x: int, y: int, p: int) -> bool:
    return (x * y) % p == 1


def generate_problems(n: int, primes: list[int], seed: int = 42) -> list[Problem]:
    """
    Generate a list of modular inverse problems for evaluation.

    This function creates n problems where each problem involves finding either x or y
    in the equation xy ≡ 1 (mod p), where p is a prime number. For each problem,
    one variable is randomly chosen to be the unknown (blank), and the other is given.

    Args:
        n (int): The number of problems to generate.
        primes (list[int]): A list of prime numbers to choose from as moduli.
        seed (int, optional): Random seed for reproducible problem generation. Defaults to 42.

    Returns:
        list[Problem]: A list of Problem instances, each containing:
            - prime: The prime modulus
            - x: The x value (or None if x is the unknown)
            - y: The y value (or None if y is the unknown)
            - blank: Either 'x' or 'y' indicating which variable to solve for
            - desc: A human-readable description of the problem

    Example:
        >>> primes = [7, 11, 13]
        >>> problems = generate_problems(2, primes, seed=123)
        >>> len(problems)
        2
        >>> problems[0].prime in primes
        True
    """
    r = random.Random(seed)
    problems = []
    for _ in range(n):
        p = r.choice(primes)
        x = r.randint(1, p - 1)
        y = modular_inverse(x, p)
        task_type = r.choice(list(TaskType))
        prob = Problem(prime=p, x_list=[x], y_list=[y], task_type=task_type)
        problems.append(prob)
    return problems


PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
    113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
    241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
    383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521,
    523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617
]
